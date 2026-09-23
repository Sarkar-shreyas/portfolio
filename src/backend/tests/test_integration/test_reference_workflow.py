"""End-to-end integration test for the library's reference workflow.

This is the one test that composes the whole library the way a research
notebook would, and checks that the pieces actually fit:

    market data -> returns -> signals -> portfolio weights
                -> walk-forward backtest -> transaction costs
                -> portfolio returns -> risk metrics

Every stage calls the real production function through its public API. Nothing
here reimplements library logic; the numerical assertions are either closed-form
(a cost charged at a known bps), identities that must hold between two stages
(equity vs. compounded returns), or independent qualitative expectations
(higher costs must hurt).

The unit suites already cover each function in isolation. What this test exists
to catch is the class of bug that only appears when the modules are wired
together: an index that silently widens between stages, weights that stop
lining up with the tickers they were built from, look-ahead creeping in through
the fold boundaries, or costs that quietly fail to reach the P&L.
"""

import numpy as np
import pandas as pd
import pytest

from src.backend.analysis.returns import simple_returns, cumulative_returns, ann_returns
from src.backend.analysis.risk_metrics import ann_sharpe, max_drawdown, rolling_sharpe
from src.backend.analysis.var import cond_var, est_var
from src.backend.analysis.vol import ann_volatility, rolling_volatility
from src.backend.portfolio_construction.dynamic_weights import (
    equal_active_weights,
    equal_split_ls_weights,
    inverse_volatility_weighted,
)
from src.backend.strategy.trend_following import sma_crossover_portfolio


# ---------------------------------------------------------------------------
# The workflow itself, built once and inspected from several angles below
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow(config, close_panel, make_backtester, signal_windows):
    """Run the reference workflow end to end and return every intermediate.

    Steps 1-7 of the reference workflow. Step 8 (risk metrics) is exercised by
    the tests that consume ``net_returns``.
    """
    short, long = signal_windows

    # 2. prices -> returns
    returns = simple_returns(close_panel)

    # 3. prices -> signals
    signals = sma_crossover_portfolio(config, close_panel, short, long)

    # 4. signals -> weights (standalone, i.e. the notebook view of this stage)
    weights = equal_active_weights(config, signals)

    # 5/6. walk-forward backtest with transaction costs
    backtester = make_backtester()
    results = backtester.run([short, long], [])
    run = backtester.runs[1]

    return {
        "returns": returns,
        "signals": signals,
        "weights": weights,
        "backtester": backtester,
        "results": results,
        "run": run,
        "net_returns": run["net_returns"],
        "equity": run["equity_curve"],
        "oos_weights": run["oos_weights"],
        "rebalance_dates": run["fold_costs"].index,
    }


# ---------------------------------------------------------------------------
# Step 1-2: market data -> returns
# ---------------------------------------------------------------------------


def test_returns_preserve_the_price_panel_layout(workflow, close_panel, tickers):
    """Tickers stay attached to their own data and dates stay aligned."""
    returns = workflow["returns"]

    assert isinstance(returns, pd.DataFrame)
    pd.testing.assert_index_equal(returns.index, close_panel.index)
    pd.testing.assert_index_equal(returns.columns, close_panel.columns)
    assert list(returns.columns) == tickers


def test_returns_handle_the_first_observation(workflow):
    """The first bar has no prior close, so it is NaN and nothing else is."""
    returns = workflow["returns"]

    assert returns.iloc[0].isna().all()
    assert returns.iloc[1:].notna().all().all()


# ---------------------------------------------------------------------------
# Step 3: returns -> signals
# ---------------------------------------------------------------------------


def test_signals_cover_the_same_universe_and_dates(workflow, close_panel):
    signals = workflow["signals"]

    pd.testing.assert_index_equal(signals.index, close_panel.index)
    pd.testing.assert_index_equal(signals.columns, close_panel.columns)


def test_signals_are_discrete_positions(workflow):
    """A signal is a position instruction: long, flat or short."""
    signals = workflow["signals"]

    assert not signals.isnull().any().any()
    assert set(np.unique(signals.to_numpy())) <= {-1, 0, 1}


# ---------------------------------------------------------------------------
# Step 4: signals -> portfolio weights
# ---------------------------------------------------------------------------


def test_weights_keep_the_signal_layout(workflow):
    """The weight stage must not widen or reorder the panel.

    Alignment here is load-bearing: the backtester reads the weight row at the
    last training date, so a weight function that returns a different index
    hands it the wrong row.
    """
    signals, weights = workflow["signals"], workflow["weights"]

    pd.testing.assert_index_equal(weights.index, signals.index)
    pd.testing.assert_index_equal(weights.columns, signals.columns)


def test_inactive_positions_carry_no_weight(workflow):
    signals, weights = workflow["signals"], workflow["weights"]

    inactive = signals.to_numpy() == 0
    assert inactive.any()
    assert (weights.to_numpy()[inactive] == 0).all()


def test_weights_respect_the_configured_exposure(workflow, config):
    """Gross exposure matches the specification on every active row."""
    signals, weights = workflow["signals"], workflow["weights"]
    active = signals.ne(0).any(axis=1)

    gross = weights.abs().sum(axis=1)
    assert active.any() and (~active).any()
    assert gross[active].to_numpy() == pytest.approx(config.tot_exposure)
    assert (gross[~active] == 0).all()


def test_weights_follow_the_direction_of_the_signal(workflow):
    """A long signal must produce a long weight, a short signal a short one."""
    signals, weights = workflow["signals"], workflow["weights"]

    assert (np.sign(weights) == np.sign(signals)).all().all()


def test_inverse_volatility_weights_also_keep_the_signal_layout(
    workflow, config, close_panel
):
    """The vol-scaled construction must align to the signals, not to the vols.

    ``inverse_volatility_weighted`` takes a second frame, so it is the stage
    where an index union can silently widen the result. When that happens the
    backtester's target row falls outside the training fold and the book ends up
    empty, with no error raised anywhere.
    """
    signals = workflow["signals"]
    vols = rolling_volatility(config, workflow["returns"], 21)

    # A training sub-period, exactly as the backtester passes it.
    fold_signals = signals.iloc[:130]
    weights = inverse_volatility_weighted(config, fold_signals, vols)

    pd.testing.assert_index_equal(weights.index, fold_signals.index)
    pd.testing.assert_index_equal(weights.columns, fold_signals.columns)
    assert weights.iloc[-1].abs().sum() > 0


def test_volatility_estimates_are_non_negative(workflow, config):
    """Inverse-vol weighting is only meaningful for a positive denominator."""
    vols = rolling_volatility(config, workflow["returns"], 21)

    assert (vols.dropna() >= 0).all().all()


# ---------------------------------------------------------------------------
# Step 5: the walk-forward backtest
# ---------------------------------------------------------------------------


def test_training_folds_strictly_precede_their_test_folds(workflow):
    """No fold may train on a bar it is about to be tested on."""
    folds = workflow["backtester"]._generate_folds()

    assert folds
    for train_fold, test_fold in folds:
        assert len(train_fold) > 0 and len(test_fold) > 0
        assert train_fold.index.max() < test_fold.index.min()


def test_test_folds_tile_the_out_of_sample_region_exactly_once(
    workflow, close_panel, train_frac
):
    folds = workflow["backtester"]._generate_folds()
    covered = pd.DatetimeIndex([]).append([test.index for _, test in folds])

    expected = close_panel.index[int(len(close_panel) * train_frac) :]
    pd.testing.assert_index_equal(covered, expected)


def test_backtest_output_is_aligned_and_ordered(
    workflow, close_panel, tickers, train_frac
):
    """Equity, returns and weights all land on the out-of-sample dates."""
    equity = workflow["equity"]
    net_returns = workflow["net_returns"]
    oos_weights = workflow["oos_weights"]

    expected_index = close_panel.index[int(len(close_panel) * train_frac) :]

    pd.testing.assert_index_equal(equity.index, expected_index)
    pd.testing.assert_index_equal(net_returns.index, expected_index)
    pd.testing.assert_index_equal(oos_weights.index, expected_index)
    assert list(oos_weights.columns) == tickers

    assert equity.index.is_monotonic_increasing
    assert equity.index.is_unique
    assert np.isfinite(net_returns).all()
    assert (equity > 0).all()


def _corrupt_future(prices: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``prices`` whose path after ``after`` is rewritten.

    Applies a decaying ramp rather than a flat multiplier. A crossover signal
    compares two moving averages of the same series, so scaling every price by a
    constant leaves it completely unchanged -- a uniform perturbation would test
    only the valuation path and quietly miss look-ahead in the *decisions*. The
    ramp moves the short average relative to the long one, so any use of future
    prices shows up as a different signal.
    """
    perturbed = prices.copy()
    mask = perturbed.index > after
    ramp = np.linspace(1.0, 0.4, int(mask.sum()))
    perturbed.loc[mask] = perturbed.loc[mask].mul(ramp, axis=0)
    return perturbed


def test_walk_forward_is_free_of_look_ahead(workflow, close_panel, make_backtester):
    """Results up to a date must not move when later prices change.

    Rewrites every price after the first fold's test window and re-runs. The
    first fold's equity path is driven only by data at or before its own dates,
    so it has to come back bit-identical.
    """
    baseline_equity = workflow["equity"]
    folds = workflow["backtester"]._generate_folds()
    first_fold_end = folds[0][1].index[-1]

    perturbed = _corrupt_future(close_panel, first_fold_end)

    shifted = make_backtester(price_data=perturbed)
    shifted.run(list(workflow["run"]["signal_args"]), [])
    shifted_equity = shifted.runs[1]["equity_curve"]

    pd.testing.assert_series_equal(
        shifted_equity.loc[:first_fold_end], baseline_equity.loc[:first_fold_end]
    )


def test_rebalance_decisions_use_only_prior_data(
    workflow, close_panel, make_backtester, config, signal_windows
):
    """The book put on at a rebalance must be decided before that bar's future.

    Rewrites every price strictly after the first rebalance. That trade is
    decided from the training fold and executed at the trade date's own close,
    so neither its target nor its traded size may move. This is the behavioural
    half of the look-ahead check: the test above pins the *valuation*, this one
    pins the *decision*, which is what a widened training window corrupts.
    """
    baseline = workflow["run"]
    first_trade_date = workflow["rebalance_dates"][0]

    perturbed = _corrupt_future(close_panel, first_trade_date)
    # The perturbation has to be able to move a signal, or this proves nothing.
    short, long = signal_windows
    assert not sma_crossover_portfolio(config, perturbed, short, long).equals(
        workflow["signals"]
    )

    shifted = make_backtester(price_data=perturbed)
    shifted.run(list(workflow["run"]["signal_args"]), [])
    shifted_run = shifted.runs[1]

    pd.testing.assert_series_equal(
        shifted_run["target_shares"].iloc[0], baseline["target_shares"].iloc[0]
    )
    pd.testing.assert_series_equal(
        shifted_run["trade_shares"].iloc[0], baseline["trade_shares"].iloc[0]
    )
    assert shifted_run["fold_costs"].iloc[0] == pytest.approx(
        baseline["fold_costs"].iloc[0]
    )


# ---------------------------------------------------------------------------
# Steps 6-7: costs and the resulting portfolio returns
# ---------------------------------------------------------------------------


def test_equity_curve_and_net_returns_are_consistent(workflow, config):
    """The two headline outputs must describe the same path."""
    equity = workflow["equity"]
    net_returns = workflow["net_returns"]

    rebuilt = config.start_capital * (1 + net_returns).cumprod()
    pd.testing.assert_series_equal(rebuilt, equity, check_names=False)

    assert workflow["results"]["final_equity"] == pytest.approx(equity.iloc[-1])
    assert workflow["results"]["cum_return"] == pytest.approx(
        equity.iloc[-1] / config.start_capital - 1
    )
    assert cumulative_returns(net_returns).iloc[-1] == pytest.approx(
        workflow["results"]["cum_return"]
    )


def test_first_out_of_sample_return_is_exactly_the_entry_cost(workflow, config):
    """Independent closed form for the opening bar.

    The book is bought at the first OOS close, so nothing has moved yet and the
    only P&L is the transaction cost. This pins that costs reach the return
    series at the right magnitude, not merely that they are subtracted somewhere.
    """
    first_cost = workflow["run"]["fold_costs"].iloc[0]

    assert first_cost > 0
    assert workflow["net_returns"].iloc[0] == pytest.approx(
        -first_cost / config.start_capital, abs=1e-12
    )


def test_portfolio_return_is_the_weighted_sum_of_asset_returns(workflow):
    """Cross-module identity tying weights, asset returns and portfolio P&L.

    Between rebalances the holding is untouched, so the portfolio return on a
    non-trading day must equal the previous day's weights dotted with that day's
    asset returns. This is the check that the backtester's own weights, its own
    returns and the returns produced by the analysis layer all agree.
    """
    net_returns = workflow["net_returns"]
    oos_weights = workflow["oos_weights"]
    asset_returns = workflow["returns"].reindex(oos_weights.index)

    implied = (oos_weights.shift(1) * asset_returns).sum(axis=1)

    held = ~net_returns.index.isin(workflow["rebalance_dates"])
    held[0] = False  # no previous weight row to carry in

    assert held.sum() > 0
    np.testing.assert_allclose(
        net_returns[held].to_numpy(), implied[held].to_numpy(), atol=1e-12
    )


def _realised_gross(workflow) -> np.ndarray:
    """Gross exposure of the book actually held at each rebalance."""
    oos_weights = workflow["oos_weights"]
    return np.array(
        [oos_weights.loc[date].abs().sum() for date in workflow["rebalance_dates"]]
    )


def test_rebalances_deliver_the_requested_gross_exposure(
    workflow, make_backtester, config
):
    """A free rebalance puts on exactly the exposure that was asked for.

    Shares are sized off the book marked to market on the rebalance bar, so with
    no transaction cost to pay the realised gross exposure is the target to
    machine precision. Removing costs isolates the sizing from the one thing
    that is legitimately allowed to move it, which makes this an exact assertion
    rather than a tolerance calibrated to one random sample.
    """
    free = make_backtester(cost_bps=0.0)
    free.run(list(workflow["run"]["signal_args"]), [])
    run = free.runs[1]
    oos_weights = run["oos_weights"]

    for date in run["fold_costs"].index:
        assert oos_weights.loc[date].abs().sum() == pytest.approx(
            config.tot_exposure, abs=1e-12
        )


def test_transaction_costs_are_the_only_drag_on_the_targeted_exposure(
    workflow, config
):
    """With costs charged, the exposure shortfall is exactly the cost drag.

    Equity on the rebalance bar is the marked book less that bar's cost, so the
    realised gross is

        tot_exposure * capital / (capital - cost)

    Pinning the identity ties the sizing to the cost model and keeps the test
    independent of the sample length and the seed.
    """
    run = workflow["run"]
    capital = run["fold_capital"].to_numpy()
    costs = run["fold_costs"].to_numpy()

    assert (costs > 0).all()
    np.testing.assert_allclose(
        _realised_gross(workflow),
        config.tot_exposure * capital / (capital - costs),
        atol=1e-12,
    )


def test_exposure_shortfall_does_not_accumulate(workflow, config):
    """The shortfall stays at the size of one fold's cost and never compounds.

    Being off by the cost charged on the bar is unavoidable. An error that grew
    with each rebalance would be a real defect, so the bound comes from the run's
    own costs rather than from a constant.
    """
    run = workflow["run"]
    cost_fraction = (run["fold_costs"] / run["fold_capital"]).to_numpy()
    deviation = np.abs(_realised_gross(workflow) - config.tot_exposure)

    assert (cost_fraction < 1).all()
    assert (
        deviation <= config.tot_exposure * cost_fraction / (1 - cost_fraction) + 1e-12
    ).all()


def test_transaction_costs_degrade_performance_monotonically(workflow, make_backtester):
    """Charging more must always leave the investor worse off."""
    equities = []
    for bps in (0.0, 10.0, 50.0):
        backtester = make_backtester(cost_bps=bps)
        results = backtester.run(list(workflow["run"]["signal_args"]), [])
        equities.append(results["final_equity"])
        # A zero-turnover book would make the comparison vacuous.
        assert results["total_turnover"] > 0

    assert equities[0] > equities[1] > equities[2]


def test_costs_are_charged_at_the_configured_rate(workflow, make_backtester):
    """Doubling the bps must double the cost charged on the same trades."""
    cheap = make_backtester(cost_bps=10.0)
    cheap.run(list(workflow["run"]["signal_args"]), [])
    dear = make_backtester(cost_bps=20.0)
    dear.run(list(workflow["run"]["signal_args"]), [])

    first_cheap = cheap.runs[1]["fold_costs"].iloc[0]
    first_dear = dear.runs[1]["fold_costs"].iloc[0]

    assert first_dear == pytest.approx(2 * first_cheap)


def test_zero_cost_backtest_still_trades(workflow, make_backtester):
    """A free book must differ from a charged one, and still turn over."""
    free = make_backtester(cost_bps=0.0)
    results = free.run(list(workflow["run"]["signal_args"]), [])

    assert (free.runs[1]["fold_costs"] == 0).all()
    assert results["total_turnover"] > 0
    assert results["final_equity"] != workflow["results"]["final_equity"]


def test_changing_the_weight_function_changes_the_returns(workflow, make_backtester):
    """Portfolio construction has to actually reach the P&L."""
    split = make_backtester(weight_fn=equal_split_ls_weights)
    split_results = split.run(list(workflow["run"]["signal_args"]), [])

    assert split_results["final_equity"] != workflow["results"]["final_equity"]
    assert not split.runs[1]["net_returns"].equals(workflow["net_returns"])


def test_an_empty_book_earns_nothing(workflow, make_backtester, config):
    """Control: with no positions there is no turnover, no cost and no P&L."""

    def no_positions(cfg, signals, *args):
        return signals * 0.0

    flat = make_backtester(weight_fn=no_positions)
    results = flat.run(list(workflow["run"]["signal_args"]), [])

    assert results["total_turnover"] == pytest.approx(0.0)
    assert results["final_equity"] == pytest.approx(config.start_capital)
    assert flat.runs[1]["net_returns"].abs().max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Step 8: portfolio returns -> risk metrics
# ---------------------------------------------------------------------------


def test_risk_layer_consumes_the_backtest_output(workflow, config):
    """The analysis layer accepts the backtester's series and returns sense."""
    net_returns = workflow["net_returns"]

    annual_return = ann_returns(config, net_returns)
    annual_vol = ann_volatility(config, net_returns)
    sharpe = ann_sharpe(config, net_returns)
    drawdown = max_drawdown(config, net_returns)
    value_at_risk = est_var(config, net_returns)
    conditional_var = cond_var(config, net_returns)

    for metric in (
        annual_return,
        annual_vol,
        sharpe,
        drawdown,
        value_at_risk,
        conditional_var,
    ):
        assert np.isscalar(metric) or np.ndim(metric) == 0
        assert np.isfinite(metric)

    assert annual_vol > 0
    assert -1 <= drawdown <= 0
    assert value_at_risk < 0
    assert conditional_var <= value_at_risk


def test_sharpe_is_consistent_with_its_own_inputs(workflow, config):
    """Sharpe must be the annualised return and vol it reports, not a re-estimate."""
    net_returns = workflow["net_returns"]

    expected = (
        ann_returns(config, net_returns) - config.risk_free_rate
    ) / ann_volatility(config, net_returns)

    assert ann_sharpe(config, net_returns) == pytest.approx(expected)


def test_max_drawdown_is_reachable_from_the_equity_curve(workflow, config):
    """The drawdown of the return series must match the equity path it came from."""
    equity = workflow["equity"]

    expected = (equity / equity.cummax() - 1).min()
    result = max_drawdown(config, workflow["net_returns"])

    assert result == pytest.approx(expected, abs=1e-12)


def test_higher_costs_produce_a_worse_risk_profile(workflow, make_backtester, config):
    """The cost setting has to survive all the way into the risk metrics."""
    free = make_backtester(cost_bps=0.0)
    free.run(list(workflow["run"]["signal_args"]), [])

    charged_return = ann_returns(config, workflow["net_returns"])
    free_return = ann_returns(config, free.runs[1]["net_returns"])

    assert free_return > charged_return


# ---------------------------------------------------------------------------
# Step 9: secondary analysis on the resulting series
# ---------------------------------------------------------------------------


def test_rolling_sharpe_accepts_the_portfolio_returns(workflow, config):
    """Rolling statistics must vary with the data, not collapse to a constant."""
    rolling = rolling_sharpe(config, workflow["net_returns"], window=30).dropna()

    pd.testing.assert_index_equal(
        rolling.index, workflow["net_returns"].index[-len(rolling) :]
    )
    assert len(rolling) > 0
    assert np.isfinite(rolling).all()
    assert rolling.nunique() > 1
