"""Tests for src.backend.backtest.walk_forward.

The suite asserts the behaviour a walk-forward backtester *should* have, not
necessarily what the current implementation does:

* folds tile the out-of-sample region exactly once, with no gaps, no overlap
  and no look-ahead (every training row strictly precedes its test rows);
* order quantities are sized so that ``quantity * price`` is the change in
  portfolio weight, which keeps transaction costs in the same units as the
  returns they are netted against;
* the cost model is called with the arguments it declares, and the
  ``cost_bps`` handed to the backtester is the one actually charged;
* ``run()`` is a pure composition of the per-fold helpers.

Tests covering behaviour that is currently broken are collected under
"Known bugs" at the bottom and marked ``xfail(strict=True)``, so each one
turns into a failure (XPASS) the moment the underlying bug is fixed and the
marker can be removed. Every bug marker names the fix in its ``reason``.
"""

import functools

import numpy as np
import pandas as pd
import pytest

from src.backend.analysis import (
    ann_calmar,
    ann_returns,
    ann_sharpe,
    ann_sortino,
    ann_volatility,
    cumulative_returns,
    max_drawdown,
)
from src.backend.backtest.costs import Order, MktState, linear_cost, sqrt_cost
from src.backend.backtest.walk_forward import WalkForwardBacktester
from src.backend.portfolio_construction import (
    avg_gross_exposure,
    avg_net_exposure,
    equal_active_weights,
)


SIGNAL_ARGS = [5]
PORTFOLIO_ARGS = []

METRIC_KEYS = {
    "cum_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "total_turnover",
    "avg_gross_exposure",
    "avg_net_exposure",
    "final_equity",
}


@pytest.fixture
def fold_inputs(backtester):
    """The first fold plus the weights the strategy produces on its train leg."""
    train, test = backtester._generate_folds()[0]
    signals = backtester._generate_signals(train, SIGNAL_ARGS)
    weights = backtester._construct_portfolio(signals, PORTFOLIO_ARGS)
    return train, test, weights


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_stores_its_inputs(backtester, config, price_data, volume_data):
    assert backtester.config is config
    assert backtester.price_data is price_data
    assert backtester.volume_data is volume_data


def test_init_defaults_cost_bps_to_config(make_backtester, config):
    assert make_backtester().cost_bps == config.cost_bps


def test_init_respects_an_explicit_cost_bps(make_backtester):
    assert make_backtester(cost_bps=42.0).cost_bps == 42.0


def test_init_defaults_start_capital_to_config(make_backtester, config):
    assert make_backtester().start_capital == config.start_capital


def test_init_respects_an_explicit_start_capital(make_backtester):
    assert make_backtester(start_capital=250_000.0).start_capital == 250_000.0


def test_init_promotes_series_price_and_volume_data(
    config, price_data, volume_data, signal_fn, tickers
):
    # Single-asset inputs are normalised once, so every downstream step can
    # assume a frame.
    backtester = WalkForwardBacktester(
        config=config,
        price_data=price_data[tickers[0]],
        volume_data=volume_data[tickers[0]],
        signal_fn=signal_fn,
        weight_fn=equal_active_weights,
        cost_model=sqrt_cost,
    )
    assert isinstance(backtester.price_data, pd.DataFrame)
    assert isinstance(backtester.volume_data, pd.DataFrame)
    assert list(backtester.price_data.columns) == [tickers[0]]


def test_init_defaults_train_frac_to_config(make_backtester, config):
    assert make_backtester(train_frac=None).train_frac == config.train_frac


def test_init_respects_an_explicit_train_frac(make_backtester):
    assert make_backtester(train_frac=0.5).train_frac == 0.5


def test_init_test_frac_complements_train_frac(make_backtester):
    backtester = make_backtester(train_frac=0.6)
    assert backtester.test_frac == pytest.approx(0.4)
    assert backtester.train_frac + backtester.test_frac == pytest.approx(1.0)


def test_init_defaults_window_params_to_config(make_backtester, config):
    params = make_backtester(window_params=None).window_params
    assert params == {
        "window_type": config.window_type,
        "window_len": config.window_len,
    }


def test_init_respects_explicit_window_params(make_backtester):
    params = {"window_type": "expanding", "window_len": 5}
    assert make_backtester(window_params=params).window_params == params


# ---------------------------------------------------------------------------
# _generate_folds()
# ---------------------------------------------------------------------------


def test_folds_are_train_test_pairs(backtester):
    folds = backtester._generate_folds()
    assert folds
    assert all(isinstance(fold, tuple) and len(fold) == 2 for fold in folds)


def test_folds_are_never_empty(backtester):
    for train, test in backtester._generate_folds():
        assert len(train) > 0
        assert len(test) > 0


def test_first_test_fold_starts_at_the_train_fraction_cutoff(
    backtester, price_data, config
):
    cutoff = int(len(price_data) * config.train_frac)
    _, first_test = backtester._generate_folds()[0]
    assert first_test.index[0] == price_data.index[cutoff]


def test_test_folds_are_window_len_long_except_the_last(backtester, window_params):
    folds = backtester._generate_folds()
    for _, test in folds[:-1]:
        assert len(test) == window_params["window_len"]
    assert 0 < len(folds[-1][1]) <= window_params["window_len"]


def test_test_folds_tile_the_oos_region_exactly_once(backtester, price_data, config):
    cutoff = int(len(price_data) * config.train_frac)
    stitched = pd.concat([test for _, test in backtester._generate_folds()])
    expected = price_data.iloc[cutoff:]
    pd.testing.assert_frame_equal(stitched, expected)


def test_test_folds_do_not_overlap(backtester):
    stitched = pd.concat([test for _, test in backtester._generate_folds()])
    assert stitched.index.is_unique
    assert stitched.index.is_monotonic_increasing


def test_training_data_never_leaks_into_the_future(backtester):
    for train, test in backtester._generate_folds():
        assert train.index.max() < test.index.min()


def test_training_data_is_contiguous_with_its_test_fold(backtester, price_data):
    positions = {date: i for i, date in enumerate(price_data.index)}
    for train, test in backtester._generate_folds():
        assert positions[test.index[0]] == positions[train.index[-1]] + 1


def test_rolling_window_keeps_a_constant_training_length(
    make_backtester, price_data, config
):
    backtester = make_backtester(
        window_params={"window_type": "rolling", "window_len": 10}
    )
    cutoff = int(len(price_data) * config.train_frac)
    lengths = {len(train) for train, _ in backtester._generate_folds()}
    assert lengths == {cutoff}


def test_rolling_window_rolls_the_training_start_forward(make_backtester):
    backtester = make_backtester(
        window_params={"window_type": "rolling", "window_len": 10}
    )
    starts = [train.index[0] for train, _ in backtester._generate_folds()]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1]


def test_expanding_window_always_starts_at_the_first_observation(
    make_backtester, price_data
):
    backtester = make_backtester(
        window_params={"window_type": "expanding", "window_len": 10}
    )
    for train, _ in backtester._generate_folds():
        assert train.index[0] == price_data.index[0]


def test_expanding_window_grows_the_training_set(make_backtester):
    backtester = make_backtester(
        window_params={"window_type": "expanding", "window_len": 10}
    )
    lengths = [len(train) for train, _ in backtester._generate_folds()]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths)


def test_both_window_types_share_the_same_test_folds(make_backtester):
    rolling = make_backtester(
        window_params={"window_type": "rolling", "window_len": 10}
    )._generate_folds()
    expanding = make_backtester(
        window_params={"window_type": "expanding", "window_len": 10}
    )._generate_folds()
    for (_, roll_test), (_, exp_test) in zip(rolling, expanding):
        pd.testing.assert_frame_equal(roll_test, exp_test)


def test_unknown_window_type_raises_value_error(make_backtester):
    backtester = make_backtester(
        window_params={"window_type": "sliding", "window_len": 10}
    )
    with pytest.raises(ValueError, match="Unknown window type"):
        backtester._generate_folds()


def test_shorter_window_len_produces_more_folds(make_backtester):
    few = make_backtester(
        window_params={"window_type": "rolling", "window_len": 20}
    )._generate_folds()
    many = make_backtester(
        window_params={"window_type": "rolling", "window_len": 5}
    )._generate_folds()
    assert len(many) > len(few)


def test_no_folds_when_all_data_is_used_for_training(make_backtester):
    backtester = make_backtester(train_frac=1.0)
    assert backtester._generate_folds() == []


# ---------------------------------------------------------------------------
# _generate_signals() / _construct_portfolio()
# ---------------------------------------------------------------------------


def test_generate_signals_delegates_to_signal_fn(make_backtester, config, price_data):
    seen = {}

    def spy_signal_fn(cfg, data, lookback):
        seen["config"] = cfg
        seen["data"] = data
        seen["lookback"] = lookback
        return pd.DataFrame(0.0, index=data.index, columns=data.columns)

    backtester = make_backtester(signal_fn=spy_signal_fn)
    train, _ = backtester._generate_folds()[0]
    backtester._generate_signals(train, [7])

    assert seen["config"] is config
    assert seen["lookback"] == 7
    pd.testing.assert_frame_equal(seen["data"], train)


def test_generate_signals_returns_one_column_per_ticker(backtester, tickers):
    train, _ = backtester._generate_folds()[0]
    signals = backtester._generate_signals(train, SIGNAL_ARGS)
    assert list(signals.columns) == tickers
    assert signals.index.equals(train.index)


def test_construct_portfolio_delegates_to_weight_fn(make_backtester, config):
    seen = {}

    def spy_weight_fn(cfg, signals, exposure):
        seen["config"] = cfg
        seen["signals"] = signals
        seen["exposure"] = exposure
        return signals

    backtester = make_backtester(weight_fn=spy_weight_fn)
    signals = pd.DataFrame({"AAA": [1.0], "BBB": [-1.0]})
    backtester._construct_portfolio(signals, [0.5])

    assert seen["config"] is config
    assert seen["exposure"] == 0.5
    pd.testing.assert_frame_equal(seen["signals"], signals)


def test_construct_portfolio_weights_align_with_the_signals(backtester, tickers):
    train, _ = backtester._generate_folds()[0]
    signals = backtester._generate_signals(train, SIGNAL_ARGS)
    weights = backtester._construct_portfolio(signals, PORTFOLIO_ARGS)
    assert list(weights.columns) == tickers
    assert weights.index.equals(signals.index)


# ---------------------------------------------------------------------------
# _generate_orders()
#
# Orders are now expressed in shares. The method is a pure translation of a
# share delta into Orders and matching MktStates; all weight arithmetic lives
# in run().
# ---------------------------------------------------------------------------


@pytest.fixture
def rebalance_inputs(backtester, fold_inputs):
    """The first fold's trade date, execution prices and share deltas from flat."""
    _, test, weights = fold_inputs
    trade_date = test.index[0]
    open_prices = test.iloc[0]
    target_weights = weights.iloc[-1]
    trade_shares = target_weights * backtester.start_capital / open_prices
    return trade_date, open_prices, trade_shares


def test_generate_orders_returns_arrays_of_orders_and_states(
    backtester, rebalance_inputs
):
    orders, states = backtester._generate_orders(*rebalance_inputs)
    assert isinstance(orders, np.ndarray)
    assert isinstance(states, np.ndarray)
    assert all(isinstance(o, Order) for o in orders)
    assert all(isinstance(s, MktState) for s in states)


def test_generate_orders_emits_one_order_per_traded_ticker(
    backtester, rebalance_inputs, tickers
):
    orders, states = backtester._generate_orders(*rebalance_inputs)
    assert [o.ticker for o in orders] == tickers
    assert len(states) == len(orders)


def test_generate_orders_follows_the_trade_shares_ordering(
    backtester, rebalance_inputs
):
    trade_date, prices, trade_shares = rebalance_inputs
    reversed_shares = trade_shares.iloc[::-1]
    orders, _ = backtester._generate_orders(trade_date, prices, reversed_shares)
    assert [o.ticker for o in orders] == list(reversed_shares.index)


def test_generate_orders_quantity_is_the_share_delta(backtester, rebalance_inputs):
    trade_date, prices, trade_shares = rebalance_inputs
    orders, _ = backtester._generate_orders(trade_date, prices, trade_shares)
    quantities = pd.Series({o.ticker: o.quantity for o in orders})
    pd.testing.assert_series_equal(quantities, trade_shares, check_names=False)


def test_generate_orders_prices_from_the_supplied_row(backtester, rebalance_inputs):
    trade_date, prices, trade_shares = rebalance_inputs
    orders, _ = backtester._generate_orders(trade_date, prices, trade_shares)
    for order in orders:
        assert order.price == prices[order.ticker]


def test_generate_orders_market_state_matches_the_order(
    backtester, rebalance_inputs, volume_data
):
    trade_date, prices, trade_shares = rebalance_inputs
    orders, states = backtester._generate_orders(trade_date, prices, trade_shares)
    for order, state in zip(orders, states):
        assert state.price == order.price
        assert state.daily_volume == volume_data.loc[trade_date, order.ticker]


def test_generate_orders_reads_volume_at_the_trade_date(backtester, rebalance_inputs):
    # A different trade date has to pull a different day of volume.
    trade_date, prices, trade_shares = rebalance_inputs
    later_date = backtester.volume_data.index[
        backtester.volume_data.index.get_loc(trade_date) + 1
    ]
    _, on_the_day = backtester._generate_orders(trade_date, prices, trade_shares)
    _, day_after = backtester._generate_orders(later_date, prices, trade_shares)
    assert [s.daily_volume for s in on_the_day] != [s.daily_volume for s in day_after]


def test_generate_orders_still_emits_an_order_for_an_untraded_ticker(
    backtester, rebalance_inputs, tickers
):
    trade_date, prices, trade_shares = rebalance_inputs
    flat = trade_shares.copy()
    flat[tickers[1]] = 0.0
    orders, _ = backtester._generate_orders(trade_date, prices, flat)
    assert [o.ticker for o in orders] == tickers
    assert next(o for o in orders if o.ticker == tickers[1]).quantity == 0.0


def test_generate_orders_preserves_the_sign_of_a_sale(backtester, rebalance_inputs):
    trade_date, prices, trade_shares = rebalance_inputs
    orders, _ = backtester._generate_orders(trade_date, prices, -trade_shares)
    for order in orders:
        assert np.sign(order.quantity) == -np.sign(trade_shares[order.ticker])


def test_generate_orders_notional_is_weight_times_capital(
    backtester, rebalance_inputs, fold_inputs
):
    # Sizing from flat with the full book, quantity * price is the target
    # weight scaled by the capital at risk - the whole point of start_capital.
    _, _, weights = fold_inputs
    target_weights = weights.iloc[-1]
    orders, _ = backtester._generate_orders(*rebalance_inputs)
    notional = pd.Series({o.ticker: o.quantity * o.price for o in orders})
    expected = target_weights * backtester.start_capital
    pd.testing.assert_series_equal(notional, expected, check_names=False)


# ---------------------------------------------------------------------------
# _compute_costs()
# ---------------------------------------------------------------------------


def test_compute_costs_delegates_to_the_cost_model(make_backtester, config):
    seen = {}

    def spy_cost_model(cfg, orders, mkt_states, cost_bps=None):
        seen["config"] = cfg
        seen["orders"] = orders
        seen["mkt_states"] = mkt_states
        seen["cost_bps"] = cost_bps
        return np.zeros(len(orders))

    backtester = make_backtester(cost_model=spy_cost_model, cost_bps=33.0)
    orders = np.array([Order("AAA", 1.0, 10.0)])
    states = np.array([MktState(10.0, 1000.0)])
    backtester._compute_costs(orders, states)

    assert seen["config"] is config
    assert seen["orders"] is orders
    assert seen["mkt_states"] is states
    assert seen["cost_bps"] == 33.0


def test_compute_costs_returns_the_cost_model_output(make_backtester):
    backtester = make_backtester(cost_model=sqrt_cost)
    orders = np.array([Order("AAA", 1000.0, 50.0)])
    states = np.array([MktState(50.0, 1_000_000.0)])
    expected = sqrt_cost(backtester.config, orders, states)
    assert backtester._compute_costs(orders, states) == pytest.approx(expected)


def test_flat_cost_model_charges_bps_of_the_traded_notional(
    make_backtester, flat_cost_model, rebalance_inputs
):
    backtester = make_backtester(cost_model=flat_cost_model(25.0))
    trade_date, prices, trade_shares = rebalance_inputs
    orders, states = backtester._generate_orders(trade_date, prices, trade_shares)

    costs = backtester._compute_costs(orders, states)
    expected = (trade_shares.abs() * prices).sum() * 25.0 / 10000
    assert costs.sum() == pytest.approx(expected)


def test_linear_cost_is_usable_as_a_cost_model(make_backtester, rebalance_inputs):
    backtester = make_backtester(cost_model=linear_cost)
    orders, states = backtester._generate_orders(*rebalance_inputs)
    costs = backtester._compute_costs(orders, states)
    expected = linear_cost(backtester.config, orders, cost_bps=backtester.cost_bps)
    assert costs == pytest.approx(expected)


def test_backtester_charges_the_cost_bps_it_was_given(
    make_backtester, rebalance_inputs
):
    cheap = make_backtester(cost_model=sqrt_cost, cost_bps=1.0)
    dear = make_backtester(cost_model=sqrt_cost, cost_bps=100.0)

    cheap_costs = cheap._compute_costs(*cheap._generate_orders(*rebalance_inputs))
    dear_costs = dear._compute_costs(*dear._generate_orders(*rebalance_inputs))

    assert np.sum(dear_costs) == pytest.approx(100 * np.sum(cheap_costs))


def test_sqrt_cost_rises_super_linearly_with_the_book(
    make_backtester, rebalance_inputs
):
    # Impact is what start_capital was introduced for: doubling the book more
    # than doubles the cost, because participation doubles too.
    trade_date, prices, trade_shares = rebalance_inputs
    backtester = make_backtester(cost_model=sqrt_cost)

    small = backtester._compute_costs(
        *backtester._generate_orders(trade_date, prices, trade_shares)
    )
    large = backtester._compute_costs(
        *backtester._generate_orders(trade_date, prices, trade_shares * 2)
    )
    assert np.sum(large) == pytest.approx(2**1.5 * np.sum(small))


# ---------------------------------------------------------------------------
# _daily_rebalanced_backtest()
#
# Kept as a reference implementation. run() no longer uses it: it models a
# daily-rebalanced portfolio, whereas the share-based loop is buy-and-hold
# within a fold. The last two tests pin exactly where the two differ.
# ---------------------------------------------------------------------------


def test_daily_rebalanced_returns_one_value_per_oos_bar(backtester, fold_inputs):
    train, test, weights = fold_inputs
    returns = backtester._daily_rebalanced_backtest(train, test, weights)
    assert isinstance(returns, pd.Series)
    assert returns.index.equals(test.index)


def test_daily_rebalanced_has_no_missing_first_return(backtester, fold_inputs):
    # The last training bar is prepended, so the first OOS return is a real
    # close-to-close return rather than NaN.
    train, test, weights = fold_inputs
    returns = backtester._daily_rebalanced_backtest(train, test, weights)
    assert not returns.isna().any()


def test_daily_rebalanced_matches_a_manual_weighted_sum(backtester, fold_inputs):
    train, test, weights = fold_inputs
    target_weights = weights.iloc[-1]
    prices = pd.concat([train.tail(1), test])
    expected = prices.pct_change().iloc[1:].mul(target_weights, axis=1).sum(axis=1)
    pd.testing.assert_series_equal(
        backtester._daily_rebalanced_backtest(train, test, weights), expected
    )


def test_daily_rebalanced_first_return_spans_the_train_test_boundary(
    backtester, fold_inputs
):
    train, test, weights = fold_inputs
    target_weights = weights.iloc[-1]
    returns = backtester._daily_rebalanced_backtest(train, test, weights)
    expected = ((test.iloc[0] / train.iloc[-1]) - 1).mul(target_weights).sum()
    assert returns.iloc[0] == pytest.approx(expected)


def test_daily_rebalanced_is_linear_in_the_weights(backtester, fold_inputs):
    train, test, weights = fold_inputs
    base = backtester._daily_rebalanced_backtest(train, test, weights)
    doubled = backtester._daily_rebalanced_backtest(train, test, weights * 2)
    pd.testing.assert_series_equal(doubled, base * 2)


def test_daily_rebalanced_is_flat_for_zero_weights(backtester, fold_inputs):
    train, test, weights = fold_inputs
    returns = backtester._daily_rebalanced_backtest(train, test, weights * 0.0)
    assert returns.eq(0.0).all()


def test_daily_rebalanced_uses_only_the_final_training_weights(backtester, fold_inputs):
    train, test, weights = fold_inputs
    scrambled = weights.copy()
    scrambled.iloc[:-1] = -scrambled.iloc[:-1]
    pd.testing.assert_series_equal(
        backtester._daily_rebalanced_backtest(train, test, weights),
        backtester._daily_rebalanced_backtest(train, test, scrambled),
    )


def test_daily_rebalanced_accepts_a_weights_series(backtester, fold_inputs):
    train, test, weights = fold_inputs
    pd.testing.assert_series_equal(
        backtester._daily_rebalanced_backtest(train, test, weights),
        backtester._daily_rebalanced_backtest(train, test, weights.iloc[-1]),
    )


def test_daily_rebalanced_tracks_a_single_asset_exactly(
    backtester, fold_inputs, tickers
):
    # Fully invested in one name, the portfolio return is that asset's return.
    train, test, weights = fold_inputs
    target_weights = pd.Series(0.0, index=tickers)
    target_weights[tickers[0]] = 1.0
    returns = backtester._daily_rebalanced_backtest(train, test, target_weights)
    expected = pd.concat([train.tail(1), test])[tickers[0]].pct_change().iloc[1:]
    pd.testing.assert_series_equal(returns, expected, check_names=False)


def test_daily_rebalanced_equals_buy_and_hold_without_dispersion(
    make_backtester, config, tickers
):
    # With every asset on the same path there is nothing to drift apart, so
    # rebalancing daily and holding shares give the same equity curve.
    index = pd.bdate_range("2022-01-03", periods=40)
    path = 100 * np.exp(np.cumsum(np.full(len(index), 0.002)))
    prices = pd.DataFrame({t: path for t in tickers}, index=index)
    backtester = make_backtester(price_data=prices, volume_data=prices * 0 + 1e6)

    train, test = prices.iloc[:20], prices.iloc[20:]
    target_weights = pd.Series(1 / 3, index=tickers)

    rebalanced = backtester._daily_rebalanced_backtest(train, test, target_weights)
    shares = target_weights * backtester.start_capital / test.iloc[0]
    held = test.mul(shares, axis=1).sum(axis=1)

    # Compare growth across the OOS window itself, dropping the first return
    # since it spans the train/test boundary and predates the share purchase.
    rebalanced_growth = (1 + rebalanced.iloc[1:]).cumprod().iloc[-1]
    held_growth = held.iloc[-1] / held.iloc[0]
    assert rebalanced_growth == pytest.approx(held_growth)


def test_daily_rebalanced_diverges_from_buy_and_hold_under_dispersion(
    make_backtester, tickers
):
    # One name triples while the others are flat: the daily-rebalanced book
    # keeps selling the winner, the share-based book does not.
    index = pd.bdate_range("2022-01-03", periods=40)
    prices = pd.DataFrame(100.0, index=index, columns=tickers)
    prices[tickers[0]] = np.linspace(100.0, 300.0, len(index))
    backtester = make_backtester(price_data=prices, volume_data=prices * 0 + 1e6)

    train, test = prices.iloc[:20], prices.iloc[20:]
    target_weights = pd.Series(1 / 3, index=tickers)

    rebalanced = backtester._daily_rebalanced_backtest(train, test, target_weights)
    shares = target_weights * backtester.start_capital / test.iloc[0]
    held = test.mul(shares, axis=1).sum(axis=1)

    rebalanced_growth = (1 + rebalanced.iloc[1:]).cumprod().iloc[-1]
    held_growth = held.iloc[-1] / held.iloc[0]
    # Holding shares lets the winner compound; rebalancing daily trims it.
    assert held_growth > rebalanced_growth


# ---------------------------------------------------------------------------
# _evaluate()
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluation_inputs(tickers):
    """A short, hand-checkable returns series and realised weight path."""
    index = pd.bdate_range("2023-01-02", periods=5)
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, -0.005], index=index)
    weights = pd.DataFrame(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
        ],
        index=index,
        columns=tickers,
    )
    return returns, weights, 2.0


def test_evaluate_reports_every_metric(backtester, evaluation_inputs):
    results = backtester._evaluate(*evaluation_inputs)
    assert set(results) == METRIC_KEYS - {"final_equity"}


def test_evaluate_metrics_are_finite_scalars(backtester, evaluation_inputs):
    results = backtester._evaluate(*evaluation_inputs)
    assert all(np.ndim(value) == 0 for value in results.values())
    assert all(np.isfinite(value) for value in results.values())


def test_evaluate_cum_return_compounds_the_net_returns(backtester, evaluation_inputs):
    returns, _, _ = evaluation_inputs
    results = backtester._evaluate(*evaluation_inputs)
    expected = np.prod(1 + returns.to_numpy()) - 1
    assert results["cum_return"] == pytest.approx(expected)


def test_evaluate_matches_the_analysis_helpers(backtester, config, evaluation_inputs):
    returns, weights, traded = evaluation_inputs
    results = backtester._evaluate(returns, weights, traded)
    assert results["cum_return"] == pytest.approx(cumulative_returns(returns).iloc[-1])
    assert results["ann_return"] == pytest.approx(ann_returns(config, returns))
    assert results["ann_vol"] == pytest.approx(ann_volatility(config, returns))
    assert results["sharpe"] == pytest.approx(ann_sharpe(config, returns))
    assert results["sortino"] == pytest.approx(ann_sortino(config, returns))
    assert results["max_drawdown"] == pytest.approx(max_drawdown(config, returns))
    assert results["calmar"] == pytest.approx(ann_calmar(config, returns))


def test_evaluate_reports_the_turnover_it_was_given(backtester, evaluation_inputs):
    # Turnover is passed in, not differenced out of the weights: a drifting
    # weight path would otherwise count price moves as trading.
    returns, weights, _ = evaluation_inputs
    assert backtester._evaluate(returns, weights, 7.5)["total_turnover"] == 7.5


def test_evaluate_turnover_is_independent_of_the_weight_path(
    backtester, evaluation_inputs
):
    returns, weights, traded = evaluation_inputs
    drifting = weights * np.linspace(1.0, 1.5, len(weights))[:, None]
    assert (
        backtester._evaluate(returns, drifting, traded)["total_turnover"]
        == backtester._evaluate(returns, weights, traded)["total_turnover"]
    )


def test_evaluate_exposures_match_the_helpers(backtester, config, evaluation_inputs):
    returns, weights, traded = evaluation_inputs
    results = backtester._evaluate(returns, weights, traded)
    assert results["avg_gross_exposure"] == pytest.approx(
        avg_gross_exposure(config, weights)
    )
    assert results["avg_net_exposure"] == pytest.approx(
        avg_net_exposure(config, weights)
    )


def test_evaluate_gross_exposure_exceeds_net_exposure_with_shorts(
    backtester, evaluation_inputs, tickers
):
    returns, weights, traded = evaluation_inputs
    shorted = weights.copy()
    shorted[tickers[0]] *= -1
    results = backtester._evaluate(returns, shorted, traded)
    assert results["avg_gross_exposure"] > results["avg_net_exposure"]


def test_evaluate_ann_vol_is_non_negative(backtester, evaluation_inputs):
    assert backtester._evaluate(*evaluation_inputs)["ann_vol"] >= 0


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    """End-to-end behaviour of the full walk-forward loop."""

    def test_run_reports_every_metric(self, backtester):
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert set(results) == METRIC_KEYS

    def test_run_metrics_are_finite(self, backtester):
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert all(np.isfinite(value) for value in results.values())

    def test_run_is_deterministic(self, make_backtester):
        first = make_backtester().run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        second = make_backtester().run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert first == pytest.approx(second)

    def test_run_final_equity_agrees_with_cum_return(self, backtester):
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert results["final_equity"] / backtester.start_capital - 1 == pytest.approx(
            results["cum_return"]
        )

    def test_run_final_equity_scales_with_start_capital(self, make_backtester):
        # Under a proportional cost model the strategy is scale-free, so ten
        # times the capital is ten times the equity and the same return.
        small = make_backtester(start_capital=100_000.0, cost_model=linear_cost).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        large = make_backtester(start_capital=1_000_000.0, cost_model=linear_cost).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        assert large["final_equity"] == pytest.approx(10 * small["final_equity"])
        assert large["cum_return"] == pytest.approx(small["cum_return"])

    def test_run_a_larger_book_pays_more_market_impact(self, make_backtester):
        # sqrt_cost is not scale-free: participation rises with the book, so
        # the same strategy nets less at size.
        small = make_backtester(start_capital=100_000.0, cost_model=sqrt_cost).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        large = make_backtester(start_capital=500_000_000.0, cost_model=sqrt_cost).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        assert large["cum_return"] < small["cum_return"]

    def test_run_costs_reduce_the_cumulative_return(
        self, make_backtester, flat_cost_model
    ):
        free = make_backtester(cost_model=flat_cost_model(0.0)).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        charged = make_backtester(cost_model=flat_cost_model(50.0)).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        assert charged["cum_return"] < free["cum_return"]

    def test_run_higher_costs_are_monotonically_worse(
        self, make_backtester, flat_cost_model
    ):
        returns = [
            make_backtester(cost_model=flat_cost_model(bps)).run(
                SIGNAL_ARGS, PORTFOLIO_ARGS
            )["cum_return"]
            for bps in (0.0, 10.0, 100.0)
        ]
        assert returns == sorted(returns, reverse=True)

    def test_run_reports_positive_turnover_for_a_trading_strategy(self, backtester):
        assert backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)["total_turnover"] > 0

    def test_run_rebalances_a_constant_target_because_the_book_drifts(
        self, make_backtester, constant_signal_fn, flat_cost_model
    ):
        # Target weights never change, so a static-weight backtester would
        # trade once and stop. Holding shares, prices pull the book off target
        # and every later fold has to trade it back.
        backtester = make_backtester(
            signal_fn=lambda cfg, data, *args: constant_signal_fn(cfg, data, 1),
            cost_model=flat_cost_model(0.0),
        )
        n_folds = len(backtester._generate_folds())
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)

        # The initial build is 1.0 of turnover; anything beyond that is drift
        # being traded back, and it cannot be attributed to a changed target.
        # The excess stays small - bounded well below a full rebuild per fold,
        # which is what a static-weight implementation would have charged.
        assert n_folds > 1
        assert 1.0 < results["total_turnover"] < 1.5

    def test_run_long_only_book_has_equal_gross_and_net_exposure(
        self, make_backtester, constant_signal_fn
    ):
        backtester = make_backtester(
            signal_fn=lambda cfg, data, *args: constant_signal_fn(cfg, data, 1)
        )
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert results["avg_net_exposure"] == pytest.approx(
            results["avg_gross_exposure"]
        )
        assert results["avg_gross_exposure"] == pytest.approx(1.0, rel=0.05)

    def test_run_long_short_book_is_less_net_than_gross(self, backtester):
        results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert results["avg_net_exposure"] < results["avg_gross_exposure"]

    def test_run_uses_the_real_linear_cost_model(self, make_backtester):
        results = make_backtester(cost_model=linear_cost).run(
            SIGNAL_ARGS, PORTFOLIO_ARGS
        )
        assert set(results) == METRIC_KEYS
        assert all(np.isfinite(value) for value in results.values())

    def test_run_cost_bps_flows_through_to_the_result(self, make_backtester):
        returns = [
            make_backtester(cost_model=linear_cost, cost_bps=bps).run(
                SIGNAL_ARGS, PORTFOLIO_ARGS
            )["cum_return"]
            for bps in (0.0, 10.0, 500.0)
        ]
        assert returns == sorted(returns, reverse=True)
        assert returns[0] > returns[-1]

    def test_run_raises_when_costs_would_wipe_out_the_book(self, make_backtester):
        # A cost model that charges more than the portfolio is worth must fail
        # loudly rather than carry a negative equity forward.
        def ruinous_cost(config, orders, mkt_states, cost_bps=None):
            return np.full(len(orders), 1e12)

        backtester = make_backtester(cost_model=ruinous_cost)
        with pytest.raises(ValueError, match="exceed the available capital"):
            backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)

    def test_run_supports_expanding_windows(self, make_backtester):
        backtester = make_backtester(
            window_params={
                "window_type": "expanding",
                "window_len": 10,
            }
        )
        assert set(backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)) == METRIC_KEYS

    def test_run_window_type_is_immaterial_for_a_short_lookback_signal(
        self, make_backtester
    ):
        # Both window types share the same test folds and the same trailing
        # training bars, and only the last row of weights is traded, so a
        # 5-bar momentum signal cannot tell rolling and expanding apart.
        rolling = make_backtester(
            window_params={
                "window_type": "rolling",
                "window_len": 10,
            }
        ).run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        expanding = make_backtester(
            window_params={
                "window_type": "expanding",
                "window_len": 10,
            }
        ).run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert rolling == pytest.approx(expanding)


def test_run_composes_the_per_fold_helpers(
    make_backtester, flat_cost_model, price_data, config
):
    # Replays the fold loop through the public helpers and checks run() adds
    # nothing of its own. Also pins that the equity curve spans exactly the
    # out-of-sample region.
    backtester = make_backtester(cost_model=flat_cost_model(15.0))
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)

    current_capital = backtester.start_capital
    current_shares = pd.Series(0.0, index=price_data.columns)
    cash = backtester.start_capital
    equity_parts = []
    weight_parts = []
    expected_turnover = 0.0

    for train, test in backtester._generate_folds():
        signals = backtester._generate_signals(train, SIGNAL_ARGS)
        target_weights = backtester._construct_portfolio(signals, PORTFOLIO_ARGS).iloc[
            -1
        ]

        open_prices = test.iloc[0]
        # The book is marked to market on the rebalance bar before it is
        # resized, so the target is expressed against the capital actually
        # standing at the trade date rather than the previous fold's close.
        current_capital = (current_shares * open_prices).sum() + cash
        target_shares = target_weights * current_capital / open_prices
        trade_shares = target_shares - current_shares

        orders, states = backtester._generate_orders(
            test.index[0], open_prices, trade_shares
        )
        fold_cost = float(np.sum(backtester._compute_costs(orders, states)))

        cash -= (trade_shares * open_prices).sum() + fold_cost
        position_vals = test.mul(target_shares, axis=1)
        fold_equity = position_vals.sum(axis=1) + cash

        expected_turnover += (trade_shares.abs() * open_prices).sum() / current_capital
        equity_parts.append(fold_equity)
        weight_parts.append(position_vals.div(fold_equity, axis=0))

        current_shares = target_shares
        current_capital = fold_equity.iloc[-1]

    equity = pd.concat(equity_parts)
    cutoff = int(len(price_data) * config.train_frac)
    assert equity.index.equals(price_data.index[cutoff:])

    net_returns = equity.pct_change()
    net_returns.iloc[0] = equity.iloc[0] / backtester.start_capital - 1
    expected = backtester._evaluate(
        net_returns, pd.concat(weight_parts), expected_turnover
    )
    expected["final_equity"] = equity.iloc[-1]
    assert results == pytest.approx(expected)


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_run_a_flat_strategy_earns_nothing_and_trades_nothing(
    make_backtester, constant_signal_fn
):
    backtester = make_backtester(
        signal_fn=lambda cfg, data, *args: constant_signal_fn(cfg, data, 0)
    )
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert results["cum_return"] == pytest.approx(0.0)
    assert results["total_turnover"] == pytest.approx(0.0)
    assert results["final_equity"] == pytest.approx(backtester.start_capital)


def test_backtester_supports_single_asset_series_data(
    config, price_data, volume_data, signal_fn, window_params, tickers
):
    backtester = WalkForwardBacktester(
        config=config,
        price_data=price_data[tickers[0]],
        volume_data=volume_data[tickers[0]],
        signal_fn=signal_fn,
        weight_fn=equal_active_weights,
        cost_model=sqrt_cost,
        window_params=dict(window_params),
    )
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert set(results) == METRIC_KEYS
    assert all(np.isfinite(value) for value in results.values())


def test_run_tolerates_a_weight_fn_that_drops_a_ticker(make_backtester, tickers):
    def subset_weight_fn(config, signals, *args):
        return equal_active_weights(config, signals).drop(columns=[tickers[-1]])

    backtester = make_backtester(weight_fn=subset_weight_fn)
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert all(np.isfinite(value) for value in results.values())


# ---------------------------------------------------------------------------
# Run history (self.runs / self.run_count)
#
# Each completed run is archived so the same backtester can be re-used and its
# earlier results plotted or compared. History is only written once a run has
# succeeded, and earlier entries are never disturbed by later ones.
# ---------------------------------------------------------------------------

HISTORY_KEYS = {
    # provenance: what was run
    "signal_args",
    "portfolio_args",
    "signal_fn",
    "weight_fn",
    "cost_model",
    "cost_bps",
    "start_capital",
    "window_params",
    "train_frac",
    # per-rebalance blotter
    "target_shares",
    "trade_shares",
    "trade_prices",
    "fold_capital",
    "cash_rebalance",
    "fold_costs",
    "fold_turnovers",
    # daily series and metrics
    "equity_curve",
    "oos_weights",
    "net_returns",
    "results",
}

PER_FOLD_KEYS = [
    "fold_capital",
    "cash_rebalance",
    "fold_costs",
    "fold_turnovers",
]

BLOTTER_KEYS = ["target_shares", "trade_shares", "trade_prices"]


def test_run_history_starts_empty(backtester):
    assert backtester.runs == {}
    assert backtester.run_count == 0


def test_run_history_records_one_entry_per_run(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.run_count == 1
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.run_count == 2


def test_run_history_is_keyed_sequentially_from_one(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert list(backtester.runs) == [1, 2, 3]


def test_run_count_matches_the_number_of_stored_runs(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
        assert backtester.run_count == len(backtester.runs)


def test_run_history_entry_holds_every_artefact(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert set(backtester.runs[1]) == HISTORY_KEYS


def test_run_history_stores_the_results_that_were_returned(backtester):
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.runs[1]["results"] == pytest.approx(results)


def test_run_history_preserves_earlier_runs(backtester):
    first = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    stored_first = dict(backtester.runs[1]["results"])
    backtester.run([10], PORTFOLIO_ARGS)
    assert backtester.runs[1]["results"] == pytest.approx(stored_first)
    assert backtester.runs[1]["results"] == pytest.approx(first)


def test_run_history_distinguishes_runs_with_different_arguments(backtester):
    backtester.run([5], PORTFOLIO_ARGS)
    backtester.run([10], PORTFOLIO_ARGS)
    assert backtester.runs[1]["results"]["cum_return"] != pytest.approx(
        backtester.runs[2]["results"]["cum_return"]
    )


def test_run_history_equity_curve_spans_the_out_of_sample_window(
    backtester, price_data, config
):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    equity = backtester.runs[1]["equity_curve"]
    cutoff = int(len(price_data) * config.train_frac)
    assert isinstance(equity, pd.Series)
    assert equity.index.equals(price_data.index[cutoff:])


def test_run_history_equity_curve_ends_at_the_reported_final_equity(backtester):
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.runs[1]["equity_curve"].iloc[-1] == pytest.approx(
        results["final_equity"]
    )


def test_run_history_equity_curve_stays_positive(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    equity = backtester.runs[1]["equity_curve"]
    # The first OOS bar has already earned a return and paid the opening
    # trade, so it sits near - not at - the starting capital.
    assert equity.iloc[0] == pytest.approx(backtester.start_capital, rel=0.1)
    assert (equity > 0).all()


def test_run_history_net_returns_align_with_the_equity_curve(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    entry = backtester.runs[1]
    assert entry["net_returns"].index.equals(entry["equity_curve"].index)
    assert not entry["net_returns"].isna().any()


def test_run_history_net_returns_reproduce_the_reported_cum_return(backtester):
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    net_returns = backtester.runs[1]["net_returns"]
    assert (1 + net_returns).prod() - 1 == pytest.approx(results["cum_return"])


def test_run_history_weights_align_with_the_equity_curve(backtester, tickers):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    entry = backtester.runs[1]
    weights = entry["oos_weights"]
    assert isinstance(weights, pd.DataFrame)
    assert weights.index.equals(entry["equity_curve"].index)
    assert list(weights.columns) == tickers


def test_run_history_weights_are_the_realised_drifting_path(backtester):
    # Held weights move between rebalances, so consecutive rows must differ.
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    weights = backtester.runs[1]["oos_weights"]
    assert (weights.diff().iloc[1:].abs().sum(axis=1) > 0).any()


def test_a_failed_run_is_not_recorded(make_backtester):
    def ruinous_cost(config, orders, mkt_states, cost_bps=None):
        return np.full(len(orders), 1e12)

    backtester = make_backtester(cost_model=ruinous_cost)
    with pytest.raises(ValueError):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.run_count == 0
    assert backtester.runs == {}


def test_run_history_survives_a_failed_run_in_between(make_backtester, flat_cost_model):
    backtester = make_backtester(cost_model=flat_cost_model(5.0))
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)

    backtester.cost_model = lambda config, orders, mkt_states, cost_bps=None: np.full(
        len(orders), 1e12
    )
    with pytest.raises(ValueError):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)

    assert list(backtester.runs) == [1]
    assert backtester.run_count == 1


def test_run_history_is_per_instance(make_backtester):
    first = make_backtester()
    second = make_backtester()
    first.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert second.runs == {}
    assert second.run_count == 0


def test_run_history_is_not_corrupted_by_mutating_the_returned_results(backtester):
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    results["label"] = "scratch"
    del results["sharpe"]
    assert "label" not in backtester.runs[1]["results"]
    assert "sharpe" in backtester.runs[1]["results"]


# ---------------------------------------------------------------------------
# Run history: provenance
#
# Two runs of the same backtester have to be tellable apart after the fact,
# and the record has to be a snapshot rather than a live view of the caller's
# arguments or the instance's attributes.
# ---------------------------------------------------------------------------


def test_run_history_records_the_arguments_it_was_called_with(backtester):
    backtester.run([7], [0.5])
    entry = backtester.runs[1]
    assert entry["signal_args"] == [7]
    assert entry["portfolio_args"] == [0.5]


def test_run_history_copies_the_arguments(backtester):
    signal_args = [5]
    portfolio_args = []
    backtester.run(signal_args, portfolio_args)
    signal_args.append(99)
    portfolio_args.append(99)
    assert backtester.runs[1]["signal_args"] == [5]
    assert backtester.runs[1]["portfolio_args"] == []


def test_run_history_accepts_tuple_arguments(backtester):
    backtester.run((5,), ())
    assert backtester.runs[1]["signal_args"] == [5]
    assert backtester.runs[1]["portfolio_args"] == []


def test_run_history_records_the_pluggable_function_names(make_backtester):
    backtester = make_backtester(weight_fn=equal_active_weights, cost_model=linear_cost)
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    entry = backtester.runs[1]
    assert entry["signal_fn"] == "momentum_signal"
    assert entry["weight_fn"] == "equal_active_weights"
    assert entry["cost_model"] == "linear_cost"


def test_run_history_handles_a_callable_without_a_name(make_backtester):
    # functools.partial is the natural way to pin a cost parameter for a
    # sweep, and it has no __name__.
    backtester = make_backtester(cost_model=functools.partial(linear_cost))
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert isinstance(backtester.runs[1]["cost_model"], str)
    assert backtester.runs[1]["cost_model"]


def test_run_history_records_the_instance_parameters(make_backtester, window_params):
    backtester = make_backtester(
        cost_bps=17.0, start_capital=250_000.0, train_frac=0.6
    )
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    entry = backtester.runs[1]
    assert entry["cost_bps"] == 17.0
    assert entry["start_capital"] == 250_000.0
    assert entry["train_frac"] == 0.6
    assert entry["window_params"] == window_params


def test_run_history_copies_the_window_params(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    recorded = dict(backtester.runs[1]["window_params"])
    backtester.window_params["window_len"] = 999
    assert backtester.runs[1]["window_params"] == recorded


def test_run_history_tracks_parameters_changed_between_runs(
    make_backtester, flat_cost_model
):
    # Sweeping a parameter on one instance is the reason provenance is stored.
    backtester = make_backtester(cost_model=flat_cost_model(0.0), cost_bps=10.0)
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.cost_bps = 250.0
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert backtester.runs[1]["cost_bps"] == 10.0
    assert backtester.runs[2]["cost_bps"] == 250.0


# ---------------------------------------------------------------------------
# Run history: per-rebalance series
# ---------------------------------------------------------------------------


@pytest.fixture
def history_entry(backtester):
    """A completed run plus the folds that produced it."""
    folds = backtester._generate_folds()
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    return backtester, backtester.runs[1], folds


@pytest.mark.parametrize("key", PER_FOLD_KEYS)
def test_per_fold_series_have_one_entry_per_fold(history_entry, key):
    _, entry, folds = history_entry
    assert isinstance(entry[key], pd.Series)
    assert len(entry[key]) == len(folds)


@pytest.mark.parametrize("key", PER_FOLD_KEYS)
def test_per_fold_series_are_labelled_by_the_rebalance_dates(history_entry, key):
    _, entry, folds = history_entry
    expected = pd.DatetimeIndex([test.index[0] for _, test in folds])
    assert entry[key].index.equals(expected)


def test_fold_costs_are_non_negative(history_entry):
    _, entry, _ = history_entry
    assert (entry["fold_costs"] >= 0).all()


def test_fold_turnovers_sum_to_the_reported_total(history_entry):
    _, entry, _ = history_entry
    assert entry["fold_turnovers"].sum() == pytest.approx(
        entry["results"]["total_turnover"]
    )


def test_fold_turnovers_are_non_negative(history_entry):
    _, entry, _ = history_entry
    assert (entry["fold_turnovers"] >= 0).all()


def test_first_fold_capital_is_the_starting_capital(history_entry):
    backtester, entry, _ = history_entry
    assert entry["fold_capital"].iloc[0] == pytest.approx(backtester.start_capital)


def test_fold_capital_is_the_book_marked_on_the_rebalance_bar(history_entry):
    # Capital is the book marked to market on the trade date, so it is that
    # bar's equity gross of the transaction cost charged on the same bar.
    _, entry, _ = history_entry
    dates = entry["fold_capital"].index
    equity_on_trade_dates = entry["equity_curve"].loc[dates]
    assert entry["fold_capital"].to_numpy() == pytest.approx(
        (equity_on_trade_dates + entry["fold_costs"]).to_numpy()
    )


def test_fold_capital_starts_at_the_opening_capital(history_entry):
    # The first rebalance has no book to mark, so it sizes off start capital.
    backtester, entry, _ = history_entry
    assert entry["fold_capital"].iloc[0] == pytest.approx(backtester.start_capital)


def test_fold_capital_is_not_the_previous_folds_stale_close(history_entry):
    # Guards the fix for sizing on stale equity: the capital a fold trades
    # against must be marked on its own rebalance bar, not carried unchanged
    # from the previous fold's final close one bar earlier.
    _, entry, folds = history_entry
    equity = entry["equity_curve"]
    stale = [equity.loc[folds[position - 1][1].index[-1]] for position in range(1, len(folds))]
    marked = entry["fold_capital"].to_numpy()[1:]
    assert not np.allclose(marked, stale)


def test_cash_is_recorded_after_the_trade_settles(history_entry):
    # Equity on the rebalance bar is the marked book plus the cash left over.
    _, entry, _ = history_entry
    positions = (entry["target_shares"] * entry["trade_prices"]).sum(axis=1)
    equity_on_trade_dates = entry["equity_curve"].loc[entry["cash_rebalance"].index]
    assert (positions + entry["cash_rebalance"]).to_numpy() == pytest.approx(
        equity_on_trade_dates.to_numpy()
    )


# ---------------------------------------------------------------------------
# Run history: the blotter frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", BLOTTER_KEYS)
def test_blotter_frames_are_folds_by_tickers(history_entry, key, tickers):
    _, entry, folds = history_entry
    frame = entry[key]
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (len(folds), len(tickers))
    assert list(frame.columns) == tickers


@pytest.mark.parametrize("key", BLOTTER_KEYS)
def test_blotter_frames_are_labelled_by_the_rebalance_dates(history_entry, key):
    _, entry, folds = history_entry
    expected = pd.DatetimeIndex([test.index[0] for _, test in folds])
    assert entry[key].index.equals(expected)


def test_trade_prices_are_the_close_prices_on_the_rebalance_date(history_entry):
    backtester, entry, _ = history_entry
    for trade_date, row in entry["trade_prices"].iterrows():
        pd.testing.assert_series_equal(
            row, backtester.price_data.loc[trade_date], check_names=False
        )


def test_first_rebalance_buys_the_whole_target_book(history_entry):
    # Starting flat, the first trade is the target position itself.
    _, entry, _ = history_entry
    pd.testing.assert_series_equal(
        entry["trade_shares"].iloc[0], entry["target_shares"].iloc[0]
    )


def test_trade_shares_are_the_change_in_target_shares(history_entry):
    # Holdings are left untouched between rebalances, so each trade is exactly
    # the step from one target book to the next.
    _, entry, _ = history_entry
    expected = entry["target_shares"].diff()
    expected.iloc[0] = entry["target_shares"].iloc[0]
    pd.testing.assert_frame_equal(entry["trade_shares"], expected)


def test_traded_notional_reproduces_the_fold_turnover(history_entry):
    _, entry, _ = history_entry
    notional = (entry["trade_shares"] * entry["trade_prices"]).abs().sum(axis=1)
    pd.testing.assert_series_equal(
        notional / entry["fold_capital"], entry["fold_turnovers"]
    )


def test_target_book_is_sized_off_the_fold_capital(
    make_backtester, constant_signal_fn
):
    # A fully invested long-only target puts the whole book to work, so the
    # positions at each rebalance are worth the capital they were sized from.
    backtester = make_backtester(
        signal_fn=lambda cfg, data, *args: constant_signal_fn(cfg, data, 1)
    )
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    entry = backtester.runs[1]
    invested = (entry["target_shares"] * entry["trade_prices"]).sum(axis=1)
    pd.testing.assert_series_equal(invested, entry["fold_capital"])


def test_blotter_expands_to_a_daily_path(history_entry):
    # Holdings forward-fill between rebalances; trades do not.
    _, entry, _ = history_entry
    equity_index = entry["equity_curve"].index
    holdings = entry["target_shares"].reindex(equity_index).ffill()
    trades = entry["trade_shares"].reindex(equity_index).fillna(0.0)

    assert not holdings.isna().any().any()
    assert holdings.index.equals(equity_index)
    # Trading only happens on the rebalance dates.
    traded_on = trades.abs().sum(axis=1) > 0
    assert set(trades.index[traded_on]) <= set(entry["trade_shares"].index)


def test_held_positions_reconcile_with_the_reported_weights(history_entry):
    # The blotter and the weight path are two views of the same book: shares
    # held, marked at that bar, over that bar's equity.
    backtester, entry, _ = history_entry
    equity = entry["equity_curve"]
    holdings = entry["target_shares"].reindex(equity.index).ffill()
    prices = backtester.price_data.loc[equity.index]
    implied = (holdings * prices).div(equity, axis=0)
    pd.testing.assert_frame_equal(implied, entry["oos_weights"])


# ---------------------------------------------------------------------------
# clear_runs()
# ---------------------------------------------------------------------------


def test_clear_runs_empties_the_whole_history(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs()
    assert backtester.runs == {}
    assert backtester.run_count == 0


def test_clear_runs_drops_the_oldest_runs_first(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs(2)
    assert list(backtester.runs) == [3]


def test_clear_runs_leaves_the_survivors_untouched(backtester):
    backtester.run([5], PORTFOLIO_ARGS)
    backtester.run([10], PORTFOLIO_ARGS)
    survivor = dict(backtester.runs[2]["results"])
    backtester.clear_runs(1)
    assert backtester.runs[2]["results"] == pytest.approx(survivor)
    assert backtester.runs[2]["signal_args"] == [10]


def test_clear_runs_is_idempotent(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs(2)
    backtester.clear_runs(2)
    assert list(backtester.runs) == []


def test_clear_runs_tolerates_a_count_beyond_the_history(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs(99)
    assert backtester.runs == {}


def test_clear_runs_on_an_empty_history_is_a_no_op(backtester):
    backtester.clear_runs()
    backtester.clear_runs(3)
    assert backtester.runs == {}
    assert backtester.run_count == 0


def test_clear_runs_zero_keeps_everything(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs(0)
    assert list(backtester.runs) == [1]


def test_running_after_a_clear_records_a_fresh_history(backtester):
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs()
    results = backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert list(backtester.runs) == [1]
    assert backtester.runs[1]["results"] == pytest.approx(results)


def test_keys_never_collide_with_a_surviving_run(backtester):
    for _ in range(3):
        backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    backtester.clear_runs(2)
    surviving = set(backtester.runs)
    backtester.run(SIGNAL_ARGS, PORTFOLIO_ARGS)
    assert len(backtester.runs) == len(surviving) + 1
    assert surviving < set(backtester.runs)
