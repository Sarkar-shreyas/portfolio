import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from src.backend.analysis.returns import ann_returns, cumulative_returns
from src.backend.analysis.vol import ann_volatility
from src.backend.analysis.risk_metrics import (
    get_correlation,
    rolling_correlation,
    get_covariance,
    rolling_covariance,
    ann_sharpe,
    rolling_sharpe,
    ann_sortino,
    rolling_sortino,
    daily_drawdowns,
    max_daily_drawdowns,
    max_drawdown,
    min_drawdown,
    rolling_drawdown,
    ann_calmar,
    probabilistic_sharpe,
    expected_max_sharpe,
    deflated_sharpe,
)


# ---------------------------------------------------------------------------
# get_correlation() / rolling_correlation()
# ---------------------------------------------------------------------------


def test_get_correlation_matches_pandas_corr(multi_asset_returns):
    result = get_correlation(multi_asset_returns)
    pd.testing.assert_frame_equal(result, multi_asset_returns.corr())


def test_get_correlation_diagonal_is_one(multi_asset_returns):
    result = get_correlation(multi_asset_returns)
    np.testing.assert_allclose(np.diag(result.values), 1.0)


def test_get_correlation_is_symmetric(multi_asset_returns):
    result = get_correlation(multi_asset_returns)
    np.testing.assert_allclose(result.values, result.values.T)


def test_rolling_correlation_matches_manual_computation(config, multi_asset_returns):
    window = 20
    cols = ["AssetA", "AssetB"]
    result = rolling_correlation(config, multi_asset_returns, cols, window=window)
    expected = (
        multi_asset_returns[cols[0]].rolling(window).corr(multi_asset_returns[cols[1]])
    )
    pd.testing.assert_series_equal(result, expected)


def test_rolling_correlation_default_window_uses_config(config, multi_asset_returns):
    cols = ["AssetA", "AssetC"]
    result = rolling_correlation(config, multi_asset_returns, cols)
    expected = rolling_correlation(
        config, multi_asset_returns, cols, window=config.sma_window
    )
    pd.testing.assert_series_equal(result, expected)


def test_rolling_correlation_raises_for_fewer_than_two_columns(
    config, multi_asset_returns
):
    with pytest.raises(ValueError):
        rolling_correlation(config, multi_asset_returns, ["AssetA"])


# ---------------------------------------------------------------------------
# get_covariance() / rolling_covariance()
# ---------------------------------------------------------------------------


def test_get_covariance_matches_pandas_cov(multi_asset_returns):
    result = get_covariance(multi_asset_returns)
    pd.testing.assert_frame_equal(result, multi_asset_returns.cov())


def test_get_covariance_is_symmetric(multi_asset_returns):
    result = get_covariance(multi_asset_returns)
    np.testing.assert_allclose(result.values, result.values.T)


def test_rolling_covariance_matches_manual_computation(config, multi_asset_returns):
    window = 20
    cols = ["AssetA", "AssetB"]
    result = rolling_covariance(config, multi_asset_returns, cols, window=window)
    expected = (
        multi_asset_returns[cols[0]].rolling(window).cov(multi_asset_returns[cols[1]])
    )
    pd.testing.assert_series_equal(result, expected)


def test_rolling_covariance_raises_for_fewer_than_two_columns(
    config, multi_asset_returns
):
    with pytest.raises(ValueError):
        rolling_covariance(config, multi_asset_returns, ["AssetB"])


# ---------------------------------------------------------------------------
# ann_sharpe()
# ---------------------------------------------------------------------------


def test_ann_sharpe_matches_manual_composition(config, returns_series):
    result = ann_sharpe(config, returns_series)
    ann_ret = ann_returns(config, returns_series)
    ann_vol = ann_volatility(config, returns_series)
    expected = (ann_ret - config.risk_free_rate) / ann_vol
    assert result == pytest.approx(expected)


def test_ann_sharpe_is_nan_for_a_flat_book(config):
    # A zero-weight backtest produces exactly-zero returns.
    data = pd.Series([0.0] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ann_sharpe(config, data)
    assert np.isnan(result)


def test_ann_sharpe_is_nan_for_a_constant_nonzero_return(config):
    # The sample std of a constant series is ~1e-19 in floating point rather
    # than exactly 0; that must not be read as a finite, enormous Sharpe.
    data = pd.Series([0.001] * 252)
    assert np.isnan(ann_sharpe(config, data))


# ---------------------------------------------------------------------------
# rolling_sharpe()
# ---------------------------------------------------------------------------


def test_rolling_sharpe_matches_manual_composition(config, returns_series):
    window = 30
    result = rolling_sharpe(config, returns_series, window=window)

    daily_rf = (1 + config.risk_free_rate) ** (1 / config.annualise) - 1
    excess_ret = returns_series - daily_rf
    rolling_ret = excess_ret.rolling(window).agg("mean").dropna()
    rolling_vol = excess_ret.rolling(window).agg("std").dropna()
    expected = rolling_ret / rolling_vol

    pd.testing.assert_series_equal(result, expected)


def test_rolling_sharpe_is_not_degenerate(config, returns_series):
    # The ratio must vary with the data. If the numerator and denominator are
    # built from the same statistic the series collapses to a constant 1.0,
    # which is the failure this pins.
    result = rolling_sharpe(config, returns_series, window=30)
    assert result.nunique() > 1
    assert result.std() > 0


def test_rolling_sharpe_daily_rf_compounds_to_the_annual_rate(config):
    # The daily risk-free rate must compound back to the configured annual rate
    # over one year. Guards the operator-precedence bug that made it ~-0.996.
    daily_rf = (1 + config.risk_free_rate) ** (1 / config.annualise) - 1
    assert (1 + daily_rf) ** config.annualise == pytest.approx(
        1 + config.risk_free_rate
    )

    # A return series sitting exactly at the risk-free rate has zero excess
    # return, so the rolling Sharpe numerator -- and the ratio -- must be zero.
    flat = pd.Series([daily_rf] * 60, index=pd.bdate_range("2024-01-01", periods=60))
    result = rolling_sharpe(config, flat, window=30)
    assert result.dropna().eq(0).all() or result.dropna().abs().max() < 1e-9


def test_rolling_sharpe_scales_with_excess_return(config, returns_series):
    # Shifting every return up by a constant raises the numerator while leaving
    # the rolling standard deviation untouched, so the ratio must rise.
    base = rolling_sharpe(config, returns_series, window=30)
    lifted = rolling_sharpe(config, returns_series + 0.01, window=30)
    assert (lifted > base).all()


def test_rolling_sharpe_default_window_uses_sharpe_window(config, returns_series):
    result = rolling_sharpe(config, returns_series)
    expected = rolling_sharpe(config, returns_series, window=config.sharpe_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# ann_sortino()
# ---------------------------------------------------------------------------


def test_ann_sortino_matches_manual_composition(config, returns_series):
    result = ann_sortino(config, returns_series)

    loss = returns_series.apply(lambda x: x if x < 0 else 0)
    ann_ret = ann_returns(config, returns_series)
    loss_vol = ann_volatility(config, loss)
    expected = (ann_ret - config.risk_free_rate) / loss_vol

    assert result == pytest.approx(expected)


def test_ann_sortino_is_nan_with_no_losing_days(config):
    # No negative returns means zero downside deviation; the ratio is undefined
    # and should be NaN (matching ann_calmar), not inf with a RuntimeWarning.
    data = pd.Series([0.01, 0.02, 0.0, 0.03] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ann_sortino(config, data)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# rolling_sortino()
# ---------------------------------------------------------------------------


def test_rolling_sortino_matches_manual_composition(config, returns_series):
    window = 30
    result = rolling_sortino(config, returns_series, window=window)

    daily_rf = (1 + config.risk_free_rate) ** (1 / config.annualise) - 1
    excess_ret = returns_series - daily_rf
    loss = returns_series.apply(lambda x: x if x < 0 else 0)
    rolling_ret = excess_ret.rolling(window).agg("mean").dropna()
    loss_vol = loss.rolling(window).agg("std").dropna()
    expected = rolling_ret / loss_vol

    pd.testing.assert_series_equal(result, expected)


def test_rolling_sortino_is_not_degenerate(config, returns_series):
    result = rolling_sortino(config, returns_series, window=30)
    assert result.nunique() > 1
    assert result.std() > 0


def test_rolling_sortino_denominator_ignores_gains(config):
    # Downside deviation must not react to upside dispersion. Two series with
    # identical losses but larger gains share a denominator, so the one with the
    # bigger gains must score strictly higher everywhere.
    idx = pd.bdate_range("2024-01-01", periods=60)
    modest = pd.Series([-0.01, 0.005] * 30, index=idx)
    generous = pd.Series([-0.01, 0.050] * 30, index=idx)

    modest_result = rolling_sortino(config, modest, window=30).dropna()
    generous_result = rolling_sortino(config, generous, window=30).dropna()

    assert (generous_result > modest_result).all()


# ---------------------------------------------------------------------------
# daily_drawdowns() / max_daily_drawdowns() / max_drawdown() / min_drawdown()
# ---------------------------------------------------------------------------


def test_daily_drawdowns_matches_manual_formula(config, returns_series):
    result = daily_drawdowns(config, returns_series)
    wealth = cumulative_returns(returns_series) + 1
    running_max = wealth.cummax()
    expected = wealth / running_max - 1
    pd.testing.assert_series_equal(result, expected)


def test_max_daily_drawdowns_matches_manual_rolling_min(config, returns_series):
    result = max_daily_drawdowns(config, returns_series)
    drawdowns = daily_drawdowns(config, returns_series)
    expected = drawdowns.rolling(min_periods=1, window=config.annualise).min()
    pd.testing.assert_series_equal(result, expected)


def test_max_drawdown_is_minimum_of_daily_drawdowns(config, returns_series):
    result = max_drawdown(config, returns_series)
    drawdowns = daily_drawdowns(config, returns_series)
    assert result == pytest.approx(drawdowns.min())


def test_min_drawdown_is_maximum_of_daily_drawdowns(config, returns_series):
    result = min_drawdown(config, returns_series)
    drawdowns = daily_drawdowns(config, returns_series)
    assert result == pytest.approx(drawdowns.max())


def test_max_drawdown_le_min_drawdown(config, returns_series):
    assert max_drawdown(config, returns_series) <= min_drawdown(config, returns_series)


# ---------------------------------------------------------------------------
# rolling_drawdown()
# ---------------------------------------------------------------------------


def test_rolling_drawdown_matches_manual_computation(config, returns_series):
    window = 15
    result = rolling_drawdown(config, returns_series, window=window, metric="mean")
    drawdowns = daily_drawdowns(config, returns_series)
    expected = drawdowns.rolling(window).agg("mean").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_drawdown_default_window_uses_config(config, returns_series):
    result = rolling_drawdown(config, returns_series)
    expected = rolling_drawdown(config, returns_series, window=config.sma_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# ann_calmar()
# ---------------------------------------------------------------------------


def test_ann_calmar_matches_manual_composition(config, returns_series):
    result = ann_calmar(config, returns_series)
    ann_ret = ann_returns(config, returns_series)
    max_draw = max_drawdown(config, returns_series)
    expected = (ann_ret - config.risk_free_rate) / np.abs(max_draw)
    assert result == pytest.approx(expected)


def test_ann_calmar_returns_nan_for_zero_drawdown(config):
    zero_returns_series = pd.Series(0)
    result = ann_calmar(config, zero_returns_series)
    assert result is np.nan


# ---------------------------------------------------------------------------
# probabilistic_sharpe()
# ---------------------------------------------------------------------------


def _sharpe_by_definition(config, data: pd.Series) -> float:
    """Per-period Sharpe: mean excess return over the sample std (ddof=1)."""
    daily_rf = (1 + config.risk_free_rate) ** (1 / config.annualise) - 1
    return (data.mean() - daily_rf) / data.std(ddof=1)


def _two_point(mean: float, spread: float, n_pairs: int) -> pd.Series:
    """mean +/- spread with equal weight: skew exactly 0, raw kurtosis exactly 1."""
    return pd.Series(mean + spread * np.tile([1.0, -1.0], n_pairs))


def test_psr_is_one_half_against_its_own_sharpe(config, returns_series):
    own = _sharpe_by_definition(config, returns_series)
    assert probabilistic_sharpe(config, returns_series, own) == pytest.approx(0.5)


def test_psr_default_benchmark_is_zero(config, returns_series):
    assert probabilistic_sharpe(config, returns_series) == probabilistic_sharpe(
        config, returns_series, 0.0
    )


def test_psr_matches_the_closed_form_for_a_two_point_distribution(config):
    # With skew 0 and raw kurtosis 1 the variance term is exactly 1, so
    # PSR = Phi((SR - SR*) * sqrt(T - 1)). Using excess kurtosis (-2) instead
    # would give 1 - 3/4 SR^2 and fail this.
    data = _two_point(0.001, 0.01, 200)
    sharpe = _sharpe_by_definition(config, data)
    benchmark = 0.02
    expected = norm.cdf((sharpe - benchmark) * np.sqrt(len(data) - 1))
    assert probabilistic_sharpe(config, data, benchmark) == pytest.approx(
        expected, rel=1e-12
    )


def test_psr_penalises_negative_skew(config, rng):
    # Mirroring a series about its mean flips the sign of its skew and leaves
    # the mean, std and kurtosis unchanged.
    right_skewed = pd.Series(0.002 + 0.01 * (rng.exponential(size=500) - 1))
    left_skewed = 2 * right_skewed.mean() - right_skewed
    assert _sharpe_by_definition(config, right_skewed) > 0
    assert probabilistic_sharpe(config, left_skewed) < probabilistic_sharpe(
        config, right_skewed
    )


def test_psr_penalises_fat_tails(config):
    # Same length, mean and variance; raw kurtosis 1 versus 4.
    spread, mean = 0.01, 0.002
    thin = pd.Series(mean + spread * np.tile([1, -1, 1, -1, 1, -1, 1, -1], 50))
    fat = pd.Series(mean + 2 * spread * np.tile([1, 0, 0, 0, -1, 0, 0, 0], 50))
    assert _sharpe_by_definition(config, thin) == pytest.approx(
        _sharpe_by_definition(config, fat)
    )
    assert probabilistic_sharpe(config, fat) < probabilistic_sharpe(config, thin)


def test_psr_rises_with_track_record_length(config):
    short = _two_point(0.002, 0.01, 50)
    long = _two_point(0.002, 0.01, 200)
    assert probabilistic_sharpe(config, long) > probabilistic_sharpe(config, short)


def test_psr_skips_the_leading_nan_of_simple_returns(config, returns_series):
    values = returns_series.reset_index(drop=True)
    with_nan = pd.concat([pd.Series([np.nan]), values], ignore_index=True)
    assert probabilistic_sharpe(config, with_nan) == probabilistic_sharpe(
        config, values
    )


def test_psr_is_nan_for_a_flat_series(config):
    assert np.isnan(probabilistic_sharpe(config, pd.Series([0.0] * 30)))


def test_psr_raises_below_three_observations(config):
    with pytest.raises(ValueError):
        probabilistic_sharpe(config, pd.Series([0.01, -0.01]))


# ---------------------------------------------------------------------------
# expected_max_sharpe()
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_is_zero_for_a_single_trial(config):
    assert expected_max_sharpe(config, pd.Series([0.05])) == 0.0
    assert expected_max_sharpe(config, pd.Series([0.05, 0.02]), n_trials=1) == 0.0


def test_expected_max_sharpe_raises_without_a_variance_estimate(config):
    with pytest.raises(ValueError):
        expected_max_sharpe(config, pd.Series([0.05]), n_trials=5)


def test_expected_max_sharpe_scales_with_the_sharpe_dispersion(config, rng):
    # SR0 is linear in sqrt(V): doubling every Sharpe doubles SR0 exactly.
    sharpes = pd.Series(rng.normal(0, 0.02, 20))
    assert expected_max_sharpe(config, 2 * sharpes) == pytest.approx(
        2 * expected_max_sharpe(config, sharpes), rel=1e-12
    )


def test_expected_max_sharpe_rises_with_the_number_of_trials(config, rng):
    sharpes = pd.Series(rng.normal(0, 0.02, 20))
    values = [expected_max_sharpe(config, sharpes, n) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("n_trials", [10, 100, 1000])
def test_expected_max_sharpe_approximates_the_maximum_of_standard_normals(
    config, n_trials
):
    # Unit-variance Sharpes, so SR0 should approximate E[max of N iid N(0,1)],
    # computed here by numerical integration. The tolerance is the known error
    # of the Bailey & Lopez de Prado approximation (~2% at N=10, <1% beyond),
    # not a fitted constant: writing N*e as N+e is ~12% off and fails.
    unit_variance = pd.Series([-1.0, 1.0]) / np.sqrt(2)
    exact = integrate.quad(
        lambda x: x * n_trials * norm.pdf(x) * norm.cdf(x) ** (n_trials - 1), -10, 10
    )[0]
    result = expected_max_sharpe(config, unit_variance, n_trials=n_trials)
    assert result == pytest.approx(exact, rel=0.03)


def test_expected_max_sharpe_ignores_nan_trials(config):
    sharpes = pd.Series([0.01, 0.03, np.nan, 0.02])
    assert expected_max_sharpe(config, sharpes) == expected_max_sharpe(
        config, sharpes.dropna()
    )


# ---------------------------------------------------------------------------
# deflated_sharpe()
# ---------------------------------------------------------------------------


@pytest.fixture
def trial_returns(rng) -> pd.DataFrame:
    """Twenty noise trials over three years; trial_0 has a genuine edge."""
    dates = pd.bdate_range("2022-01-03", periods=756)
    trials = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(756, 20)),
        index=dates,
        columns=[f"trial_{i}" for i in range(20)],
    )
    trials["trial_0"] += 0.0008
    return trials


def test_dsr_of_a_single_trial_is_its_psr_against_zero(config, trial_returns):
    single = trial_returns[["trial_0"]]
    assert deflated_sharpe(config, single) == probabilistic_sharpe(
        config, trial_returns["trial_0"], 0.0
    )


def test_dsr_is_lower_than_the_undeflated_psr(config, trial_returns):
    dsr = deflated_sharpe(config, trial_returns)
    psr = probabilistic_sharpe(config, trial_returns["trial_0"])
    assert 0.0 <= dsr < psr <= 1.0


def test_dsr_defaults_to_the_highest_sharpe_trial(config, trial_returns):
    sharpes = trial_returns.apply(lambda col: _sharpe_by_definition(config, col))
    best = sharpes.idxmax()
    assert deflated_sharpe(config, trial_returns) == deflated_sharpe(
        config, trial_returns, selected=best
    )


def test_dsr_is_the_selected_psr_against_the_expected_max_sharpe(
    config, trial_returns
):
    sharpes = trial_returns.apply(lambda col: _sharpe_by_definition(config, col))
    benchmark = expected_max_sharpe(config, sharpes)
    expected = probabilistic_sharpe(config, trial_returns["trial_5"], benchmark)
    assert deflated_sharpe(config, trial_returns, selected="trial_5") == pytest.approx(
        expected, rel=1e-12
    )


def test_dsr_falls_as_more_independent_trials_are_assumed(config, trial_returns):
    values = [deflated_sharpe(config, trial_returns, n_trials=n) for n in (5, 20, 200)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_dsr_raises_for_an_unknown_trial(config, trial_returns):
    with pytest.raises(KeyError):
        deflated_sharpe(config, trial_returns, selected="not_a_trial")


def test_dsr_drops_zero_volatility_trials(config, trial_returns, capsys):
    with_flat = trial_returns.assign(flat=0.0)
    assert deflated_sharpe(config, with_flat) == deflated_sharpe(config, trial_returns)
    assert "flat" in capsys.readouterr().out


def test_dsr_uses_only_the_dates_all_trials_share(config, trial_returns, capsys):
    ragged = trial_returns.copy()
    ragged.iloc[:100, 3] = np.nan
    assert deflated_sharpe(config, ragged) == deflated_sharpe(
        config, trial_returns.iloc[100:]
    )
    assert "656" in capsys.readouterr().out


def test_dsr_is_nan_when_every_trial_is_flat(config, trial_returns):
    flat = pd.DataFrame(0.0, index=trial_returns.index, columns=["a", "b"])
    assert np.isnan(deflated_sharpe(config, flat))
