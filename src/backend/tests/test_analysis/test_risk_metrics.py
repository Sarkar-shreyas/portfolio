import numpy as np
import pandas as pd
import pytest

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


def test_ann_sortino_uses_only_negative_returns(config):
    # With no negative returns, the downside-deviation term should be zero,
    # producing an infinite (or undefined) Sortino ratio.
    data = pd.Series([0.01] * 30)
    with np.errstate(divide="ignore"):
        result = ann_sortino(config, data)
    assert np.isinf(result) or np.isnan(result)


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
