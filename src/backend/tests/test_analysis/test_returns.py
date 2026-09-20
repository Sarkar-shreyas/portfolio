import numpy as np
import pandas as pd
import pytest

from src.backend.analysis.returns import (
    returns,
    log_returns,
    cumulative_returns,
    ann_returns,
    rolling_returns,
    sma_returns,
    ema_returns,
    simple_rsi,
    ewm_rsi,
)


# ---------------------------------------------------------------------------
# returns()
# ---------------------------------------------------------------------------


def test_returns_matches_pct_change(price_series):
    result = returns(price_series)
    pd.testing.assert_series_equal(result, price_series.pct_change())


def test_returns_first_value_is_nan(price_series):
    result = returns(price_series)
    assert np.isnan(result.iloc[0])


def test_returns_known_values():
    prices = pd.Series([100.0, 110.0, 121.0, 108.9])
    result = returns(prices)
    expected = pd.Series([np.nan, 0.10, 0.10, -0.10])
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# log_returns()
# ---------------------------------------------------------------------------


def test_log_returns_matches_manual_formula(price_series):
    result = log_returns(price_series)
    expected = np.log(price_series) - np.log(price_series.shift(1))
    pd.testing.assert_series_equal(result, expected)


def test_log_returns_close_to_simple_returns_for_small_moves(price_series):
    simple = returns(price_series).dropna()
    log_ret = log_returns(price_series).dropna()
    # For small daily moves, log returns approximate simple returns.
    np.testing.assert_allclose(log_ret.values, simple.values, atol=1e-3)


# ---------------------------------------------------------------------------
# cumulative_returns()
# ---------------------------------------------------------------------------


def test_cumulative_returns_known_values():
    rets = pd.Series([0.1, 0.2, -0.1])
    result = cumulative_returns(rets)
    expected = pd.Series(
        [1.1 - 1, 1.1 * 1.2 - 1, 1.1 * 1.2 * 0.9 - 1]
    )
    pd.testing.assert_series_equal(result, expected)


def test_cumulative_returns_matches_price_growth(price_series, returns_series):
    cum = cumulative_returns(returns_series)
    # returns_series covers price_series[0] -> price_series[1], ..., up to
    # price_series[-2] -> price_series[-1], so the compounded cumulative
    # return should equal the total growth across the whole price series.
    expected_total_growth = price_series.iloc[-1] / price_series.iloc[0] - 1
    assert cum.iloc[-1] == pytest.approx(expected_total_growth)


# ---------------------------------------------------------------------------
# ann_returns()
# ---------------------------------------------------------------------------


def test_ann_returns_with_ann_equal_to_n_days_matches_total_return(config, returns_series):
    n_days = len(returns_series)
    result = ann_returns(config, returns_series, ann=n_days)
    expected_total = cumulative_returns(returns_series).iloc[-1]
    assert result == pytest.approx(expected_total)


def test_ann_returns_uses_config_annualise_by_default(config, returns_series):
    result = ann_returns(config, returns_series)
    expected = ann_returns(config, returns_series, ann=config.annualise)
    assert result == pytest.approx(expected)


def test_ann_returns_constant_daily_return():
    daily_r = 0.001
    n_days = 100
    data = pd.Series([daily_r] * n_days)

    class _Cfg:
        annualise = 252

    result = ann_returns(_Cfg(), data)
    total_return = (1 + daily_r) ** n_days - 1
    expected = (1 + total_return) ** (252 / n_days) - 1
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# rolling_returns()
# ---------------------------------------------------------------------------


def test_rolling_returns_matches_manual_rolling_mean(config, returns_series):
    window = 10
    result = rolling_returns(config, returns_series, window=window, metric="mean")
    expected = returns_series.rolling(window).agg("mean").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_returns_with_sum_metric(config, returns_series):
    window = 5
    result = rolling_returns(config, returns_series, window=window, metric="sum")
    expected = returns_series.rolling(window).agg("sum").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_returns_default_window_uses_config_sma_window(config, returns_series):
    result = rolling_returns(config, returns_series)
    expected = rolling_returns(config, returns_series, window=config.sma_window)
    pd.testing.assert_series_equal(result, expected)


def test_rolling_returns_output_length(config, returns_series):
    window = 20
    result = rolling_returns(config, returns_series, window=window)
    assert len(result) == len(returns_series) - window + 1


# ---------------------------------------------------------------------------
# sma_returns()
# ---------------------------------------------------------------------------


def test_sma_returns_matches_manual_rolling_mean(config, returns_series):
    period = 15
    result = sma_returns(config, returns_series, period=period)
    expected = returns_series.rolling(period).mean().dropna()
    pd.testing.assert_series_equal(result, expected)


def test_sma_returns_default_period_uses_config(config, returns_series):
    result = sma_returns(config, returns_series)
    expected = sma_returns(config, returns_series, period=config.sma_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# ema_returns()
# ---------------------------------------------------------------------------


def test_ema_returns_matches_pandas_ewm(config, returns_series):
    span = 12
    result = ema_returns(config, returns_series, span=span)
    expected = returns_series.ewm(span=span, adjust=False).mean().dropna()
    pd.testing.assert_series_equal(result, expected)


def test_ema_returns_default_span_uses_config(config, returns_series):
    result = ema_returns(config, returns_series)
    expected = ema_returns(config, returns_series, span=config.ema_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# simple_rsi()
# ---------------------------------------------------------------------------


def test_simple_rsi_all_gains_is_100(config):
    data = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.015, 0.025])
    result = simple_rsi(config, data, window=3)
    assert (result.dropna() == 100.0).all()


def test_simple_rsi_all_losses_is_0(config):
    data = pd.Series([-0.01, -0.02, -0.03, -0.01, -0.02, -0.015, -0.025])
    result = simple_rsi(config, data, window=3)
    assert (result.dropna() == 0.0).all()


def test_simple_rsi_known_values(config):
    data = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    result = simple_rsi(config, data, window=1)
    expected = pd.Series([100.0, 0.0, 100.0, 0.0], index=[1, 2, 3, 4])
    pd.testing.assert_series_equal(result, expected)


def test_simple_rsi_bounded_between_0_and_100(config, returns_series):
    result = simple_rsi(config, returns_series)
    assert result.dropna().between(0, 100).all()


# ---------------------------------------------------------------------------
# ewm_rsi()
# ---------------------------------------------------------------------------


def test_ewm_rsi_all_gains_is_100(config):
    data = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.015, 0.025])
    result = ewm_rsi(config, data, span=3)
    assert (result.dropna() == 100.0).all()


def test_ewm_rsi_all_losses_is_0(config):
    data = pd.Series([-0.01, -0.02, -0.03, -0.01, -0.02, -0.015, -0.025])
    result = ewm_rsi(config, data, span=3)
    assert (result.dropna() == 0.0).all()


def test_ewm_rsi_bounded_between_0_and_100(config, returns_series):
    result = ewm_rsi(config, returns_series)
    assert result.dropna().between(0, 100).all()


def test_ewm_rsi_default_span_uses_config(config, returns_series):
    result = ewm_rsi(config, returns_series)
    expected = ewm_rsi(config, returns_series, span=config.ema_window)
    pd.testing.assert_series_equal(result, expected)
