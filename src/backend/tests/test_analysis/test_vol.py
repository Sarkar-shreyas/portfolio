import numpy as np
import pandas as pd
import pytest

from arch.univariate.base import ARCHModelResult

from src.backend.analysis.vol import (
    ann_volatility,
    rolling_volatility,
    ewma_volatility,
    fit_garch,
)


# ---------------------------------------------------------------------------
# ann_volatility()
# ---------------------------------------------------------------------------


def test_ann_volatility_matches_manual_formula(config, returns_series):
    result = ann_volatility(config, returns_series)
    expected = returns_series.std() * np.sqrt(config.annualise)
    assert result == pytest.approx(expected)


def test_ann_volatility_custom_ann_factor(config, returns_series):
    result = ann_volatility(config, returns_series, ann=12)
    expected = returns_series.std() * np.sqrt(12)
    assert result == pytest.approx(expected)


def test_ann_volatility_zero_for_constant_series(config):
    data = pd.Series([0.01] * 30)
    result = ann_volatility(config, data)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# rolling_volatility()
# ---------------------------------------------------------------------------


def test_rolling_volatility_matches_manual_rolling_std(config, returns_series):
    window = 21
    result = rolling_volatility(config, returns_series, period=window, metric="std")
    expected = returns_series.rolling(window).agg("std").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_volatility_default_metric_is_std(config, returns_series):
    result = rolling_volatility(config, returns_series, period=10)
    expected = returns_series.rolling(10).agg("std").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_volatility_is_never_negative(config, returns_series):
    # A volatility estimate is a dispersion measure, so it cannot be negative.
    # Guards the default metric: a rolling *mean* of returns goes negative
    # roughly half the time, which silently flips inverse-volatility weights.
    result = rolling_volatility(config, returns_series, period=10)
    assert (result >= 0).all()


def test_rolling_volatility_default_period_uses_config(config, returns_series):
    result = rolling_volatility(config, returns_series)
    expected = rolling_volatility(config, returns_series, period=config.sma_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# ewma_volatility()
# ---------------------------------------------------------------------------


def test_ewma_volatility_matches_manual_ewm(config, returns_series):
    span = 21
    result = ewma_volatility(config, returns_series, span=span, metric="std")
    expected = returns_series.ewm(span=span, adjust=False).agg("std").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_ewma_volatility_default_metric_is_std(config, returns_series):
    result = ewma_volatility(config, returns_series, span=10)
    expected = returns_series.ewm(span=10, adjust=False).agg("std").dropna()
    pd.testing.assert_series_equal(result, expected)


def test_ewma_volatility_is_never_negative(config, returns_series):
    # Same dispersion invariant as rolling_volatility.
    result = ewma_volatility(config, returns_series, span=10)
    assert (result >= 0).all()


def test_ewma_volatility_default_span_uses_config(config, returns_series):
    result = ewma_volatility(config, returns_series)
    expected = ewma_volatility(config, returns_series, span=config.ema_window)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# fit_garch()
# ---------------------------------------------------------------------------


@pytest.fixture
def garch_returns(rng) -> pd.Series:
    """Returns scaled to percentage units, as arch_model expects for stable fits."""
    n = 300
    return pd.Series(rng.normal(0.0, 1.0, size=n))


def test_fit_garch_returns_arch_model_result(config, garch_returns):
    model_params = {"p": 1, "q": 1, "vol": "GARCH", "dist": "normal"}
    fit_params = {"disp": "off"}
    result = fit_garch(config, garch_returns, model_params, fit_params)
    assert isinstance(result, ARCHModelResult)


def test_fit_garch_params_contain_expected_coefficients(config, garch_returns):
    model_params = {"p": 1, "q": 1, "vol": "GARCH", "dist": "normal"}
    fit_params = {"disp": "off"}
    result = fit_garch(config, garch_returns, model_params, fit_params)
    for expected_param in ("omega", "alpha[1]", "beta[1]"):
        assert expected_param in result.params.index


def test_fit_garch_conditional_volatility_matches_input_length(config, garch_returns):
    model_params = {"p": 1, "q": 1, "vol": "GARCH", "dist": "normal"}
    fit_params = {"disp": "off"}
    result = fit_garch(config, garch_returns, model_params, fit_params)
    assert len(result.conditional_volatility) == len(garch_returns)


def test_fit_garch_conditional_volatility_is_non_negative(config, garch_returns):
    model_params = {"p": 1, "q": 1, "vol": "GARCH", "dist": "normal"}
    fit_params = {"disp": "off"}
    result = fit_garch(config, garch_returns, model_params, fit_params)
    assert (result.conditional_volatility >= 0).all()
