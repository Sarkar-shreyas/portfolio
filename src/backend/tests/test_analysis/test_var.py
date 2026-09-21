import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from src.backend.analysis.var import (
    est_var,
    cond_var,
    norm_parametric_var,
    t_parametric_var,
)


# ---------------------------------------------------------------------------
# est_var()
# ---------------------------------------------------------------------------


def test_est_var_matches_manual_percentile(config, returns_series):
    result = est_var(config, returns_series)
    expected = np.percentile(returns_series, (1 - config.var_conf) * 100)
    assert result == pytest.approx(expected)


def test_est_var_custom_confidence(config, returns_series):
    result = est_var(config, returns_series, conf=0.99)
    expected = np.percentile(returns_series, 1)
    assert result == pytest.approx(expected)


def test_est_var_known_values():
    class _Cfg:
        var_conf = 0.9

    data = pd.Series(np.arange(1, 101))  # 1..100
    result = est_var(_Cfg(), data)
    expected = np.percentile(data, 10)
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# cond_var()
# ---------------------------------------------------------------------------


def test_cond_var_matches_manual_computation(config, returns_series):
    result = cond_var(config, returns_series)
    var = est_var(config, returns_series)
    lower = returns_series[returns_series < var]
    expected = lower.mean()
    assert result == pytest.approx(expected)


def test_cond_var_is_more_extreme_than_est_var(config, returns_series):
    var = est_var(config, returns_series)
    cvar = cond_var(config, returns_series)
    # The conditional VaR averages the tail beyond the VaR threshold, so it
    # should be at least as extreme (<=) as the VaR estimate itself.
    assert cvar <= var


def test_cond_var_nan_when_no_data_beyond_threshold(config):
    # If nothing falls below the VaR threshold, the mean of an empty slice
    # is NaN.
    class _Cfg:
        var_conf = 0.5

    data = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = cond_var(_Cfg(), data)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# norm_parametric_var()
# ---------------------------------------------------------------------------


def test_norm_parametric_var_matches_manual_formula(config, returns_series):
    window = 21
    result = norm_parametric_var(config, returns_series, window=window)

    rolling_data = returns_series.rolling(window)
    mu = rolling_data.mean()
    sigma = rolling_data.std(ddof=1)
    expected = -norm.ppf(config.var_conf, loc=mu, scale=sigma)

    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_norm_parametric_var_leading_values_are_nan(config, returns_series):
    window = 21
    result = norm_parametric_var(config, returns_series, window=window)
    assert np.all(np.isnan(result[: window - 1]))


def test_norm_parametric_var_output_length_matches_input(config, returns_series):
    result = norm_parametric_var(config, returns_series)
    assert len(result) == len(returns_series)


def test_norm_parametric_var_default_window_and_conf_use_config(config, returns_series):
    result = norm_parametric_var(config, returns_series)
    expected = norm_parametric_var(
        config, returns_series, conf=config.var_conf, window=config.sma_window
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# t_parametric_var()
# ---------------------------------------------------------------------------


def test_t_parametric_var_matches_manual_formula(config, returns_series):
    window = 21
    dof = 5
    result = t_parametric_var(config, returns_series, window=window, dof=dof)

    rescale_factor = np.sqrt(dof / (dof - 2))
    rolling_data = returns_series.rolling(window)
    mu = rolling_data.mean()
    sigma = rolling_data.std(ddof=1)
    expected = rescale_factor * -t.ppf(1 - config.var_conf, df=dof, loc=mu, scale=sigma)

    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_t_parametric_var_output_length_matches_input(config, returns_series):
    result = t_parametric_var(config, returns_series)
    assert len(result) == len(returns_series)


def test_t_parametric_var_default_dof_is_5(config, returns_series):
    result = t_parametric_var(config, returns_series)
    expected = t_parametric_var(config, returns_series, dof=5)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_t_parametric_var_leading_values_are_nan(config, returns_series):
    window = 30
    result = t_parametric_var(config, returns_series, window=window)
    assert np.all(np.isnan(result[: window - 1]))
