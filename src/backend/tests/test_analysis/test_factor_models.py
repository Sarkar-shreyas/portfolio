"""Tests for ``src.backend.analysis.factor_models``.

Expected values are built independently of statsmodels: coefficients from the
normal equations, standard errors from the closed-form OLS and Newey-West
sandwich estimators, and inference columns from the t distribution. None of
the tests re-run the production ``ols(...).fit(...)`` call as its own oracle.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backend.analysis.factor_models import (
    regress,
    capm_regression,
    fama_french_three,
    fama_french_five,
)


@pytest.fixture
def regression_data(rng) -> pd.DataFrame:
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + noise
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2})


@pytest.fixture
def factor_data(rng) -> pd.DataFrame:
    """Synthetic factor-model data with the columns used across CAPM/FF regressions.

    The "Mkt" column is assumed to already be the excess market return
    (i.e. Mkt-RF), produced by upstream data pre-cleaning, so no separate
    "RF" column is needed here.
    """
    n = 200
    mkt = rng.normal(0.0005, 0.01, size=n)
    smb = rng.normal(0, 0.005, size=n)
    hml = rng.normal(0, 0.005, size=n)
    rmw = rng.normal(0, 0.004, size=n)
    cma = rng.normal(0, 0.004, size=n)
    noise = rng.normal(0, 0.002, size=n)
    target = 0.0002 + 1.1 * mkt + 0.3 * smb - 0.2 * hml + 0.15 * rmw - 0.1 * cma + noise
    return pd.DataFrame(
        {
            "target": target,
            "Mkt": mkt,
            "SMB": smb,
            "HML": hml,
            "RMW": rmw,
            "CMA": cma,
        }
    )


@pytest.fixture
def autocorrelated_data(rng) -> pd.DataFrame:
    """A persistent regressor with persistent errors: the case HAC exists for."""
    n = 500
    phi = 0.9
    x = np.zeros(n)
    u = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
        u[i] = phi * u[i - 1] + rng.normal()
    return pd.DataFrame({"Y": 0.5 * x + u, "X": x})


def _design(data: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = data[cols[0]].to_numpy()
    X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in cols[1:]])
    return X, y


def _ols_by_hand(data: pd.DataFrame, cols: list[str]) -> dict:
    """Coefficients, residuals and nonrobust standard errors from the normal equations."""
    X, y = _design(data, cols)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - k)
    return {
        "coef": beta,
        "resid": resid,
        "fitted": X @ beta,
        "se": np.sqrt(np.diag(sigma2 * xtx_inv)),
    }


def _newey_west_se_by_hand(data: pd.DataFrame, cols: list[str], lags: int) -> np.ndarray:
    """Bartlett-weighted HAC sandwich with the n / (n - k) small-sample correction."""
    X, y = _design(data, cols)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    resid = y - X @ (xtx_inv @ X.T @ y)
    scores = X * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = n / (n - k) * xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.diag(cov))


# ---------------------------------------------------------------------------
# Point estimates -- independent of the covariance choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["HAC", "nonrobust", "HC0"])
def test_regress_coefficients_solve_the_normal_equations(
    config, regression_data, cov_type
):
    summary, _ = regress(config, regression_data, ["Y", "X1", "X2"], cov_type=cov_type)
    expected = _ols_by_hand(regression_data, ["Y", "X1", "X2"])
    np.testing.assert_allclose(summary["coef"].values, expected["coef"], rtol=1e-12)


def test_regress_residuals_and_fitted_values_reconstruct_the_target(
    config, regression_data
):
    _, data_df = regress(config, regression_data, ["Y", "X1", "X2"])
    expected = _ols_by_hand(regression_data, ["Y", "X1", "X2"])
    np.testing.assert_allclose(data_df["resid"].values, expected["resid"], atol=1e-12)
    np.testing.assert_allclose(
        data_df["fittedvalues"].values, expected["fitted"], atol=1e-12
    )
    np.testing.assert_allclose(
        (data_df["fittedvalues"] + data_df["resid"]).values,
        data_df["actual"].values,
        atol=1e-12,
    )


def test_regress_hac_changes_standard_errors_but_not_coefficients(
    config, autocorrelated_data
):
    hac, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="HAC")
    plain, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="nonrobust")
    pd.testing.assert_series_equal(hac["coef"], plain["coef"])
    assert not np.allclose(hac["std err"].values, plain["std err"].values)


def test_regress_recovers_approximate_true_coefficients(config, regression_data):
    summary, _ = regress(config, regression_data, cols=["Y", "X1", "X2"])
    assert summary.loc["X1", "coef"] == pytest.approx(2.0, abs=0.2)
    assert summary.loc["X2", "coef"] == pytest.approx(-0.5, abs=0.2)


def test_regress_defaults_to_all_columns_when_cols_is_none(
    config, regression_data, capsys
):
    summary, _ = regress(config, regression_data, cols=None)
    assert list(summary.index) == ["Intercept", "X1", "X2"]
    assert "No columns were specified" in capsys.readouterr().out


def test_regress_raises_without_a_regressor(config, regression_data):
    with pytest.raises(ValueError, match="Insufficient number of factors"):
        regress(config, regression_data, cols=["Y"])


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------


def test_nonrobust_standard_errors_match_the_classical_formula(
    config, regression_data
):
    summary, _ = regress(
        config, regression_data, ["Y", "X1", "X2"], cov_type="nonrobust"
    )
    expected = _ols_by_hand(regression_data, ["Y", "X1", "X2"])["se"]
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


@pytest.mark.parametrize("lags", [0, 1, 5])
def test_hac_standard_errors_match_the_newey_west_sandwich(
    config, autocorrelated_data, lags
):
    # lags=0 is White's heteroskedasticity-robust estimator (with the n/(n-k)
    # correction); lags>0 adds the Bartlett-weighted autocovariance terms.
    summary, _ = regress(config, autocorrelated_data, ["Y", "X"], maxlags=lags)
    expected = _newey_west_se_by_hand(autocorrelated_data, ["Y", "X"], lags)
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


def test_hac_widens_the_standard_error_under_positive_autocorrelation(
    config, autocorrelated_data
):
    # With a persistent regressor and persistent errors, classical OLS
    # understates the slope's uncertainty several-fold.
    hac, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="HAC")
    plain, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="nonrobust")
    assert hac.loc["X", "std err"] > 1.5 * plain.loc["X", "std err"]


@pytest.mark.parametrize("cov_type", ["HAC", "nonrobust"])
def test_inference_columns_follow_the_t_distribution(
    config, regression_data, cov_type
):
    summary, _ = regress(config, regression_data, ["Y", "X1", "X2"], cov_type=cov_type)
    dof = len(regression_data) - 3
    half_width = stats.t.ppf(0.975, dof) * summary["std err"]

    np.testing.assert_allclose(
        summary["t"].values, (summary["coef"] / summary["std err"]).values, rtol=1e-12
    )
    np.testing.assert_allclose(
        summary["P>|t|"].values,
        2 * stats.t.sf(np.abs(summary["t"].values), dof),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        summary["0.025"].values, (summary["coef"] - half_width).values, rtol=1e-10
    )
    np.testing.assert_allclose(
        summary["0.975"].values, (summary["coef"] + half_width).values, rtol=1e-10
    )


# ---------------------------------------------------------------------------
# Covariance and lag defaults
# ---------------------------------------------------------------------------


def test_default_cov_type_comes_from_config(config, regression_data):
    assert config.regression_cov_type == "HAC"
    summary, _ = regress(config, regression_data, ["Y", "X1", "X2"])
    assert summary.attrs["cov_type"] == "HAC"


def test_config_cov_type_is_respected(config, regression_data):
    plain_config = dataclasses.replace(config, regression_cov_type="nonrobust")
    summary, _ = regress(plain_config, regression_data, ["Y", "X1", "X2"])
    expected = _ols_by_hand(regression_data, ["Y", "X1", "X2"])["se"]
    assert summary.attrs["cov_type"] == "nonrobust"
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


@pytest.mark.parametrize("n_obs, expected_lags", [(100, 4), (252, 4), (1000, 6)])
def test_default_maxlags_follows_the_newey_west_rule(
    config, rng, n_obs, expected_lags
):
    # floor(4 * (T / 100) ** (2 / 9))
    data = pd.DataFrame({"Y": rng.normal(size=n_obs), "X": rng.normal(size=n_obs)})
    summary, _ = regress(config, data, ["Y", "X"])
    assert summary.attrs["maxlags"] == expected_lags
    assert isinstance(summary.attrs["maxlags"], int)


def test_maxlags_rule_counts_only_complete_rows(config, rng):
    # 300 rows with 48 NaN leaves T = 252 usable observations -> 4 lags, where
    # counting all 300 rows would give 5.
    data = pd.DataFrame({"Y": rng.normal(size=300), "X": rng.normal(size=300)})
    data.iloc[:48, 0] = np.nan
    summary, _ = regress(config, data, ["Y", "X"])
    assert summary.attrs["maxlags"] == 4


def test_config_maxlags_overrides_the_rule(config, autocorrelated_data):
    lagged_config = dataclasses.replace(config, hac_maxlags=2)
    summary, _ = regress(lagged_config, autocorrelated_data, ["Y", "X"])
    expected = _newey_west_se_by_hand(autocorrelated_data, ["Y", "X"], 2)
    assert summary.attrs["maxlags"] == 2
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


def test_explicit_maxlags_overrides_config(config, autocorrelated_data):
    lagged_config = dataclasses.replace(config, hac_maxlags=2)
    summary, _ = regress(lagged_config, autocorrelated_data, ["Y", "X"], maxlags=7)
    assert summary.attrs["maxlags"] == 7


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1"])
def test_non_hac_fits_record_no_lag_length(config, regression_data, cov_type):
    summary, _ = regress(config, regression_data, ["Y", "X1", "X2"], cov_type=cov_type)
    assert summary.attrs["cov_type"] == cov_type
    assert summary.attrs["maxlags"] is None


def test_cov_type_is_case_insensitive_for_hac(config, autocorrelated_data):
    upper, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="HAC")
    lower, _ = regress(config, autocorrelated_data, ["Y", "X"], cov_type="hac")
    pd.testing.assert_series_equal(upper["std err"], lower["std err"])


# ---------------------------------------------------------------------------
# capm_regression()
# ---------------------------------------------------------------------------


def test_capm_regression_uses_only_the_market_factor(config, factor_data):
    # factor_data also carries SMB/HML/RMW/CMA; CAPM must ignore them.
    summary, _ = capm_regression(config, factor_data)
    assert list(summary.index) == ["Intercept", "Mkt"]
    expected = _ols_by_hand(factor_data, ["target", "Mkt"])
    np.testing.assert_allclose(summary["coef"].values, expected["coef"], rtol=1e-12)


def test_capm_regression_data_df_actual_matches_target_column(config, factor_data):
    _, data_df = capm_regression(config, factor_data)
    np.testing.assert_allclose(data_df["actual"].values, factor_data["target"].values)


def test_capm_regression_recovers_approximate_beta(config, factor_data):
    summary, _ = capm_regression(config, factor_data)
    assert summary.loc["Mkt", "coef"] == pytest.approx(1.1, abs=0.3)


def test_capm_regression_forwards_cov_type_and_maxlags(config, factor_data):
    summary, _ = capm_regression(config, factor_data, cov_type="HAC", maxlags=3)
    expected = _newey_west_se_by_hand(factor_data, ["target", "Mkt"], 3)
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# fama_french_three()
# ---------------------------------------------------------------------------


def test_fama_french_three_solves_the_normal_equations(config, factor_data):
    summary, data_df = fama_french_three(config, factor_data)
    expected = _ols_by_hand(factor_data, ["target", "Mkt", "SMB", "HML"])
    np.testing.assert_allclose(summary["coef"].values, expected["coef"], rtol=1e-10)
    np.testing.assert_allclose(data_df["resid"].values, expected["resid"], atol=1e-12)


def test_fama_french_three_includes_expected_factors(config, factor_data):
    summary, _ = fama_french_three(config, factor_data)
    assert list(summary.index) == ["Intercept", "Mkt", "SMB", "HML"]


def test_fama_french_three_forwards_cov_type_and_maxlags(config, factor_data):
    summary, _ = fama_french_three(config, factor_data, cov_type="HAC", maxlags=3)
    expected = _newey_west_se_by_hand(factor_data, ["target", "Mkt", "SMB", "HML"], 3)
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# fama_french_five()
# ---------------------------------------------------------------------------


def test_fama_french_five_solves_the_normal_equations(config, factor_data):
    cols = ["target", "Mkt", "SMB", "HML", "RMW", "CMA"]
    summary, data_df = fama_french_five(config, factor_data)
    expected = _ols_by_hand(factor_data, cols)
    np.testing.assert_allclose(summary["coef"].values, expected["coef"], rtol=1e-10)
    np.testing.assert_allclose(data_df["resid"].values, expected["resid"], atol=1e-12)


def test_fama_french_five_includes_expected_factors(config, factor_data):
    summary, _ = fama_french_five(config, factor_data)
    assert list(summary.index) == ["Intercept", "Mkt", "SMB", "HML", "RMW", "CMA"]


def test_fama_french_five_forwards_cov_type_and_maxlags(config, factor_data):
    cols = ["target", "Mkt", "SMB", "HML", "RMW", "CMA"]
    summary, _ = fama_french_five(config, factor_data, cov_type="nonrobust")
    expected = _ols_by_hand(factor_data, cols)["se"]
    np.testing.assert_allclose(summary["std err"].values, expected, rtol=1e-10)
