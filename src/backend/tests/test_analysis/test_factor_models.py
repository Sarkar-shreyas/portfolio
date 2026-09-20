import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols

from src.backend.analysis.factor_models import (
    regress,
    capm_regression,
    fama_french_three,
    fama_french_five,
)


def _summary_df(model):
    return pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "0.025": model.conf_int()[0],
            "0.975": model.conf_int()[1],
        }
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


# ---------------------------------------------------------------------------
# regress()
# ---------------------------------------------------------------------------


def test_regress_matches_manual_ols(config, regression_data):
    result = regress(config, regression_data, cols=["Y", "X1", "X2"])
    expected_model = ols(formula="Y ~ X1 + X2", data=regression_data).fit()
    expected = _summary_df(expected_model)
    pd.testing.assert_frame_equal(result, expected)


def test_regress_single_predictor(config, regression_data):
    result = regress(config, regression_data, cols=["Y", "X1"])
    expected_model = ols(formula="Y ~ X1", data=regression_data[["Y", "X1"]]).fit()
    expected = _summary_df(expected_model)
    pd.testing.assert_frame_equal(result, expected)


def test_regress_defaults_to_all_columns_when_cols_is_none(config, regression_data, capsys):
    result = regress(config, regression_data, cols=None)
    expected_model = ols(formula="Y ~ X1 + X2", data=regression_data).fit()
    expected = _summary_df(expected_model)
    pd.testing.assert_frame_equal(result, expected)


def test_regress_recovers_approximate_true_coefficients(config, regression_data):
    result = regress(config, regression_data, cols=["Y", "X1", "X2"])
    assert result.loc["X1", "coef"] == pytest.approx(2.0, abs=0.2)
    assert result.loc["X2", "coef"] == pytest.approx(-0.5, abs=0.2)


# ---------------------------------------------------------------------------
# capm_regression()
# ---------------------------------------------------------------------------


def test_capm_regression_matches_manual_ols(config, factor_data):
    summary_df, data_df = capm_regression(config, factor_data)

    expected_model = ols(formula="target ~ Mkt", data=factor_data).fit()
    expected_summary = _summary_df(expected_model)

    pd.testing.assert_frame_equal(summary_df, expected_summary)
    np.testing.assert_allclose(data_df["resid"].values, expected_model.resid.values)
    np.testing.assert_allclose(
        data_df["fittedvalues"].values, expected_model.fittedvalues.values
    )


def test_capm_regression_data_df_actual_matches_target_column(config, factor_data):
    _, data_df = capm_regression(config, factor_data)
    np.testing.assert_allclose(data_df["actual"].values, factor_data["target"].values)


def test_capm_regression_recovers_approximate_beta(config, factor_data):
    summary_df, _ = capm_regression(config, factor_data)
    assert summary_df.loc["Mkt", "coef"] == pytest.approx(1.1, abs=0.3)


# ---------------------------------------------------------------------------
# fama_french_three()
# ---------------------------------------------------------------------------


def test_fama_french_three_matches_manual_ols(config, factor_data):
    summary_df, data_df = fama_french_three(config, factor_data)

    expected_model = ols(formula="target ~ Mkt + SMB + HML", data=factor_data).fit()
    expected_summary = _summary_df(expected_model)

    pd.testing.assert_frame_equal(summary_df, expected_summary)
    np.testing.assert_allclose(data_df["resid"].values, expected_model.resid.values)


def test_fama_french_three_includes_expected_factors(config, factor_data):
    summary_df, _ = fama_french_three(config, factor_data)
    assert set(summary_df.index) == {"Intercept", "Mkt", "SMB", "HML"}


# ---------------------------------------------------------------------------
# fama_french_five()
# ---------------------------------------------------------------------------


def test_fama_french_five_matches_manual_ols(config, factor_data):
    summary_df, data_df = fama_french_five(config, factor_data)

    expected_model = ols(
        formula="target ~ Mkt + SMB + HML + RMW + CMA", data=factor_data
    ).fit()
    expected_summary = _summary_df(expected_model)

    pd.testing.assert_frame_equal(summary_df, expected_summary)
    np.testing.assert_allclose(data_df["resid"].values, expected_model.resid.values)


def test_fama_french_five_includes_expected_factors(config, factor_data):
    summary_df, _ = fama_french_five(config, factor_data)
    assert set(summary_df.index) == {"Intercept", "Mkt", "SMB", "HML", "RMW", "CMA"}
