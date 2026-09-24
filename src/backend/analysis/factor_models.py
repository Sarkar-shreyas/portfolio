import pandas as pd
import numpy as np
from typing import Optional
from statsmodels.formula.api import ols

from src.backend.config import DevConfig

__all__ = ["regress", "capm_regression", "fama_french_three", "fama_french_five"]


def _summary(model) -> pd.DataFrame:
    """
    Takes in a fitted statsmodels regression model and outputs stats of interest into
    a dataframe
    """
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


def _factor_regression(
    config: DevConfig,
    data: pd.DataFrame,
    factors: list,
    cov_type: Optional[str] = None,
    maxlags: Optional[int] = None,
) -> tuple:
    """
    Performs an OLS regression on the input data against the specified factors. Uses
    statsmodels OLS model with the specified covariance type and maxlags parameter,
    defaulting to Newey-West HAC standard errors. Target var is assumed to be the first
    column of the factors list.
    """
    if cov_type is None:
        cov_type = config.regression_cov_type
    if len(factors) <= 1:
        raise ValueError(
            f"Insufficient number of factors entered : {len(factors)}. Cannot perform regression."
        )
    t = len(data[factors].dropna())
    if maxlags is None:
        # Check if config has a value
        maxlags = config.hac_maxlags
    if maxlags is None:
        # Compute manually
        maxlags = int(np.floor(4 * (t / 100) ** (2 / 9)))
    if cov_type.upper() == "HAC":
        cov_kwds = {"maxlags": maxlags, "use_correction": True}
    else:
        cov_kwds = None
    target = factors[0]
    formula_str = f"{factors[0]} ~ {' + '.join(factors[1:])}"
    model = ols(formula=formula_str, data=data[factors]).fit(
        cov_type=cov_type,  # type: ignore
        cov_kwds=cov_kwds,
        use_t=True,
    )
    data_df = pd.DataFrame(
        {
            "actual": data[target],
            "resid": model.resid,
            "fittedvalues": model.fittedvalues,
        }
    )
    summary_df = _summary(model)
    summary_df.attrs["cov_type"] = cov_type
    summary_df.attrs["maxlags"] = maxlags if cov_type.upper() == "HAC" else None
    return summary_df, data_df


def regress(
    config: DevConfig,
    data: pd.DataFrame,
    cols: Optional[list[str]] = None,
    cov_type: Optional[str] = None,
    maxlags: Optional[int] = None,
) -> tuple:
    """
    Performs an OLS regression on the given data for the specified columns. Assumes cols[0]
    is the target variable.
    """
    if cols is None:
        print("No columns were specified. Regressing all existing columns.")
        cols = list(data.columns)
    return _factor_regression(config, data, cols, cov_type, maxlags)


def capm_regression(
    config: DevConfig,
    data: pd.DataFrame,
    cov_type: Optional[str] = None,
    maxlags: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform an OLS regression on a given returns dataframe, following the CAPM model.
    Assumes the data represents excess returns.
    """
    factors = [data.columns[0], "Mkt"]
    return _factor_regression(config, data, factors, cov_type, maxlags)


def fama_french_three(
    config: DevConfig,
    data: pd.DataFrame,
    cov_type: Optional[str] = None,
    maxlags: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform an OLS regression on a given returns dataframe, following the Fama-French
    3-factor model.
    """
    factors = [data.columns[0], "Mkt", "SMB", "HML"]
    return _factor_regression(config, data, factors, cov_type, maxlags)


def fama_french_five(
    config: DevConfig,
    data: pd.DataFrame,
    cov_type: Optional[str] = None,
    maxlags: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform an OLS regression on a given returns dataframe, following the Fama-French
    5-factor model.
    """
    factors = [data.columns[0], "Mkt", "SMB", "HML", "RMW", "CMA"]
    return _factor_regression(config, data, factors, cov_type, maxlags)
