import pandas as pd
import numpy as np
from typing import Optional
from statsmodels.formula.api import ols

from src.backend.config import DevConfig


def regress(
    config: DevConfig, data: pd.DataFrame, cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Performs an OLS regression on the given data for the specified columns. Assumes cols[0] is the target variable."""
    if cols is None:
        print("No columns were specified. Regressing all existing columns.")
        cols = list(data.columns)
    target = cols[0]
    formula_str = f"{target} ~ {cols[1]}"
    if len(cols) > 2:
        for col in cols[2:]:
            formula_str = formula_str + " + " + f"{col}"
    model = ols(formula=formula_str, data=data[cols]).fit()
    summary_df = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "0.025": model.conf_int()[0],
            "0.975": model.conf_int()[1],
        }
    )
    return summary_df


def capm_regression(
    config: DevConfig, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an OLS regression on a given returns dataframe, following the CAPM model. Assumes the data represents excess returns"""
    target = data.columns[0]
    formula_str = f"{target} ~ Mkt-RF"
    model = ols(formula=formula_str, data=data).fit()
    summary_df = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "0.025": model.conf_int()[0],
            "0.975": model.conf_int()[1],
        }
    )
    data_df = pd.DataFrame(
        {
            "actual": data[[target]],
            "resid": model.resid,
            "fittedvalues": model.fittedvalues,
        }
    )
    return summary_df, data_df


def fama_french_three(
    config: DevConfig, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an OLS regression on a given returns dataframe, following the Fama-French 3-factor model."""
    factors = ["Mkt-RF", "SMB", "HML"]
    target = data.columns[0]
    formula_str = f"{target} ~ {' + '.join(factors)}"
    model = ols(formula=formula_str, data=data).fit()
    summary_df = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "0.025": model.conf_int()[0],
            "0.975": model.conf_int()[1],
        }
    )
    data_df = pd.DataFrame(
        {
            "actual": data[[target]],
            "resid": model.resid,
            "fittedvalues": model.fittedvalues,
        }
    )
    return summary_df, data_df


def fama_french_five(
    config: DevConfig, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an OLS regression on a given returns dataframe, following the Fama-French 5-factor model."""
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    target = data.columns[0]
    formula_str = f"{target} ~ {' + '.join(factors)}"
    model = ols(formula=formula_str, data=data).fit()
    summary_df = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "0.025": model.conf_int()[0],
            "0.975": model.conf_int()[1],
        }
    )
    data_df = pd.DataFrame(
        {
            "actual": data[[target]],
            "resid": model.resid,
            "fittedvalues": model.fittedvalues,
        }
    )
    return summary_df, data_df
