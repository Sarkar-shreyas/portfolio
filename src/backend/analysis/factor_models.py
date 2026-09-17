import pandas as pd
import numpy as np
from typing import Optional
from statsmodels.formula.api import ols

from src.backend.config import DevConfig


def get_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the correlation matrix for a given dataframe of returns"""
    return data.corr()


def rolling_correlation(
    config: DevConfig, data: pd.DataFrame, cols: list, window: Optional[int] = None
) -> pd.DataFrame:
    """Computes the rolling correlations matrix for the specified columns of a given dataframe"""
    if window is None:
        window = config.sma_window
    if len(cols) < 2:
        raise ValueError("Cannot compute correlations for fewer than two columns.")
    if not cols:
        print(
            "No columns were specified. Computing correlations for first two columns."
        )
        cols = list(data.columns)
    rolling_corr = data[cols[0]].rolling(window).corr(data[cols[1]])
    return rolling_corr


def get_covariance(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the covariance matrix for a given dataframe of returns"""
    return data.cov()


def rolling_covariance(
    config: DevConfig, data: pd.DataFrame, cols: list, window: Optional[int] = None
) -> pd.DataFrame:
    """Computes the rolling covariance matrix for the specified columns of a given dataframe"""
    if window is None:
        window = config.sma_window
    if len(cols) < 2:
        raise ValueError("Cannot compute covariance for fewer than two columns.")
    if not cols:
        print("No columns were specified. Computing covariances for first two columns.")
        cols = list(data.columns)
    rolling_cov = data[cols[0]].rolling(window).cov(data[cols[1]])
    return rolling_cov


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


def fama_french_three(config: DevConfig, data: pd.DataFrame) -> pd.DataFrame:
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
    return summary_df


def fama_french_five(config: DevConfig, data: pd.DataFrame) -> pd.DataFrame:
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
    return summary_df
