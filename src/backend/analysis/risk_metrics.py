import pandas as pd

import numpy as np
from typing import Optional
from src.backend.config import DevConfig
from src.backend.analysis.returns import (
    ann_returns,
    cumulative_returns,
    rolling_returns,
)
from src.backend.analysis.vol import ann_volatility, rolling_volatility


def get_metrics(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes several key statistics for each column of the input data.
    """
    metrics = data.agg(["max", "min", "mean", "median", "std", "skew", "kurtosis"])
    lower = np.quantile(data.dropna(), 0.25, axis=0)
    upper = np.quantile(data.dropna(), 0.75, axis=0)
    iqr = upper - lower
    metrics.loc["0.25"] = pd.Series(lower, index=data.columns)
    metrics.loc["0.75"] = pd.Series(upper, index=data.columns)
    metrics.loc["iqr"] = pd.Series(iqr, index=data.columns)
    metrics = metrics.T
    return metrics


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


def ann_sharpe(
    config: DevConfig,
    data: pd.Series,
) -> float:
    """Computes the Sharpe ratio for annualised data"""

    ann_ret = ann_returns(config, data)
    ann_vol = ann_volatility(config, data)

    sharpe = (ann_ret - config.risk_free_rate) / ann_vol
    return sharpe


def rolling_sharpe(
    config: DevConfig, data: pd.Series, window: Optional[int] = None
) -> pd.Series:
    """Computes the rolling Sharpe ratio over a given window"""
    if window is None:
        window = config.sharpe_window
    daily_rf = (1 + config.risk_free_rate) ** 1 / 252 - 1
    excess_ret = data - daily_rf
    rolling_ret = rolling_returns(config, excess_ret, window)
    rolling_vol = rolling_volatility(config, excess_ret, window)
    sharpe = rolling_ret / rolling_vol
    return sharpe


def ann_sortino(config: DevConfig, data: pd.Series) -> float:
    """Computes the Sortino ratio for annualised data"""

    loss = data.copy()
    loss = loss.apply(lambda x: x if x < 0 else 0)
    ann_ret = ann_returns(config, data)
    loss_vol = ann_volatility(config, loss)

    sortino = (ann_ret - config.risk_free_rate) / loss_vol
    return sortino


def rolling_sortino(
    config: DevConfig, data: pd.Series, window: Optional[int] = None
) -> pd.Series:
    """Computes the rolling Sortino ratio over a given window"""
    if window is None:
        window = config.sharpe_window
    daily_rf = (1 + config.risk_free_rate) ** 1 / 252 - 1
    excess_ret = data - daily_rf
    loss = data.copy()
    loss = loss.apply(lambda x: x if x < 0 else 0)
    rolling_ret = rolling_returns(config, excess_ret, window)
    loss_vol = rolling_volatility(config, loss, window)
    sortino = rolling_ret / loss_vol
    return sortino


def daily_drawdowns(config: DevConfig, data: pd.Series) -> pd.Series:
    """Computes the drawdown array for a given return series"""
    cum_returns = cumulative_returns(data)
    running_max = (cum_returns + 1).cummax()
    drawdowns = data / running_max - 1
    return drawdowns


def max_daily_drawdowns(config: DevConfig, data: pd.Series) -> pd.Series:
    """Computes the running max drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    max_daily_drawdowns = drawdowns.rolling(
        min_periods=1, window=config.annualise
    ).min()
    return max_daily_drawdowns


def max_drawdown(config: DevConfig, data: pd.Series) -> float:
    """Computes the maximum drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    max_dd = drawdowns.min()
    return max_dd


def min_drawdown(config: DevConfig, data: pd.Series) -> float:
    """Computes the maximum drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    min_dd = drawdowns.max()
    return min_dd


def rolling_drawdown(
    config: DevConfig,
    data: pd.Series,
    window: Optional[int] = None,
    metric: str = "mean",
) -> pd.Series:
    """Computes the rolling drawdown for a given return series"""
    if window is None:
        window = config.sma_window
    drawdowns = daily_drawdowns(config, data)
    rolling_draw = drawdowns.rolling(window).agg(metric).dropna()
    return rolling_draw


def ann_calmar(config: DevConfig, data: pd.Series) -> float:
    """Computes the Calmar ratio for a given return series"""
    ann_ret = ann_returns(config, data)
    max_draw = max_drawdown(config, data)

    calmar = (ann_ret - config.risk_free_rate) / max_draw
    return calmar
