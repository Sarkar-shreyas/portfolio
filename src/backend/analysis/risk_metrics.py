import pandas as pd

# import numpy as np
from typing import Optional
from src.backend.config import DevConfig
from src.backend.analysis.returns import (
    ann_returns,
    cumulative_returns,
    rolling_returns,
)
from src.backend.analysis.vol import ann_volatility, rolling_volatility


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
