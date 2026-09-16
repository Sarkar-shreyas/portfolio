import pandas as pd
import numpy as np
from typing import Optional
from src.backend.config import DevConfig


def returns(data: pd.Series) -> pd.Series:
    """Computes simple returns given the close price"""
    returns = data.pct_change()
    return returns


def log_returns(data: pd.Series) -> np.ndarray:
    """Computes log returns given the close price"""
    log_returns = np.log(data) - np.log(data.shift(1))
    return log_returns


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Computes cumulative returns"""
    return (1 + returns).cumprod() - 1


def ann_returns(config: DevConfig, data: pd.Series, ann: Optional[int] = None) -> float:
    """Compute the annualised returns assuming a given daily return series"""
    if ann is None:
        ann = config.annualise
    n_days = len(data)
    total_returns = cumulative_returns(data).iloc[-1]
    ann_returns = ((1 + total_returns) ** (ann / n_days)) - 1
    return ann_returns


def rolling_returns(
    config: DevConfig,
    data: pd.Series,
    window: Optional[int] = None,
    metric: str = "mean",
) -> pd.Series:
    """Compute the rolling statistic for the returns"""
    if window is None:
        window = config.sma_window
    return data.rolling(window).agg(metric).dropna()


def sma_returns(
    config: DevConfig, data: pd.Series, period: Optional[int] = None
) -> pd.Series:
    """Compute the SMA for the input data, defaulting to a 21 day period."""
    if period is None:
        period = config.sma_window
    return data.rolling(period).mean().dropna()


def ema_returns(
    config: DevConfig, data: pd.Series, span: Optional[int] = None
) -> pd.Series:
    """Compute the EMA for the input data, defaulting to a 21 day period."""
    if span is None:
        span = config.ema_window
    return data.ewm(span=span, adjust=False).mean().dropna()
