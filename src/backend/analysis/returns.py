import pandas as pd
import numpy as np
from typing import Optional
from src.backend.config import DevConfig

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "ann_returns",
    "rolling_returns",
    "sma_returns",
    "ema_returns",
    "simple_rsi",
    "ewm_rsi",
]


def simple_returns(data: pd.Series) -> pd.Series:
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


def simple_rsi(
    config: DevConfig, data: pd.Series, window: Optional[int] = None
) -> pd.Series:
    """Computes the simple RSI for a given returns series."""
    if window is None:
        window = config.sma_window
    d = data.copy()
    gains = (
        d.apply(lambda x: x if x > 0 else 0).shift(1).rolling(window).mean().dropna()
    )
    losses = (
        d.apply(lambda x: -x if x < 0 else 0).shift(1).rolling(window).mean().dropna()
    )
    rel_strength = gains / losses
    rsi = 100 - (100 / (1 + rel_strength))
    return rsi


def ewm_rsi(
    config: DevConfig, data: pd.Series, span: Optional[int] = None
) -> pd.Series:
    """Computes the exponential window RSI for a given returns series"""
    if span is None:
        span = config.ema_window

    d = data.copy()
    gains = (
        d.apply(lambda x: x if x > 0 else 0)
        .shift(1)
        .ewm(span=span, adjust=False)
        .mean()
        .dropna()
    )
    losses = (
        d.apply(lambda x: -x if x < 0 else 0)
        .shift(1)
        .ewm(span=span, adjust=False)
        .mean()
        .dropna()
    )
    rel_strength = gains / losses
    rsi = 100 - (100 / (1 + rel_strength))
    return rsi
