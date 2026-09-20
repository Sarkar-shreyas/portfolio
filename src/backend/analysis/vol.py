from arch.univariate.base import ARCHModelResult
import pandas as pd
import numpy as np
from typing import Optional
from src.backend.config import DevConfig
from arch import arch_model


def ann_volatility(
    config: DevConfig, data: pd.Series, ann: Optional[int] = None
) -> float:
    """Computes the annualised volatility for a given returns series"""
    if ann is None:
        ann = config.annualise
    daily_vol = data.std()
    ann_vol = daily_vol * np.sqrt(ann)
    return ann_vol


def rolling_volatility(
    config: DevConfig,
    data: pd.Series,
    period: Optional[int] = None,
    metric: str = "mean",
) -> pd.Series:
    """Compute the simple rolling volatility statistic"""
    if period is None:
        period = config.sma_window
    return data.rolling(period).agg(metric).dropna()


def ewma_volatility(
    config: DevConfig, data: pd.Series, span: Optional[int] = None, metric: str = "mean"
) -> pd.Series:
    """Compute the EWMA rolling volatility statistic"""
    if span is None:
        span = config.ema_window
    return data.ewm(span=span, adjust=False).agg(metric).dropna()


def fit_garch(
    config: DevConfig, data: pd.Series, model_params: dict, fit_params: dict
) -> ARCHModelResult:
    """Perform a GARCH or GJR-GARCH volatility analysis on the given returns series."""
    am = arch_model(data, **model_params)
    res = am.fit(**fit_params)
    return res
