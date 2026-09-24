import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import norm, t

from src.backend.config import DevConfig
from src.backend.analysis.returns import rolling_returns, simple_returns, ann_returns
from src.backend.analysis.risk_metrics import max_daily_drawdowns


def est_var(config: DevConfig, data: pd.Series, conf: Optional[float] = None) -> float:
    """Computes the estimated Value-at-Risk for the given returns series at the given confidence interval"""
    if conf is None:
        conf = config.var_conf

    return np.nanpercentile(data, (1 - conf) * 100)


def cond_var(config: DevConfig, data: pd.Series, conf: Optional[float] = None) -> float:
    """Computes the conditional Value-at-Risk for the given returns series at the given confidence interval"""
    if conf is None:
        conf = config.var_conf

    var = est_var(config, data, conf)
    lower = data[data < var]
    return lower.mean()


def norm_parametric_var(
    config: DevConfig,
    data: pd.Series,
    conf: Optional[float] = None,
    window: Optional[int] = None,
) -> pd.Series:
    """Estimates Value-at-Risk for a given returns series at the given confidence interval, over a rolling window, using the normal distribution"""
    if conf is None:
        conf = config.var_conf
    if window is None:
        window = config.sma_window
    rolling_data = data.rolling(window)
    mu = rolling_data.mean()
    sigma = rolling_data.std(ddof=1)
    var_estimate = -norm.ppf(conf, loc=mu, scale=sigma)
    var_estimate = pd.Series(var_estimate, index=data.index)
    return var_estimate


def t_parametric_var(
    config: DevConfig,
    data: pd.Series,
    conf: Optional[float] = None,
    window: Optional[int] = None,
    dof: int = 5,
) -> pd.Series:
    """Estimates Value-at-Risk for a given returns series at the given confidence interval, over a rolling window, using the Students-t distribution"""
    if conf is None:
        conf = config.var_conf
    if window is None:
        window = config.sma_window

    rescale_factor = np.sqrt(dof / (dof - 2))
    rolling_data = data.rolling(window)
    mu = rolling_data.mean()
    sigma = rolling_data.std(ddof=1)
    var_estimate = -t.ppf(1 - conf, df=dof, loc=mu, scale=sigma)
    var_estimate = pd.Series(var_estimate, index=data.index)
    return rescale_factor * var_estimate
