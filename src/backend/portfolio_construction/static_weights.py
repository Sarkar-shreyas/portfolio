import pandas as pd
import numpy as np
from typing import Optional

from src.backend.config import DevConfig

__all__ = [
    "equal_weight_benchmark",
    "mkt_cap_weight_benchmark",
    "turnover",
    "avg_gross_exposure",
    "avg_net_exposure",
]


def equal_weight_benchmark(
    config: DevConfig, tickers: list, tot_exposure: Optional[float] = None
) -> pd.Series:
    """Return portfolio weights for the static, equally-weighted benchmark."""
    if tot_exposure is None:
        tot_exposure = config.tot_exposure
    n_tickers = len(tickers)
    weights = np.repeat(tot_exposure / n_tickers, n_tickers)
    port_weights = pd.Series(weights, index=tickers, name="weights")
    return port_weights


def mkt_cap_weight_benchmark(
    config: DevConfig, mkt_caps: pd.Series, tot_exposure: Optional[float] = None
) -> pd.Series:
    """Return portfolio weights for the static, market-cap weighted benchmark."""
    if tot_exposure is None:
        tot_exposure = config.tot_exposure
    total_mkt_cap = mkt_caps.sum()
    mkt_cap_weights = mkt_caps.to_numpy() / total_mkt_cap
    port_weights = pd.Series(
        mkt_cap_weights * tot_exposure, index=mkt_caps.index, name="weights"
    )
    return port_weights


def turnover(config: DevConfig, weights: pd.DataFrame) -> pd.Series:
    """
    Computes the turnover for a given change in portfolio weights
    """
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    return changes.abs().sum(axis=1)


def avg_gross_exposure(config: DevConfig, weights: pd.DataFrame) -> float:
    """
    Computes the average gross exposure to each individual asset in a portfolio
    """
    return weights.abs().sum(axis=1).mean()


def avg_net_exposure(config: DevConfig, weights: pd.DataFrame) -> float:
    """
    Computes the average net exposure to each individual asset in a portfolio
    """
    return weights.sum(axis=1).mean()
