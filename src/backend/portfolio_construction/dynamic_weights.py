import pandas as pd
import numpy as np
from typing import Optional
from src.backend.config import DevConfig


def equal_active_weights(
    config: DevConfig, signals: pd.DataFrame, tot_exposure: Optional[float] = None
) -> pd.DataFrame:
    """Computes portfolio weights by equally-weighing active legs (signal != 0)"""
    if tot_exposure is None:
        tot_exposure = config.tot_exposure
    active = signals.ne(0)
    n_active = active.sum(axis=1).replace(0, np.nan)

    weights = signals.div(n_active, axis=0).mul(tot_exposure).fillna(0.0)
    return weights


def equal_split_ls_weights(
    config: DevConfig,
    signals: pd.DataFrame,
    long_exposure: Optional[float] = None,
    short_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """Computes portfolio weights by equally-weighing long and short legs for given exposure caps."""
    if long_exposure is None:
        long_exposure = config.long_exposure
    if short_exposure is None:
        short_exposure = config.short_exposure

    long_mask = signals.eq(1)
    short_mask = signals.eq(-1)

    n_long = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)

    n_long.replace(0.0, np.nan, inplace=True)
    n_short.replace(0.0, np.nan, inplace=True)

    long_weights = long_mask.div(n_long, axis=0).mul(long_exposure)
    short_weights = short_mask.div(n_short, axis=0).mul(short_exposure)

    weights = long_weights + short_weights
    weights.fillna(0.0, inplace=True)
    return weights


def inverse_volatility_weighted(
    config: DevConfig,
    signals: pd.DataFrame,
    volatilities: pd.DataFrame,
    tot_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """Compute portfolio weights by the inverse of ticker volatility"""
    if tot_exposure is None:
        tot_exposure = config.tot_exposure
    vol = signals.div(volatilities)
    vol_exposure = vol.abs().sum(axis=1).replace(0.0, np.nan)
    normed = vol.div(vol_exposure, axis=0).mul(tot_exposure).fillna(0.0)

    return normed


def inverse_volatility_split_ls_weights(
    config: DevConfig,
    signals: pd.DataFrame,
    volatilities: pd.DataFrame,
    long_exposure: Optional[float] = None,
    short_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """Compute the portfolio weights by the inverse of ticker volatility, for a given long and short exposure cap."""
    if long_exposure is None:
        long_exposure = config.long_exposure
    if short_exposure is None:
        short_exposure = config.short_exposure

    long_mask = signals.eq(1)
    short_mask = signals.eq(-1)

    inv_vol = 1 / volatilities
    long_vols = inv_vol.where(long_mask, 0.0)
    short_vols = inv_vol.where(short_mask, 0.0)

    n_long = long_vols.sum(axis=1)
    n_short = short_vols.sum(axis=1)

    n_long.replace(0.0, np.nan, inplace=True)
    n_short.replace(0.0, np.nan, inplace=True)

    long_weights = long_vols.div(n_long, axis=0).mul(long_exposure).fillna(0.0)
    short_weights = short_vols.div(n_short, axis=0).mul(short_exposure).fillna(0.0)

    weights = long_weights + short_weights

    return weights
