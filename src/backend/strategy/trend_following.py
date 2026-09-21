import pandas as pd
import numpy as np

from src.backend.config import DevConfig


def sma_crossover(config: DevConfig, data: pd.DataFrame) -> pd.Series:
    """Generates a signal series based off SMA crossover. Assumes data columns are ['sma_short', 'sma_long']"""
    # take_long = data[data["sma_short"] > data["sma_long"]].index
    # take_short = data[data["sma_short"] < data["sma_long"]].index

    signal = np.where(
        data["sma_short"] > data["sma_long"],
        1,
        np.where(data["sma_short"] < data["sma_long"], -1, 0),
    )
    signal_series = pd.Series(signal, index=data.index, name="signal")
    return signal_series


def ema_crossover(config: DevConfig, data: pd.DataFrame) -> pd.Series:
    """Generates a signal series based off EMA crossover. Assumes the data columns are ['ema_short', 'ema_long']"""
    # take_long = data[data["ema_short"] > data["ema_long"]].index
    # take_short = data[data["ema_short"] < data["ema_long"]].index

    signal = np.where(
        data["ema_short"] > data["ema_long"],
        1,
        np.where(data["ema_short"] < data["ema_long"], -1, 0),
    )
    signal_series = pd.Series(signal, index=data.index, name="signal")
    return signal_series
