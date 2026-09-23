import pandas as pd
import numpy as np
from typing import Optional
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


def sma_crossover_portfolio(
    config: DevConfig,
    prices: pd.Series | pd.DataFrame,
    short_window: Optional[int] = None,
    long_window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a signal dataframe based off SMA crossover for a portfolio of close prices.
    Computes 'sma_short' and 'sma_long' columns on the given data and pass price series
    into sma_crossover for each individual ticker.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    prices: pd.Series | pd.DataFrame
        A series or dataframe containing close prices for the ticker universe. A series
        is promoted to a single column dataframe
    short_window: int, Optional
        Optional short SMA period to overwrite default config value
    long_window: int, Optional
        Optional long SMA period to overwrite default config value

    Returns
    -------
    pd.DataFrame
        A dataframe with the same index and columns of prices, populated with signals in
        {-1, 0, 1}
    """
    if short_window is None:
        short_window = config.sma_short_window
    if long_window is None:
        long_window = config.sma_window
    if short_window >= long_window:
        raise ValueError(
            f"short_window must be shorter than long_window: {short_window} >= {long_window}"
        )
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    short_sma = prices.rolling(short_window).mean()
    long_sma = prices.rolling(long_window).mean()

    signals = {
        ticker: sma_crossover(
            config,
            pd.DataFrame(
                {"sma_short": short_sma[ticker], "sma_long": long_sma[ticker]}
            ),
        )
        for ticker in prices.columns
    }
    return pd.DataFrame(signals, index=prices.index, columns=prices.columns)


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


def ema_crossover_portfolio(
    config: DevConfig,
    prices: pd.Series | pd.DataFrame,
    short_window: Optional[int] = None,
    long_window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a signal dataframe based off EMA crossover for a portfolio of close prices.
    Computes 'ema_short' and 'ema_long' columns on the given data and pass price series
    into ema_crossover for each individual ticker.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    prices: pd.Series | pd.DataFrame
        A series or dataframe containing close prices for the ticker universe. A series
        is promoted to a single column dataframe
    short_window: int, Optional
        Optional short SMA period to overwrite default config value
    long_window: int, Optional
        Optional long SMA period to overwrite default config value

    Returns
    -------
    pd.DataFrame
        A dataframe with the same index and columns of prices, populated with signals in
        {-1, 0, 1}
    """
    if short_window is None:
        short_window = config.ema_short_window
    if long_window is None:
        long_window = config.ema_window
    if short_window >= long_window:
        raise ValueError(
            f"short_window must be shorter than long_window: {short_window} >= {long_window}"
        )
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()

    signals = {
        ticker: ema_crossover(
            config,
            pd.DataFrame(
                {"ema_short": short_ema[ticker], "ema_long": long_ema[ticker]}
            ),
        )
        for ticker in prices.columns
    }
    return pd.DataFrame(signals, index=prices.index, columns=prices.columns)
