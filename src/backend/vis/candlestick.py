import mplfinance as mpf
import pandas as pd

__all__ = ["plot_candlesticks"]


def plot_candlesticks(data: pd.DataFrame, ax, params_dict: dict):
    """Plot a candlestick chart for the given data"""
    output = mpf.plot(data, ax=ax, type="candle", **params_dict)
    return output
