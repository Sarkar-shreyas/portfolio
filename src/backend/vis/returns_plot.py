import pandas as pd

__all__ = ["plot_timeseries"]


def plot_timeseries(data: pd.DataFrame, ax, params_dict: dict):
    """Plot input data on the specified axes and return the artists"""
    output = ax.plot(data, **params_dict)
    return output
