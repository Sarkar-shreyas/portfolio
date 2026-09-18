from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.backend.config import DevConfig


def plot_timeseries(data: pd.DataFrame, ax, params_dict: dict):
    """Plot input data on the specified axes and return the artists"""
    output = ax.plot(data, **params_dict)
    return output
