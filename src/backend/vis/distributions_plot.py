import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def plot_qq(data: pd.DataFrame, ax, params_dict: dict):
    """Returns a qq plot for the given data"""
    output = sm.qqplot(data, ax=ax, **params_dict)
    return output


def plot_hist(data: pd.DataFrame, ax, params_dict: dict):
    """Returns a histogram plot for the given data"""
    output = ax.hist(data, **params_dict)
    return output
