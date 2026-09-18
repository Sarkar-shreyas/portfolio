import matplotlib as plt
import pandas as pd


def heatmap(data: pd.DataFrame, ax, params_dict: dict):
    """Returns a heatmap for the given data"""
    output = ax.imshow(data, **params_dict)
    return output
