import pandas as pd

__all__ = ["heatmap"]


def heatmap(data: pd.DataFrame, ax, params_dict: dict):
    """Returns a heatmap for the given data"""
    output = ax.imshow(data, **params_dict)
    return output
