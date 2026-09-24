"""Tests for ``src.backend.vis.returns_plot``."""

import numpy as np
import pandas as pd

from src.backend.vis.returns_plot import plot_timeseries


def test_plot_timeseries_draws_one_line_per_column(close_panel, ax):
    lines = plot_timeseries(close_panel, ax, {})
    assert len(lines) == close_panel.shape[1]
    for line, ticker in zip(lines, close_panel.columns):
        np.testing.assert_array_equal(line.get_ydata(), close_panel[ticker].values)


def test_plot_timeseries_keeps_the_dates(close_panel, ax):
    (line,) = plot_timeseries(close_panel["AAA"], ax, {})
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(line.get_xdata()), close_panel.index, check_names=False
    )


def test_plot_timeseries_forwards_params_dict(close_panel, ax):
    (line,) = plot_timeseries(close_panel["AAA"], ax, {"color": "red", "linewidth": 3})
    assert line.get_color() == "red"
    assert line.get_linewidth() == 3


def test_plot_timeseries_draws_on_the_given_axes(close_panel, ax):
    lines = plot_timeseries(close_panel, ax, {})
    assert all(line.axes is ax for line in lines)
