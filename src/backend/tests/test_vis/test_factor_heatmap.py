"""Tests for ``src.backend.vis.factor_heatmap``."""

import numpy as np
from matplotlib.image import AxesImage

from src.backend.vis.factor_heatmap import heatmap


def test_heatmap_draws_the_matrix_values(close_panel, ax):
    corr = close_panel.pct_change().corr()
    image = heatmap(corr, ax, {})
    assert isinstance(image, AxesImage)
    np.testing.assert_array_equal(image.get_array(), corr.values)


def test_heatmap_forwards_params_dict(close_panel, ax):
    corr = close_panel.pct_change().corr()
    image = heatmap(corr, ax, {"cmap": "coolwarm", "vmin": -1, "vmax": 1})
    assert image.get_cmap().name == "coolwarm"
    assert image.get_clim() == (-1, 1)


def test_heatmap_draws_on_the_given_axes(close_panel, ax):
    image = heatmap(close_panel.pct_change().corr(), ax, {})
    assert image.axes is ax
    assert list(ax.images) == [image]
