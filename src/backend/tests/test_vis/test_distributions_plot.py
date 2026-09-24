"""Tests for ``src.backend.vis.distributions_plot``."""

import numpy as np
import pandas as pd
import pytest

from src.backend.vis.distributions_plot import plot_hist, plot_qq


@pytest.fixture
def returns(config) -> pd.Series:
    rng = np.random.default_rng(config.random_seed)
    return pd.Series(rng.normal(0.0, 0.01, size=200))


# ---------------------------------------------------------------------------
# plot_hist()
# ---------------------------------------------------------------------------


def test_plot_hist_counts_every_observation_once(returns, ax):
    counts, _, _ = plot_hist(returns, ax, {})
    assert counts.sum() == len(returns)


def test_plot_hist_uses_the_requested_bins(returns, ax):
    counts, edges, _ = plot_hist(returns, ax, {"bins": 7})
    assert len(counts) == 7
    assert len(edges) == 8
    assert edges[0] == returns.min()
    assert edges[-1] == returns.max()


def test_plot_hist_matches_a_known_answer(ax):
    data = pd.Series([0.0, 0.1, 0.1, 0.2, 0.2, 0.2])
    counts, _, _ = plot_hist(data, ax, {"bins": [-0.05, 0.05, 0.15, 0.25]})
    np.testing.assert_array_equal(counts, [1, 2, 3])


def test_plot_hist_draws_on_the_given_axes(returns, ax):
    _, _, patches = plot_hist(returns, ax, {"bins": 5})
    assert len(ax.patches) == 5
    assert all(patch.axes is ax for patch in patches)


# ---------------------------------------------------------------------------
# plot_qq()
# ---------------------------------------------------------------------------


def test_plot_qq_draws_on_the_given_axes(returns, ax):
    fig = plot_qq(returns, ax, {})
    assert fig is ax.figure
    assert ax.has_data()


def test_plot_qq_plots_the_sorted_sample_against_normal_quantiles(returns, ax):
    plot_qq(returns, ax, {})
    points = ax.get_lines()[0]
    np.testing.assert_allclose(points.get_ydata(), np.sort(returns.values))
    theoretical = points.get_xdata()
    assert len(theoretical) == len(returns)
    assert (np.diff(theoretical) > 0).all()
    # Theoretical normal quantiles are symmetric about zero.
    np.testing.assert_allclose(theoretical, -theoretical[::-1], atol=1e-12)


def test_plot_qq_forwards_params_dict(returns, ax):
    plot_qq(returns, ax, {"line": "45"})
    assert len(ax.get_lines()) == 2
