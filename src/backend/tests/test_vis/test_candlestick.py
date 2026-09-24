"""Tests for ``src.backend.vis.candlestick``.

mplfinance draws each bar as a body (a polygon from open to close) and two wicks
(line segments out to the low and the high). The fixture has hand-chosen values so
every expected extent is known in advance.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from src.backend.vis.candlestick import plot_candlesticks

N_BARS = 12
DOWN_BAR = 3


@pytest.fixture
def bars() -> pd.DataFrame:
    """Rising bars with one down day, in clean_timeseries' lowercase layout."""
    base = np.arange(N_BARS, dtype=float) + 10.0
    frame = pd.DataFrame(
        {
            "open": base,
            "close": base + 0.5,
            "high": base + 1.0,
            "low": base - 1.0,
            "volume": 1e6,
        },
        index=pd.bdate_range("2024-01-01", periods=N_BARS),
    )
    frame.loc[frame.index[DOWN_BAR], ["open", "close", "high"]] = [15.0, 12.0, 15.5]
    return frame


def _bodies(ax) -> PolyCollection:
    (bodies,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    return bodies


def _wicks(ax) -> LineCollection:
    (wicks,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    return wicks


def test_plot_candlesticks_draws_one_body_per_bar(bars, ax):
    plot_candlesticks(bars, ax, {})
    assert len(_bodies(ax).get_paths()) == N_BARS


def test_candle_bodies_span_open_to_close(bars, ax):
    plot_candlesticks(bars, ax, {})
    for path, (_, bar) in zip(_bodies(ax).get_paths(), bars.iterrows()):
        y = path.vertices[:, 1]
        assert y.min() == pytest.approx(min(bar["open"], bar["close"]))
        assert y.max() == pytest.approx(max(bar["open"], bar["close"]))


def test_candle_wicks_reach_the_high_and_low(bars, ax):
    plot_candlesticks(bars, ax, {})
    segments = _wicks(ax).get_segments()
    assert len(segments) == 2 * N_BARS
    all_y = np.concatenate([seg[:, 1] for seg in segments])
    assert all_y.min() == pytest.approx(bars["low"].min())
    assert all_y.max() == pytest.approx(bars["high"].max())


def test_down_bars_are_coloured_differently_from_up_bars(bars, ax):
    plot_candlesticks(bars, ax, {})
    colours = _bodies(ax).get_facecolors()
    up = colours[0]
    assert not np.allclose(colours[DOWN_BAR], up)
    others = np.delete(colours, DOWN_BAR, axis=0)
    assert np.allclose(others, up)


def test_plot_candlesticks_accepts_capitalised_columns(bars, ax):
    plot_candlesticks(bars.rename(columns=str.title), ax, {})
    assert len(_bodies(ax).get_paths()) == N_BARS


def test_plot_candlesticks_draws_on_the_given_axes(bars, ax):
    plot_candlesticks(bars, ax, {})
    assert ax.has_data()
    assert _bodies(ax).axes is ax
