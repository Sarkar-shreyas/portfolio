"""Shared fixtures for the vis test suite.

The suite runs on the non-interactive Agg backend, so nothing opens a window. Tests
assert on what was drawn -- the data behind each artist, labels, counts -- rather than
comparing rendered images, which is fragile across matplotlib versions and fonts.

The backtest-plot fixtures run a real WalkForwardBacktester, so the helpers are
exercised on genuine run records rather than hand-built dicts.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.backend.config import TestConfig
from src.backend.backtest.costs import linear_cost
from src.backend.backtest.walk_forward import WalkForwardBacktester
from src.backend.portfolio_construction.dynamic_weights import equal_active_weights
from src.backend.strategy.trend_following import sma_crossover_portfolio

N_OBS = 300
TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def config() -> TestConfig:
    return TestConfig()


@pytest.fixture
def ax():
    """A fresh axes on its own figure; every figure is closed after the test."""
    fig, axes = plt.subplots()
    yield axes
    plt.close("all")


@pytest.fixture(scope="module")
def close_panel(config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    dates = pd.bdate_range("2023-01-02", periods=N_OBS)
    steps = rng.normal(0.0003, 0.012, size=(N_OBS, len(TICKERS)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=TICKERS)


def _backtester(config, close_panel, weight_fn) -> WalkForwardBacktester:
    volume = pd.DataFrame(1e6, index=close_panel.index, columns=close_panel.columns)
    return WalkForwardBacktester(
        config=config,
        price_data=close_panel,
        volume_data=volume,
        signal_fn=sma_crossover_portfolio,
        weight_fn=weight_fn,
        cost_model=linear_cost,
        cost_bps=10.0,
        window_params={"window_type": "rolling", "window_len": 21},
        train_frac=0.5,
    )


@pytest.fixture(scope="module")
def run(config, close_panel) -> dict:
    """A run record from a trading backtest."""
    backtester = _backtester(config, close_panel, equal_active_weights)
    backtester.run([5, 21], [])
    return backtester.runs[1]


@pytest.fixture(scope="module")
def flat_run(config, close_panel) -> dict:
    """A run record from a backtest that never trades: zero returns throughout."""

    def zero_weights(config, signals):
        return signals * 0.0

    backtester = _backtester(config, close_panel, zero_weights)
    backtester.run([5, 21], [])
    return backtester.runs[1]
