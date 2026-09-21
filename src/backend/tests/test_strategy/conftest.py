"""Shared fixtures for the strategy test suite.

Fixtures build small, hand-constructed indicator inputs (SMA/EMA pairs and
RSI series) that deliberately include the edge cases a signal generator has
to handle: warm-up NaNs, exact equality / threshold values, and neutral
regions. Larger synthetic data uses a fixed seed via ``TestConfig``.
"""

import numpy as np
import pandas as pd
import pytest

from src.backend.config import TestConfig


@pytest.fixture
def config() -> TestConfig:
    """A TestConfig instance with fixed, reproducible parameters."""
    return TestConfig()


@pytest.fixture
def rng(config) -> np.random.Generator:
    """A seeded random number generator for reproducible synthetic data."""
    return np.random.default_rng(config.random_seed)


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    """A short business-day index shared by the hand-built fixtures."""
    return pd.bdate_range("2024-01-01", periods=8)


@pytest.fixture
def crossover_values() -> dict:
    """Raw short/long moving-average values and the expected signal.

    Row-by-row:
        0: long is NaN (warm-up)            -> 0
        1: long is NaN (warm-up)            -> 0
        2: short == long                    -> 0
        3: short > long                     -> 1
        4: short > long                     -> 1
        5: short < long                     -> -1
        6: short is NaN                     -> 0
        7: short < long                     -> -1
    """
    return {
        "short": [100.0, 101.0, 102.0, 104.0, 106.0, 103.0, np.nan, 99.0],
        "long": [np.nan, np.nan, 102.0, 103.0, 103.5, 104.0, 104.0, 104.0],
        "expected": [0, 0, 0, 1, 1, -1, 0, -1],
    }


@pytest.fixture
def sma_frame(dates, crossover_values) -> pd.DataFrame:
    """A ``['sma_short', 'sma_long']`` frame covering all crossover cases."""
    return pd.DataFrame(
        {"sma_short": crossover_values["short"], "sma_long": crossover_values["long"]},
        index=dates,
    )


@pytest.fixture
def ema_frame(dates, crossover_values) -> pd.DataFrame:
    """A ``['ema_short', 'ema_long']`` frame covering all crossover cases."""
    return pd.DataFrame(
        {"ema_short": crossover_values["short"], "ema_long": crossover_values["long"]},
        index=dates,
    )


@pytest.fixture
def expected_crossover_signal(dates, crossover_values) -> pd.Series:
    """The correct signal for ``sma_frame`` / ``ema_frame``."""
    return pd.Series(crossover_values["expected"], index=dates, name="signal")


@pytest.fixture
def rsi_series(dates) -> pd.Series:
    """An RSI series covering oversold, overbought, neutral, boundary and NaN.

    Row-by-row (defaults: oversold < 30, overbought > 70):
        0: NaN   (warm-up)        -> 0
        1: 25.0  (oversold)       -> 1
        2: 50.0  (neutral)        -> 0
        3: 75.0  (overbought)     -> -1
        4: 30.0  (on threshold)   -> 0
        5: 70.0  (on threshold)   -> 0
        6: 10.5  (oversold)       -> 1
        7: 99.9  (overbought)     -> -1
    """
    return pd.Series(
        [np.nan, 25.0, 50.0, 75.0, 30.0, 70.0, 10.5, 99.9], index=dates, name="rsi"
    )


@pytest.fixture
def expected_rsi_signal(dates) -> pd.Series:
    """The correct signal for ``rsi_series`` using the config defaults."""
    return pd.Series([0, 1, 0, -1, 0, 0, 1, -1], index=dates, name="signal")


@pytest.fixture
def price_series(rng) -> pd.Series:
    """Synthetic daily close-price series generated via a geometric random walk."""
    n = 252
    idx = pd.bdate_range("2023-01-01", periods=n)
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * np.cumprod(1 + daily_returns)
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture
def returns_series(price_series) -> pd.Series:
    """Simple daily returns derived from ``price_series``."""
    return price_series.pct_change().dropna()
