"""Shared fixtures for the analysis test suite.

These fixtures generate deterministic synthetic data (fixed random seed via
``TestConfig.random_seed``) so tests are reproducible across runs.
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
def price_series(rng) -> pd.Series:
    """Synthetic daily close-price series generated via a geometric random walk."""
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * np.cumprod(1 + daily_returns)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def returns_series(price_series) -> pd.Series:
    """Simple daily returns derived from ``price_series``."""
    return price_series.pct_change().dropna()


@pytest.fixture
def multi_asset_returns(rng) -> pd.DataFrame:
    """Synthetic multi-asset daily returns with deliberate cross-correlation."""
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    market = rng.normal(0.0004, 0.01, size=n)
    asset_a = 0.8 * market + rng.normal(0, 0.004, size=n)
    asset_b = 0.3 * market + rng.normal(0, 0.008, size=n)
    asset_c = rng.normal(0.0002, 0.012, size=n)
    return pd.DataFrame(
        {"AssetA": asset_a, "AssetB": asset_b, "AssetC": asset_c},
        index=dates,
    )
