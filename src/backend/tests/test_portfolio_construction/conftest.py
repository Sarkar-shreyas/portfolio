"""Shared fixtures for the portfolio construction test suite.

The ``signals`` fixture is a small hand-built (dates x tickers) frame that
covers every row-type a weighting scheme has to handle: mixed long/short,
long-only, short-only, all-flat and all-long. ``volatilities`` is aligned
to it with distinct, easily-invertible values so inverse-vol weights can be
computed by hand.
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
def tickers() -> list:
    return ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=6)


@pytest.fixture
def mkt_caps(tickers) -> pd.Series:
    """Market caps in arbitrary units, summing to 1000 for easy arithmetic."""
    return pd.Series([500.0, 250.0, 200.0, 50.0], index=tickers, name="mkt_cap")


@pytest.fixture
def signals(dates, tickers) -> pd.DataFrame:
    """Signal matrix (dates x tickers) with values in {-1, 0, 1}.

    Row-by-row:
        0: mixed     -> AAA=1,  BBB=-1, CCC=1,  DDD=0
        1: long only -> AAA=1,  BBB=1,  CCC=0,  DDD=0
        2: short only-> AAA=0,  BBB=-1, CCC=-1, DDD=-1
        3: all flat  -> 0, 0, 0, 0
        4: all long  -> 1, 1, 1, 1
        5: mixed     -> AAA=-1, BBB=0,  CCC=0,  DDD=1
    """
    data = [
        [1, -1, 1, 0],
        [1, 1, 0, 0],
        [0, -1, -1, -1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [-1, 0, 0, 1],
    ]
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def volatilities(dates, tickers) -> pd.DataFrame:
    """Constant per-ticker volatilities aligned to ``signals``.

    Inverse vols are 10, 5, 2.5 and 1.25 respectively, so the weights of
    any subset are easy to compute by hand.
    """
    vols = np.tile([0.10, 0.20, 0.40, 0.80], (len(dates), 1))
    return pd.DataFrame(vols, index=dates, columns=tickers)


@pytest.fixture
def random_signals(rng, tickers) -> pd.DataFrame:
    """A larger random signal matrix for property-style tests."""
    idx = pd.bdate_range("2023-01-01", periods=252)
    data = rng.choice([-1, 0, 1], size=(len(idx), len(tickers)))
    return pd.DataFrame(data, index=idx, columns=tickers)


@pytest.fixture
def random_volatilities(rng, random_signals) -> pd.DataFrame:
    """Strictly positive random volatilities aligned to ``random_signals``."""
    vols = rng.uniform(0.05, 0.60, size=random_signals.shape)
    return pd.DataFrame(vols, index=random_signals.index, columns=random_signals.columns)
