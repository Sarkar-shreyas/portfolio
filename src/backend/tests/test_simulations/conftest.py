"""Shared fixtures for the simulations test suite.

All fixtures are deterministic: randomness is driven by ``TestConfig.random_seed``
so results are reproducible across runs. The default ``TestConfig`` simulates
100,000 paths over 252 timesteps, which is fine for the single-period Monte Carlo
VaR functions but slow for the path simulators, so a ``fast_config`` fixture with
much smaller defaults is provided for those.
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
def fast_config() -> TestConfig:
    """A TestConfig with small path/timestep defaults for the path simulators."""
    return TestConfig(n_paths=500, n_timesteps=20)


# ---------------------------------------------------------------------------
# Portfolio inputs for mc_norm_port_var()
# ---------------------------------------------------------------------------


@pytest.fixture
def asset_names() -> list[str]:
    return ["AssetA", "AssetB", "AssetC"]


@pytest.fixture
def mus(asset_names) -> pd.Series:
    """Daily mean returns for a three-asset portfolio."""
    return pd.Series([0.0005, 0.0003, 0.0008], index=asset_names)


@pytest.fixture
def Sigma(asset_names) -> pd.DataFrame:
    """A valid (symmetric, positive-definite) daily covariance matrix."""
    vols = np.array([0.01, 0.015, 0.02])
    corr = np.array(
        [
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=asset_names, columns=asset_names)


@pytest.fixture
def weights(asset_names) -> pd.Series:
    """Long-only portfolio weights summing to one."""
    return pd.Series([0.5, 0.3, 0.2], index=asset_names)
