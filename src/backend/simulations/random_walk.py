import numpy as np
from scipy.stats import norm
from typing import Optional

from src.backend.config import DevConfig

__all__ = ["additive_random_walk", "multiplicative_random_walk", "gbm"]


def additive_random_walk(
    config: DevConfig,
    S0: float,
    mu: float,
    sigma: float,
    n_paths: Optional[int] = None,
    n_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the evolution of an asset's price using an additive random walk. Random variates
    from the normal distribution are used in this process. The first column of the output
    array will contain timestep t = 0, i.e the initial price, for every path.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    S0: float
        The starting price of the asset
    mu: float
        The mean of the normal distribution to sample from
    sigma: float
        The standard deviation of the normal distribution to sample from
    n_paths: int, Optional
        Optional parameter to overwrite the default number of paths to simulate
    n_timesteps: int, Optional
        Optional parameter to overwrite the default number of timesteps to compute
    seed: int, optional
        Optional integer seed to overwrite the default seed used for random operations.

    Returns
    -------
    np.ndarray
        An array of shape (n_paths, n_timesteps+1) containing the simulated price data
    """
    if n_paths is None:
        n_paths = config.n_paths
    if n_timesteps is None:
        n_timesteps = config.n_timesteps
    if seed is None:
        seed = config.random_seed

    random_steps = norm.rvs(
        loc=mu, scale=sigma, size=(n_paths, n_timesteps), random_state=seed
    )
    sim_prices = S0 + np.cumsum(random_steps, axis=1)

    initial_prices = np.full((n_paths, 1), S0)

    return np.concatenate([initial_prices, sim_prices], axis=1)


def multiplicative_random_walk(
    config: DevConfig,
    S0: float,
    mu: float,
    sigma: float,
    n_paths: Optional[int] = None,
    n_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the evolution of an asset's price using an multiplicative random walk. Random
    variates from the normal distribution are used in this process. The first column of
    the output array will contain timestep t = 0, i.e the initial price, for every path.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    S0: float
        The starting price of the asset
    mu: float
        The mean of the normal distribution to sample from
    sigma: float
        The standard deviation of the normal distribution to sample from
    n_paths: int, Optional
        Optional parameter to overwrite the default number of paths to simulate
    n_timesteps: int, Optional
        Optional parameter to overwrite the default number of timesteps to compute
    seed: int, optional
        Optional integer seed to overwrite the default seed used for random operations.

    Returns
    -------
    np.ndarray
        An array of shape (n_paths, n_timesteps+1) containing the simulated price data
    """
    if n_paths is None:
        n_paths = config.n_paths
    if n_timesteps is None:
        n_timesteps = config.n_timesteps
    if seed is None:
        seed = config.random_seed

    random_returns = norm.rvs(
        loc=mu, scale=sigma, size=(n_paths, n_timesteps), random_state=seed
    )
    sim_prices = S0 * np.cumprod(1 + random_returns, axis=1)

    initial_prices = np.full((n_paths, 1), S0)

    return np.concatenate([initial_prices, sim_prices], axis=1)


def gbm(
    config: DevConfig,
    S0: float,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    n_paths: Optional[int] = None,
    n_timesteps: Optional[int] = None,
    T: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the evolution of an asset's price via Geometric Brownian Motion for the given
    parameters. Random shocks are given using the standard Normal distribution. The first
    column of the output array will contain timestep t = 0, i.e the initial price, for every path.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    S0: float
        The starting price of the asset
    mu: float | np.ndarray
        Drift param of the brownian motion. Float input will be broadcasted
    sigma: float | np.ndarray
        Volatility param of the brownian motion. Float input will be broadcasted
    n_paths: int, Optional
        Optional parameter to overwrite the default number of paths to simulate
    n_timesteps: int, Optional
        Optional parameter to overwrite the default number of timesteps to compute
    T: float, Optional
        Optional parameter to overwrite the default time horizon to simulate until
    seed: int, optional
        Optional integer seed to overwrite the default seed used for random operations.

    Returns
    -------
    np.ndarray
        An array of shape (n_paths, n_timesteps+1) containing the simulated price data
    """
    if n_paths is None:
        n_paths = config.n_paths
    if n_timesteps is None:
        n_timesteps = config.n_timesteps
    if T is None:
        T = config.T
    if seed is None:
        seed = config.random_seed
    dt = T / n_timesteps
    if np.isscalar(mu):
        mu_arr = np.full(n_timesteps, mu)
    else:
        mu_arr = mu
    if np.isscalar(sigma):
        sigma_arr = np.full(n_timesteps, sigma)
    else:
        sigma_arr = sigma

    random_shocks = norm.rvs(size=(n_paths, n_timesteps), random_state=seed)

    price_increments = (mu_arr - 0.5 * sigma_arr**2) * dt + (
        sigma_arr * np.sqrt(dt) * random_shocks
    )
    log_paths = np.cumsum(price_increments, axis=1)
    sim_prices = S0 * np.exp(log_paths)
    initial_prices = np.full((n_paths, 1), S0)

    return np.concatenate([initial_prices, sim_prices], axis=1)
