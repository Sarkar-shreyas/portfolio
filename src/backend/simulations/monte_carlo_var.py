from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from src.backend.config import DevConfig


def mc_norm_asset_var(
    config: DevConfig,
    mu: float,
    sigma: float,
    n_paths: Optional[int] = None,
    conf: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple:
    """
    Computes the Value-at-Risk at a given confidence interval by generating an array of
    random returns, assuming a normal distribution on the given returns and volatility
    data. This function performs the simulation for a single period.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    mu: float
        A float representing the average returns for an individual asset
    sigma: float
        A float representing the volatiltiy for an individual asset
    n_paths: int, optional
        Optional parameter to overwrite the default number of paths to simulate
    conf: float, optional
        Optional parameter to overwrite the default confidence interval to compute the
        VaR over.
    seed: int, optional
        Optional integer seed to overwrite the default seed used for random operations.

    Returns
    -------
    tuple[float, float]
        A tuple containing the simulated Value-at-Risk and Conditional Value-at-Risk for
        the given confidence level.

    """
    if n_paths is None:
        n_paths = config.n_paths
    if conf is None:
        conf = config.var_conf
    if seed is None:
        seed = config.random_seed

    rand_rets = np.array(norm.rvs(loc=mu, scale=sigma, size=n_paths, random_state=seed))
    mc_var = np.quantile(rand_rets, 1 - conf)
    mc_cvar = np.mean(rand_rets[rand_rets < mc_var])
    return mc_var, mc_cvar


def mc_norm_port_var(
    config: DevConfig,
    mus: pd.Series,
    Sigma: pd.DataFrame,
    weights: pd.Series,
    n_paths: Optional[int] = None,
    conf: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple:
    """
    Computes the Value-at-Risk at a given confidence interval by generating a matrix of
    random returns, and computing portfolio returns with the given weights array. Assumes
    a normal distribution on the given asset returns and covariance matrix. This function
    performs the simulation for a single period.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    mus: pd.Series
        A pandas series containing mean returns data for each individual asset of the portfolio
    Sigma: pd.DataFrame
        A pandas dataframe containing the covariance matrix for a portfolio of assets
    weights: pd.Series
        A pandas series containing the individual weights for each asset in the portfolio
    n_paths: int, optional
        Optional parameter to overwrite the default number of paths to simulate
    conf: float, optional
        Optional parameter to overwrite the default confidence interval to compute the
        VaR over.
    seed: int, optional
        Optional integer seed to overwrite the default seed used for random operations.

    Returns
    -------
    tuple[float, float]
        A tuple containing the simulated Value-at-Risk and Conditional Value-at-Risk for
        the given confidence level.

    """
    if n_paths is None:
        n_paths = config.n_paths
    if conf is None:
        conf = config.var_conf
    if seed is None:
        seed = config.random_seed

    rand_rets = multivariate_normal.rvs(
        mean=mus.to_numpy(),
        cov=Sigma.to_numpy(),  # type: ignore
        size=n_paths,
        random_state=seed,
    )
    rand_rets = np.asarray(rand_rets).reshape(n_paths, len(mus))
    rand_port_rets = rand_rets @ weights.reindex(mus.index).to_numpy()
    mc_var = np.quantile(rand_port_rets, 1 - conf)
    mc_cvar = np.mean(rand_port_rets[rand_port_rets < mc_var])
    return mc_var, mc_cvar
