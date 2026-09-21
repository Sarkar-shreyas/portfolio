"""Tests for src.backend.simulations.monte_carlo_var.

Sign convention: both functions return the (1 - conf) quantile of the simulated
*returns* distribution, i.e. a negative number for a loss, and the CVaR as the
mean of the returns beyond that quantile. The analytic benchmarks below follow
the same convention.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.backend.simulations.monte_carlo_var import mc_norm_asset_var, mc_norm_port_var


# ---------------------------------------------------------------------------
# Analytic benchmarks for a normal returns distribution
# ---------------------------------------------------------------------------


def analytic_var(mu: float, sigma: float, conf: float) -> float:
    """(1 - conf) quantile of N(mu, sigma^2)."""
    return mu + sigma * norm.ppf(1 - conf)


def analytic_cvar(mu: float, sigma: float, conf: float) -> float:
    """E[X | X < q] for X ~ N(mu, sigma^2) and q the (1 - conf) quantile."""
    alpha = 1 - conf
    z = norm.ppf(alpha)
    return mu - sigma * norm.pdf(z) / alpha


MU = 0.0005
SIGMA = 0.01


# ---------------------------------------------------------------------------
# mc_norm_asset_var()
# ---------------------------------------------------------------------------


def test_asset_var_returns_tuple_of_two_floats(config):
    result = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)
    assert all(np.isfinite(x) for x in result)


def test_asset_var_reproducible_with_same_seed(config):
    first = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000, seed=7)
    second = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000, seed=7)
    assert first == second


def test_asset_var_differs_across_seeds(config):
    first = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000, seed=1)
    second = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000, seed=2)
    assert first != second


def test_asset_var_default_seed_comes_from_config(config):
    implicit = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000)
    explicit = mc_norm_asset_var(config, MU, SIGMA, n_paths=1000, seed=config.random_seed)
    assert implicit == explicit


def test_asset_var_default_n_paths_and_conf_come_from_config(config):
    implicit = mc_norm_asset_var(config, MU, SIGMA)
    explicit = mc_norm_asset_var(
        config, MU, SIGMA, n_paths=config.n_paths, conf=config.var_conf
    )
    assert implicit == explicit


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_asset_var_matches_analytic_normal_quantile(config, conf):
    mc_var, _ = mc_norm_asset_var(config, MU, SIGMA, n_paths=200_000, conf=conf)
    expected = analytic_var(MU, SIGMA, conf)
    # Monte Carlo error on the quantile is a small fraction of sigma at 200k paths.
    assert mc_var == pytest.approx(expected, abs=0.03 * SIGMA)


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_asset_cvar_matches_analytic_normal_tail_mean(config, conf):
    _, mc_cvar = mc_norm_asset_var(config, MU, SIGMA, n_paths=200_000, conf=conf)
    expected = analytic_cvar(MU, SIGMA, conf)
    assert mc_cvar == pytest.approx(expected, abs=0.05 * SIGMA)


def test_asset_cvar_is_more_extreme_than_var(config):
    mc_var, mc_cvar = mc_norm_asset_var(config, MU, SIGMA, n_paths=10_000)
    # CVaR averages the tail beyond VaR so it must be at least as negative.
    assert mc_cvar < mc_var


def test_asset_var_higher_confidence_is_more_extreme(config):
    var_95, cvar_95 = mc_norm_asset_var(config, MU, SIGMA, n_paths=50_000, conf=0.95)
    var_99, cvar_99 = mc_norm_asset_var(config, MU, SIGMA, n_paths=50_000, conf=0.99)
    assert var_99 < var_95
    assert cvar_99 < cvar_95


def test_asset_var_is_a_loss_for_typical_inputs(config):
    mc_var, mc_cvar = mc_norm_asset_var(config, MU, SIGMA, n_paths=10_000)
    assert mc_var < 0
    assert mc_cvar < 0


def test_asset_var_shifts_with_mean(config):
    # Returns are mu + sigma * Z, so shifting mu by c shifts both the quantile
    # and the tail mean by exactly c when the same random draws are reused.
    shift = 0.01
    base_var, base_cvar = mc_norm_asset_var(config, MU, SIGMA, n_paths=10_000, seed=3)
    shifted_var, shifted_cvar = mc_norm_asset_var(
        config, MU + shift, SIGMA, n_paths=10_000, seed=3
    )
    assert shifted_var == pytest.approx(base_var + shift)
    assert shifted_cvar == pytest.approx(base_cvar + shift)


def test_asset_var_scales_with_volatility(config):
    # With mu = 0 and the same draws, doubling sigma doubles VaR and CVaR.
    var_1, cvar_1 = mc_norm_asset_var(config, 0.0, SIGMA, n_paths=10_000, seed=3)
    var_2, cvar_2 = mc_norm_asset_var(config, 0.0, 2 * SIGMA, n_paths=10_000, seed=3)
    assert var_2 == pytest.approx(2 * var_1)
    assert cvar_2 == pytest.approx(2 * cvar_1)


def test_asset_var_respects_n_paths_override(config):
    # Different path counts consume different draws so results should differ,
    # but both should stay close to the analytic value.
    small_var, _ = mc_norm_asset_var(config, MU, SIGMA, n_paths=5_000)
    large_var, _ = mc_norm_asset_var(config, MU, SIGMA, n_paths=200_000)
    assert small_var != large_var
    expected = analytic_var(MU, SIGMA, config.var_conf)
    assert small_var == pytest.approx(expected, abs=0.2 * SIGMA)
    assert large_var == pytest.approx(expected, abs=0.03 * SIGMA)


# ---------------------------------------------------------------------------
# mc_norm_port_var()
# ---------------------------------------------------------------------------


def _port_moments(mus, Sigma, weights):
    w = weights.reindex(mus.index).to_numpy()
    port_mu = float(w @ mus.to_numpy())
    port_sigma = float(np.sqrt(w @ Sigma.to_numpy() @ w))
    return port_mu, port_sigma


def test_port_var_returns_tuple_of_two_floats(config, mus, Sigma, weights):
    result = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)
    assert all(np.isfinite(x) for x in result)


def test_port_var_reproducible_with_same_seed(config, mus, Sigma, weights):
    first = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000, seed=7)
    second = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000, seed=7)
    assert first == second


def test_port_var_differs_across_seeds(config, mus, Sigma, weights):
    first = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000, seed=1)
    second = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000, seed=2)
    assert first != second


def test_port_var_default_seed_comes_from_config(config, mus, Sigma, weights):
    implicit = mc_norm_port_var(config, mus, Sigma, weights, n_paths=1000)
    explicit = mc_norm_port_var(
        config, mus, Sigma, weights, n_paths=1000, seed=config.random_seed
    )
    assert implicit == explicit


def test_port_var_default_n_paths_and_conf_come_from_config(config, mus, Sigma, weights):
    implicit = mc_norm_port_var(config, mus, Sigma, weights)
    explicit = mc_norm_port_var(
        config, mus, Sigma, weights, n_paths=config.n_paths, conf=config.var_conf
    )
    assert implicit == explicit


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_port_var_matches_analytic_normal_quantile(config, mus, Sigma, weights, conf):
    port_mu, port_sigma = _port_moments(mus, Sigma, weights)
    mc_var, _ = mc_norm_port_var(config, mus, Sigma, weights, n_paths=200_000, conf=conf)
    expected = analytic_var(port_mu, port_sigma, conf)
    assert mc_var == pytest.approx(expected, abs=0.03 * port_sigma)


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_port_cvar_matches_analytic_normal_tail_mean(config, mus, Sigma, weights, conf):
    port_mu, port_sigma = _port_moments(mus, Sigma, weights)
    _, mc_cvar = mc_norm_port_var(config, mus, Sigma, weights, n_paths=200_000, conf=conf)
    expected = analytic_cvar(port_mu, port_sigma, conf)
    assert mc_cvar == pytest.approx(expected, abs=0.05 * port_sigma)


def test_port_cvar_is_more_extreme_than_var(config, mus, Sigma, weights):
    mc_var, mc_cvar = mc_norm_port_var(config, mus, Sigma, weights, n_paths=10_000)
    assert mc_cvar < mc_var


def test_port_var_higher_confidence_is_more_extreme(config, mus, Sigma, weights):
    var_95, cvar_95 = mc_norm_port_var(config, mus, Sigma, weights, n_paths=50_000, conf=0.95)
    var_99, cvar_99 = mc_norm_port_var(config, mus, Sigma, weights, n_paths=50_000, conf=0.99)
    assert var_99 < var_95
    assert cvar_99 < cvar_95


def test_port_var_scales_linearly_with_leverage(config, mus, Sigma, weights):
    # Portfolio returns are R @ w, so scaling every weight by k scales the
    # quantile and tail mean by exactly k when the same draws are reused.
    var_1, cvar_1 = mc_norm_port_var(config, mus, Sigma, weights, n_paths=10_000, seed=3)
    var_2, cvar_2 = mc_norm_port_var(config, mus, Sigma, 2 * weights, n_paths=10_000, seed=3)
    assert var_2 == pytest.approx(2 * var_1)
    assert cvar_2 == pytest.approx(2 * cvar_1)


def test_port_var_uncorrelated_equal_weight_portfolio_diversifies(config):
    # Three identical, uncorrelated, zero-mean assets: the equal-weight portfolio
    # has std sigma / sqrt(3), so its VaR must be less extreme than a single
    # asset held outright.
    names = ["A", "B", "C"]
    sigma = 0.02
    mus = pd.Series(0.0, index=names)
    Sigma = pd.DataFrame(np.eye(3) * sigma**2, index=names, columns=names)

    single = pd.Series([1.0, 0.0, 0.0], index=names)
    equal = pd.Series([1 / 3] * 3, index=names)

    single_var, _ = mc_norm_port_var(config, mus, Sigma, single, n_paths=100_000)
    equal_var, _ = mc_norm_port_var(config, mus, Sigma, equal, n_paths=100_000)

    assert equal_var > single_var
    assert single_var == pytest.approx(
        analytic_var(0.0, sigma, config.var_conf), abs=0.03 * sigma
    )
    assert equal_var == pytest.approx(
        analytic_var(0.0, sigma / np.sqrt(3), config.var_conf), abs=0.03 * sigma
    )


def test_port_var_single_asset_portfolio_matches_asset_var(config):
    # A one-asset portfolio must be supported and must reduce to the
    # single-asset normal VaR.
    mu, sigma = 0.001, 0.02
    mus = pd.Series([mu], index=["A"])
    Sigma = pd.DataFrame([[sigma**2]], index=["A"], columns=["A"])
    weights = pd.Series([1.0], index=["A"])

    mc_var, mc_cvar = mc_norm_port_var(config, mus, Sigma, weights, n_paths=200_000)

    assert mc_var == pytest.approx(analytic_var(mu, sigma, config.var_conf), abs=0.03 * sigma)
    assert mc_cvar == pytest.approx(analytic_cvar(mu, sigma, config.var_conf), abs=0.05 * sigma)


def test_port_var_aligns_weights_by_index(config, mus, Sigma, weights):
    # Weights are a labelled Series, so their order must not matter: a
    # reordered Series with the same labels must give the same result.
    reordered = weights.reindex(["AssetC", "AssetA", "AssetB"])
    original = mc_norm_port_var(config, mus, Sigma, weights, n_paths=10_000, seed=3)
    shuffled = mc_norm_port_var(config, mus, Sigma, reordered, n_paths=10_000, seed=3)
    assert shuffled == pytest.approx(original)


def test_port_var_zero_weight_asset_does_not_contribute(config, mus, Sigma):
    # Putting all weight on one asset must give that asset alone, regardless
    # of the other assets in the covariance matrix.
    weights = pd.Series([0.0, 1.0, 0.0], index=mus.index)
    mc_var, _ = mc_norm_port_var(config, mus, Sigma, weights, n_paths=200_000)
    mu_b = mus["AssetB"]
    sigma_b = np.sqrt(Sigma.loc["AssetB", "AssetB"])
    assert mc_var == pytest.approx(
        analytic_var(mu_b, sigma_b, config.var_conf), abs=0.03 * sigma_b
    )
