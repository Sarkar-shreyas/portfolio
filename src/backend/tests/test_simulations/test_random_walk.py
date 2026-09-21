"""Tests for src.backend.simulations.random_walk.

Every simulator returns an array of shape (n_paths, n_timesteps + 1) whose first
column is the initial price S0. Tests with ``sigma = 0`` pin down the exact
deterministic dynamics of each process; the remaining tests check the
statistical properties of the random component and the seed/config plumbing.
"""

import numpy as np
import pytest

from src.backend.simulations.random_walk import (
    additive_random_walk,
    gbm,
    multiplicative_random_walk,
)

S0 = 100.0
MU = 0.001
SIGMA = 0.01

SIMULATORS = [additive_random_walk, multiplicative_random_walk, gbm]
SIMULATOR_IDS = ["additive", "multiplicative", "gbm"]


# ---------------------------------------------------------------------------
# Behaviour shared by all three simulators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
@pytest.mark.parametrize("n_paths, n_timesteps", [(1, 1), (7, 3), (50, 100)])
def test_output_shape(fast_config, simulate, n_paths, n_timesteps):
    paths = simulate(fast_config, S0, MU, SIGMA, n_paths=n_paths, n_timesteps=n_timesteps)
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (n_paths, n_timesteps + 1)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_first_column_is_initial_price(fast_config, simulate):
    paths = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10)
    np.testing.assert_array_equal(paths[:, 0], S0)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_output_is_finite(fast_config, simulate):
    paths = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10)
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_paths_are_not_constant(fast_config, simulate):
    paths = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10)
    # Every step after t = 0 should carry random variation across paths.
    assert np.all(np.std(paths[:, 1:], axis=0) > 0)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_reproducible_with_same_seed(fast_config, simulate):
    first = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=7)
    second = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=7)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_differs_across_seeds(fast_config, simulate):
    first = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=1)
    second = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=2)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_default_seed_comes_from_config(fast_config, simulate):
    implicit = simulate(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10)
    explicit = simulate(
        fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=fast_config.random_seed
    )
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_default_n_paths_and_n_timesteps_come_from_config(fast_config, simulate):
    implicit = simulate(fast_config, S0, MU, SIGMA)
    explicit = simulate(
        fast_config,
        S0,
        MU,
        SIGMA,
        n_paths=fast_config.n_paths,
        n_timesteps=fast_config.n_timesteps,
    )
    assert implicit.shape == (fast_config.n_paths, fast_config.n_timesteps + 1)
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("simulate", SIMULATORS, ids=SIMULATOR_IDS)
def test_paths_are_independent_across_rows(fast_config, simulate):
    # Distinct paths must not share the same random draws.
    paths = simulate(fast_config, S0, MU, SIGMA, n_paths=5, n_timesteps=10)
    for i in range(1, paths.shape[0]):
        assert not np.array_equal(paths[0, 1:], paths[i, 1:])


# ---------------------------------------------------------------------------
# additive_random_walk()
# ---------------------------------------------------------------------------


def test_additive_zero_vol_is_linear_drift(fast_config):
    n_timesteps = 10
    paths = additive_random_walk(fast_config, S0, MU, 0.0, n_paths=3, n_timesteps=n_timesteps)
    t = np.arange(n_timesteps + 1)
    expected = np.tile(S0 + MU * t, (3, 1))
    np.testing.assert_allclose(paths, expected)


def test_additive_increments_have_correct_mean_and_std(fast_config):
    n_paths, n_timesteps = 2000, 50
    paths = additive_random_walk(fast_config, S0, MU, SIGMA, n_paths=n_paths, n_timesteps=n_timesteps)
    increments = np.diff(paths, axis=1)
    n_draws = n_paths * n_timesteps
    assert increments.mean() == pytest.approx(MU, abs=5 * SIGMA / np.sqrt(n_draws))
    assert increments.std(ddof=1) == pytest.approx(SIGMA, rel=0.02)


def test_additive_initial_price_shifts_paths(fast_config):
    # An additive walk is S0 + cumulative sum of steps, so changing S0 shifts
    # every price by the same constant when the draws are reused.
    shift = 25.0
    base = additive_random_walk(fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=3)
    shifted = additive_random_walk(
        fast_config, S0 + shift, MU, SIGMA, n_paths=20, n_timesteps=10, seed=3
    )
    np.testing.assert_allclose(shifted, base + shift)


def test_additive_terminal_value_variance_grows_linearly(fast_config):
    # Var(S_t - S0) = t * sigma^2 for an additive walk.
    n_paths, n_timesteps = 5000, 40
    paths = additive_random_walk(fast_config, S0, 0.0, SIGMA, n_paths=n_paths, n_timesteps=n_timesteps)
    terminal_std = np.std(paths[:, -1] - S0, ddof=1)
    assert terminal_std == pytest.approx(SIGMA * np.sqrt(n_timesteps), rel=0.05)


# ---------------------------------------------------------------------------
# multiplicative_random_walk()
# ---------------------------------------------------------------------------


def test_multiplicative_zero_vol_is_compound_growth(fast_config):
    n_timesteps = 10
    paths = multiplicative_random_walk(fast_config, S0, MU, 0.0, n_paths=3, n_timesteps=n_timesteps)
    t = np.arange(n_timesteps + 1)
    expected = np.tile(S0 * (1 + MU) ** t, (3, 1))
    np.testing.assert_allclose(paths, expected)


def test_multiplicative_simple_returns_have_correct_mean_and_std(fast_config):
    n_paths, n_timesteps = 2000, 50
    paths = multiplicative_random_walk(
        fast_config, S0, MU, SIGMA, n_paths=n_paths, n_timesteps=n_timesteps
    )
    simple_returns = paths[:, 1:] / paths[:, :-1] - 1
    n_draws = n_paths * n_timesteps
    assert simple_returns.mean() == pytest.approx(MU, abs=5 * SIGMA / np.sqrt(n_draws))
    assert simple_returns.std(ddof=1) == pytest.approx(SIGMA, rel=0.02)


def test_multiplicative_initial_price_scales_paths(fast_config):
    base = multiplicative_random_walk(
        fast_config, S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=3
    )
    doubled = multiplicative_random_walk(
        fast_config, 2 * S0, MU, SIGMA, n_paths=20, n_timesteps=10, seed=3
    )
    np.testing.assert_allclose(doubled, 2 * base)


def test_multiplicative_prices_stay_positive_for_small_vol(fast_config):
    paths = multiplicative_random_walk(fast_config, S0, MU, SIGMA, n_paths=2000, n_timesteps=50)
    assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# gbm()
# ---------------------------------------------------------------------------

GBM_MU = 0.05
GBM_SIGMA = 0.2


@pytest.mark.parametrize("T, n_timesteps", [(1.0, 12), (1.0, 252), (2.0, 24), (0.5, 10)])
def test_gbm_zero_vol_is_exponential_drift(fast_config, T, n_timesteps):
    # With sigma = 0 the log-price grows deterministically at rate mu * dt per step.
    paths = gbm(fast_config, S0, GBM_MU, 0.0, n_paths=3, n_timesteps=n_timesteps, T=T)
    dt = T / n_timesteps
    t = np.arange(n_timesteps + 1)
    expected = np.tile(S0 * np.exp(GBM_MU * dt * t), (3, 1))
    np.testing.assert_allclose(paths, expected)


def test_gbm_zero_vol_terminal_value_depends_on_horizon(fast_config):
    # Terminal value should be S0 * exp(mu * T) regardless of how many steps
    # the horizon is split into.
    coarse = gbm(fast_config, S0, GBM_MU, 0.0, n_paths=1, n_timesteps=4, T=2.0)
    fine = gbm(fast_config, S0, GBM_MU, 0.0, n_paths=1, n_timesteps=400, T=2.0)
    assert coarse[0, -1] == pytest.approx(S0 * np.exp(GBM_MU * 2.0))
    assert fine[0, -1] == pytest.approx(S0 * np.exp(GBM_MU * 2.0))


@pytest.mark.parametrize("n_timesteps", [12, 252])
def test_gbm_log_returns_have_correct_mean_and_std(fast_config, n_timesteps):
    # Per-step log-returns of a GBM are N((mu - sigma^2 / 2) dt, sigma^2 dt).
    n_paths, T = 2000, 1.0
    dt = T / n_timesteps
    paths = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=n_paths, n_timesteps=n_timesteps, T=T)
    log_returns = np.diff(np.log(paths), axis=1)

    expected_mean = (GBM_MU - 0.5 * GBM_SIGMA**2) * dt
    expected_std = GBM_SIGMA * np.sqrt(dt)
    n_draws = n_paths * n_timesteps

    assert log_returns.mean() == pytest.approx(expected_mean, abs=5 * expected_std / np.sqrt(n_draws))
    assert log_returns.std(ddof=1) == pytest.approx(expected_std, rel=0.02)


@pytest.mark.parametrize("T", [0.5, 1.0])
@pytest.mark.parametrize("n_timesteps", [12, 252])
def test_gbm_terminal_log_price_std_is_sigma_sqrt_T(fast_config, T, n_timesteps):
    # log(S_T / S0) ~ N((mu - sigma^2 / 2) T, sigma^2 T): the terminal
    # dispersion depends on the horizon T, not on how finely it is discretised.
    n_paths = 5000
    paths = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=n_paths, n_timesteps=n_timesteps, T=T)
    terminal_log = np.log(paths[:, -1] / S0)
    assert terminal_log.std(ddof=1) == pytest.approx(GBM_SIGMA * np.sqrt(T), rel=0.05)
    assert terminal_log.mean() == pytest.approx(
        (GBM_MU - 0.5 * GBM_SIGMA**2) * T, abs=5 * GBM_SIGMA * np.sqrt(T) / np.sqrt(n_paths)
    )


def test_gbm_expected_terminal_price_is_S0_exp_mu_T(fast_config):
    # E[S_T] = S0 * exp(mu * T) for a GBM.
    T = 1.0
    paths = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=50_000, n_timesteps=50, T=T)
    assert paths[:, -1].mean() == pytest.approx(S0 * np.exp(GBM_MU * T), rel=0.01)


def test_gbm_prices_are_strictly_positive(fast_config):
    paths = gbm(fast_config, S0, GBM_MU, 1.5, n_paths=2000, n_timesteps=50, T=1.0)
    assert np.all(paths > 0)


def test_gbm_initial_price_scales_paths(fast_config):
    base = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=20, n_timesteps=10, seed=3)
    doubled = gbm(fast_config, 2 * S0, GBM_MU, GBM_SIGMA, n_paths=20, n_timesteps=10, seed=3)
    np.testing.assert_allclose(doubled, 2 * base)


def test_gbm_default_horizon_comes_from_config(fast_config):
    implicit = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=20, n_timesteps=10)
    explicit = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=20, n_timesteps=10, T=fast_config.T)
    np.testing.assert_array_equal(implicit, explicit)


def test_gbm_scalar_and_broadcast_array_params_agree(fast_config):
    # Passing mu / sigma as constant arrays of length n_timesteps must give the
    # same paths as passing the equivalent scalars.
    n_timesteps = 10
    scalar = gbm(fast_config, S0, GBM_MU, GBM_SIGMA, n_paths=20, n_timesteps=n_timesteps, seed=3)
    arrays = gbm(
        fast_config,
        S0,
        np.full(n_timesteps, GBM_MU),
        np.full(n_timesteps, GBM_SIGMA),
        n_paths=20,
        n_timesteps=n_timesteps,
        seed=3,
    )
    np.testing.assert_allclose(arrays, scalar)


def test_gbm_time_varying_drift_with_zero_vol(fast_config):
    n_timesteps, T = 8, 2.0
    dt = T / n_timesteps
    mu_arr = np.linspace(-0.1, 0.3, n_timesteps)
    paths = gbm(fast_config, S0, mu_arr, 0.0, n_paths=2, n_timesteps=n_timesteps, T=T)
    expected_log = np.concatenate([[0.0], np.cumsum(mu_arr * dt)])
    expected = np.tile(S0 * np.exp(expected_log), (2, 1))
    np.testing.assert_allclose(paths, expected)


def test_gbm_time_varying_vol_is_applied_per_step(fast_config):
    # Zero vol and zero drift over the first half of the horizon must leave the
    # price pinned at S0; the second half should then move.
    n_timesteps = 10
    half = n_timesteps // 2
    sigma_arr = np.concatenate([np.zeros(half), np.full(half, GBM_SIGMA)])
    paths = gbm(fast_config, S0, 0.0, sigma_arr, n_paths=20, n_timesteps=n_timesteps)

    np.testing.assert_allclose(paths[:, : half + 1], S0)
    assert np.all(np.std(paths[:, half + 1 :], axis=0) > 0)


def test_gbm_per_path_parameter_arrays_broadcast(fast_config):
    # A (n_paths, n_timesteps) drift array with zero vol gives each path its
    # own deterministic exponential growth.
    n_paths, n_timesteps, T = 3, 5, 1.0
    dt = T / n_timesteps
    mu_arr = np.array([[0.0] * n_timesteps, [0.1] * n_timesteps, [-0.1] * n_timesteps])
    paths = gbm(fast_config, S0, mu_arr, 0.0, n_paths=n_paths, n_timesteps=n_timesteps, T=T)
    t = np.arange(n_timesteps + 1)
    expected = S0 * np.exp(mu_arr[:, :1] * dt * t)
    np.testing.assert_allclose(paths, expected)


def test_gbm_mismatched_parameter_array_length_raises(fast_config):
    with pytest.raises(ValueError):
        gbm(fast_config, S0, np.full(7, GBM_MU), GBM_SIGMA, n_paths=5, n_timesteps=10)
    with pytest.raises(ValueError):
        gbm(fast_config, S0, GBM_MU, np.full(7, GBM_SIGMA), n_paths=5, n_timesteps=10)
