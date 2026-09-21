import numpy as np
import pandas as pd
import pytest

from src.backend.portfolio_construction.dynamic_weights import (
    equal_active_weights,
    equal_split_ls_weights,
    inverse_volatility_weighted,
    inverse_volatility_split_ls_weights,
)


# Conventions asserted throughout this module:
#   * A signal of 1 is a long, -1 a short, 0 flat.
#   * Long weights are positive and short weights are negative.
#   * ``tot_exposure`` caps the *gross* exposure (sum of |weights|) of a row.
#   * ``long_exposure`` / ``short_exposure`` are positive magnitudes: the long
#     leg sums to +long_exposure and the short leg to -short_exposure.
#   * A row with no active signals produces all-zero weights, never NaN.


def _assert_frame_shape(weights: pd.DataFrame, signals: pd.DataFrame):
    assert isinstance(weights, pd.DataFrame)
    pd.testing.assert_index_equal(weights.index, signals.index)
    pd.testing.assert_index_equal(weights.columns, signals.columns)
    assert not weights.isnull().any().any()


def _assert_signs_match(weights: pd.DataFrame, signals: pd.DataFrame):
    """Weights must be > 0 on longs, < 0 on shorts and exactly 0 on flats."""
    w, s = weights.to_numpy(), signals.to_numpy()
    assert (w[s == 1] > 0).all()
    assert (w[s == -1] < 0).all()
    assert (w[s == 0] == 0).all()


def _assert_split_leg_sums(
    weights: pd.DataFrame, signals: pd.DataFrame, long_exposure, short_exposure
):
    """Each row's long leg sums to +long_exposure (if any longs) and short leg to
    -short_exposure (if any shorts)."""
    long_sum = weights.where(signals.eq(1), 0.0).sum(axis=1)
    short_sum = weights.where(signals.eq(-1), 0.0).sum(axis=1)
    has_long = signals.eq(1).any(axis=1)
    has_short = signals.eq(-1).any(axis=1)
    np.testing.assert_allclose(long_sum[has_long], long_exposure)
    np.testing.assert_allclose(long_sum[~has_long], 0.0)
    np.testing.assert_allclose(short_sum[has_short], -short_exposure)
    np.testing.assert_allclose(short_sum[~has_short], 0.0)


# ---------------------------------------------------------------------------
# equal_active_weights()
# ---------------------------------------------------------------------------


def test_equal_active_shape(config, signals):
    _assert_frame_shape(equal_active_weights(config, signals), signals)


def test_equal_active_known_values(config, signals):
    result = equal_active_weights(config, signals, tot_exposure=1.0)
    expected = pd.DataFrame(
        [
            [1 / 3, -1 / 3, 1 / 3, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, -1 / 3, -1 / 3, -1 / 3],
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [-0.5, 0.0, 0.0, 0.5],
        ],
        index=signals.index,
        columns=signals.columns,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_equal_active_gross_exposure_equals_config_default(config, signals):
    result = equal_active_weights(config, signals)
    gross = result.abs().sum(axis=1)
    active = signals.ne(0).any(axis=1)
    np.testing.assert_allclose(gross[active], config.tot_exposure)


def test_equal_active_flat_row_is_all_zero(config, signals):
    result = equal_active_weights(config, signals)
    flat_rows = ~signals.ne(0).any(axis=1)
    assert flat_rows.any()
    assert (result[flat_rows] == 0).all().all()


def test_equal_active_signs_match_signals(config, signals):
    _assert_signs_match(equal_active_weights(config, signals), signals)


def test_equal_active_magnitudes_equal_within_row(config, signals):
    result = equal_active_weights(config, signals)
    for date, row in result.iterrows():
        active = row[signals.loc[date].ne(0)].abs()
        if len(active):
            np.testing.assert_allclose(active.to_numpy(), active.iloc[0])


def test_equal_active_custom_exposure(config, signals):
    half = equal_active_weights(config, signals, tot_exposure=0.5)
    full = equal_active_weights(config, signals, tot_exposure=1.0)
    pd.testing.assert_frame_equal(half, full * 0.5)
    active = signals.ne(0).any(axis=1)
    np.testing.assert_allclose(half.abs().sum(axis=1)[active], 0.5)


def test_equal_active_explicit_default_matches_implicit(config, signals):
    implicit = equal_active_weights(config, signals)
    explicit = equal_active_weights(config, signals, tot_exposure=config.tot_exposure)
    pd.testing.assert_frame_equal(implicit, explicit)


def test_equal_active_net_exposure(config, signals):
    """Net exposure is (n_long - n_short) / n_active * tot_exposure."""
    result = equal_active_weights(config, signals, tot_exposure=1.0)
    n_long = signals.eq(1).sum(axis=1)
    n_short = signals.eq(-1).sum(axis=1)
    n_active = (n_long + n_short).replace(0, np.nan)
    expected_net = ((n_long - n_short) / n_active).fillna(0.0)
    np.testing.assert_allclose(result.sum(axis=1), expected_net)


def test_equal_active_input_not_mutated(config, signals):
    original = signals.copy(deep=True)
    equal_active_weights(config, signals)
    pd.testing.assert_frame_equal(signals, original)


def test_equal_active_random_signals_properties(config, random_signals):
    result = equal_active_weights(config, random_signals, tot_exposure=0.8)
    _assert_frame_shape(result, random_signals)
    _assert_signs_match(result, random_signals)
    active = random_signals.ne(0).any(axis=1)
    np.testing.assert_allclose(result.abs().sum(axis=1)[active], 0.8)
    np.testing.assert_allclose(result.abs().sum(axis=1)[~active], 0.0)


# ---------------------------------------------------------------------------
# equal_split_ls_weights()
# ---------------------------------------------------------------------------


def test_equal_split_shape(config, signals):
    _assert_frame_shape(equal_split_ls_weights(config, signals), signals)


def test_equal_split_known_values(config, signals):
    result = equal_split_ls_weights(config, signals, long_exposure=0.5, short_exposure=0.5)
    expected = pd.DataFrame(
        [
            [0.25, -0.5, 0.25, 0.0],
            [0.25, 0.25, 0.0, 0.0],
            [0.0, -1 / 6, -1 / 6, -1 / 6],
            [0.0, 0.0, 0.0, 0.0],
            [0.125, 0.125, 0.125, 0.125],
            [-0.5, 0.0, 0.0, 0.5],
        ],
        index=signals.index,
        columns=signals.columns,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_equal_split_short_weights_are_negative(config, signals):
    result = equal_split_ls_weights(config, signals)
    shorts = result.to_numpy()[signals.eq(-1).to_numpy()]
    assert len(shorts) > 0
    assert (shorts < 0).all()


def test_equal_split_signs_match_signals(config, signals):
    _assert_signs_match(equal_split_ls_weights(config, signals), signals)


def test_equal_split_leg_sums_with_config_defaults(config, signals):
    result = equal_split_ls_weights(config, signals)
    _assert_split_leg_sums(result, signals, config.long_exposure, config.short_exposure)


def test_equal_split_long_only_row_keeps_longs(config, signals):
    """A row with no shorts must still carry its long leg (not be zeroed)."""
    result = equal_split_ls_weights(config, signals, long_exposure=0.5, short_exposure=0.5)
    row = result.iloc[1]  # AAA=1, BBB=1, no shorts
    np.testing.assert_allclose(row.to_numpy(), [0.25, 0.25, 0.0, 0.0])


def test_equal_split_short_only_row_keeps_shorts(config, signals):
    """A row with no longs must still carry its short leg (not be zeroed)."""
    result = equal_split_ls_weights(config, signals, long_exposure=0.5, short_exposure=0.5)
    row = result.iloc[2]  # BBB=CCC=DDD=-1, no longs
    np.testing.assert_allclose(row.to_numpy(), [0.0, -1 / 6, -1 / 6, -1 / 6])


def test_equal_split_all_long_row(config, signals):
    result = equal_split_ls_weights(config, signals, long_exposure=0.6, short_exposure=0.4)
    np.testing.assert_allclose(result.iloc[4].to_numpy(), 0.15)


def test_equal_split_flat_row_is_all_zero(config, signals):
    result = equal_split_ls_weights(config, signals)
    assert (result.iloc[3] == 0).all()


def test_equal_split_equal_magnitudes_within_each_leg(config, signals):
    result = equal_split_ls_weights(config, signals)
    for date, row in result.iterrows():
        sig = signals.loc[date]
        longs = row[sig.eq(1)]
        shorts = row[sig.eq(-1)]
        if len(longs):
            np.testing.assert_allclose(longs.to_numpy(), longs.iloc[0])
        if len(shorts):
            np.testing.assert_allclose(shorts.to_numpy(), shorts.iloc[0])


def test_equal_split_custom_asymmetric_exposures(config, signals):
    result = equal_split_ls_weights(config, signals, long_exposure=1.3, short_exposure=0.3)
    _assert_split_leg_sums(result, signals, 1.3, 0.3)


def test_equal_split_only_long_override(config, signals):
    result = equal_split_ls_weights(config, signals, long_exposure=0.9)
    _assert_split_leg_sums(result, signals, 0.9, config.short_exposure)


def test_equal_split_only_short_override(config, signals):
    result = equal_split_ls_weights(config, signals, short_exposure=0.2)
    _assert_split_leg_sums(result, signals, config.long_exposure, 0.2)


def test_equal_split_explicit_default_matches_implicit(config, signals):
    implicit = equal_split_ls_weights(config, signals)
    explicit = equal_split_ls_weights(
        config,
        signals,
        long_exposure=config.long_exposure,
        short_exposure=config.short_exposure,
    )
    pd.testing.assert_frame_equal(implicit, explicit)


def test_equal_split_input_not_mutated(config, signals):
    original = signals.copy(deep=True)
    equal_split_ls_weights(config, signals)
    pd.testing.assert_frame_equal(signals, original)


def test_equal_split_random_signals_properties(config, random_signals):
    result = equal_split_ls_weights(config, random_signals, long_exposure=0.7, short_exposure=0.3)
    _assert_frame_shape(result, random_signals)
    _assert_signs_match(result, random_signals)
    _assert_split_leg_sums(result, random_signals, 0.7, 0.3)


def test_equal_split_dollar_neutral_when_both_legs_present(config, signals):
    result = equal_split_ls_weights(config, signals, long_exposure=0.5, short_exposure=0.5)
    both = signals.eq(1).any(axis=1) & signals.eq(-1).any(axis=1)
    assert both.any()
    np.testing.assert_allclose(result[both].sum(axis=1), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# inverse_volatility_weighted()
# ---------------------------------------------------------------------------


def test_inv_vol_shape(config, signals, volatilities):
    _assert_frame_shape(inverse_volatility_weighted(config, signals, volatilities), signals)


def test_inv_vol_known_values(config, signals, volatilities):
    result = inverse_volatility_weighted(config, signals, volatilities, tot_exposure=1.0)
    expected = pd.DataFrame(
        [
            [10 / 17.5, -5 / 17.5, 2.5 / 17.5, 0.0],
            [10 / 15, 5 / 15, 0.0, 0.0],
            [0.0, -5 / 8.75, -2.5 / 8.75, -1.25 / 8.75],
            [0.0, 0.0, 0.0, 0.0],
            [10 / 18.75, 5 / 18.75, 2.5 / 18.75, 1.25 / 18.75],
            [-10 / 11.25, 0.0, 0.0, 1.25 / 11.25],
        ],
        index=signals.index,
        columns=signals.columns,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_inv_vol_gross_exposure_equals_config_default(config, signals, volatilities):
    result = inverse_volatility_weighted(config, signals, volatilities)
    active = signals.ne(0).any(axis=1)
    np.testing.assert_allclose(result.abs().sum(axis=1)[active], config.tot_exposure)
    np.testing.assert_allclose(result.abs().sum(axis=1)[~active], 0.0)


def test_inv_vol_signs_match_signals(config, signals, volatilities):
    _assert_signs_match(inverse_volatility_weighted(config, signals, volatilities), signals)


def test_inv_vol_flat_row_is_all_zero(config, signals, volatilities):
    result = inverse_volatility_weighted(config, signals, volatilities)
    assert (result.iloc[3] == 0).all()


def test_inv_vol_weights_proportional_to_inverse_vol(config, signals, volatilities):
    """Within a row, |w_i| * vol_i is constant across active tickers."""
    result = inverse_volatility_weighted(config, signals, volatilities)
    risk_contrib = result.abs() * volatilities
    for date, row in risk_contrib.iterrows():
        active = row[signals.loc[date].ne(0)]
        if len(active):
            np.testing.assert_allclose(active.to_numpy(), active.iloc[0])


def test_inv_vol_lower_vol_gets_larger_weight(config, signals, volatilities):
    result = inverse_volatility_weighted(config, signals, volatilities)
    all_long = result.iloc[4].abs()  # vols ascend AAA < BBB < CCC < DDD
    assert all_long["AAA"] > all_long["BBB"] > all_long["CCC"] > all_long["DDD"]


def test_inv_vol_custom_exposure(config, signals, volatilities):
    half = inverse_volatility_weighted(config, signals, volatilities, tot_exposure=0.5)
    full = inverse_volatility_weighted(config, signals, volatilities, tot_exposure=1.0)
    pd.testing.assert_frame_equal(half, full * 0.5)


def test_inv_vol_explicit_default_matches_implicit(config, signals, volatilities):
    implicit = inverse_volatility_weighted(config, signals, volatilities)
    explicit = inverse_volatility_weighted(
        config, signals, volatilities, tot_exposure=config.tot_exposure
    )
    pd.testing.assert_frame_equal(implicit, explicit)


def test_inv_vol_equal_vols_reduce_to_equal_active(config, signals):
    vols = pd.DataFrame(0.2, index=signals.index, columns=signals.columns)
    result = inverse_volatility_weighted(config, signals, vols)
    expected = equal_active_weights(config, signals)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_inv_vol_scale_invariant_in_vol(config, signals, volatilities):
    """Scaling every vol by a constant leaves the weights unchanged."""
    base = inverse_volatility_weighted(config, signals, volatilities)
    scaled = inverse_volatility_weighted(config, signals, volatilities * 3.7)
    pd.testing.assert_frame_equal(base, scaled)


def test_inv_vol_nan_volatility_drops_ticker_and_renormalises(config, signals, volatilities):
    """A ticker with unknown vol cannot be sized: weight 0, others re-scaled."""
    vols = volatilities.copy()
    vols.loc[signals.index[0], "AAA"] = np.nan
    result = inverse_volatility_weighted(config, signals, vols, tot_exposure=1.0)
    row = result.iloc[0]  # signals: AAA=1, BBB=-1, CCC=1
    assert row["AAA"] == 0
    np.testing.assert_allclose(row[["BBB", "CCC"]].to_numpy(), [-5 / 7.5, 2.5 / 7.5])
    assert not result.isnull().any().any()


def test_inv_vol_inputs_not_mutated(config, signals, volatilities):
    sig_orig = signals.copy(deep=True)
    vol_orig = volatilities.copy(deep=True)
    inverse_volatility_weighted(config, signals, volatilities)
    pd.testing.assert_frame_equal(signals, sig_orig)
    pd.testing.assert_frame_equal(volatilities, vol_orig)


def test_inv_vol_random_properties(config, random_signals, random_volatilities):
    result = inverse_volatility_weighted(config, random_signals, random_volatilities, tot_exposure=1.2)
    _assert_frame_shape(result, random_signals)
    _assert_signs_match(result, random_signals)
    active = random_signals.ne(0).any(axis=1)
    np.testing.assert_allclose(result.abs().sum(axis=1)[active], 1.2)
    np.testing.assert_allclose(result.abs().sum(axis=1)[~active], 0.0)


# ---------------------------------------------------------------------------
# inverse_volatility_split_ls_weights()
# ---------------------------------------------------------------------------


def test_inv_vol_split_shape(config, signals, volatilities):
    _assert_frame_shape(
        inverse_volatility_split_ls_weights(config, signals, volatilities), signals
    )


def test_inv_vol_split_known_values(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(
        config, signals, volatilities, long_exposure=0.5, short_exposure=0.5
    )
    expected = pd.DataFrame(
        [
            [0.4, -0.5, 0.1, 0.0],
            [1 / 3, 1 / 6, 0.0, 0.0],
            [0.0, -5 / 17.5, -2.5 / 17.5, -1.25 / 17.5],
            [0.0, 0.0, 0.0, 0.0],
            [10 / 37.5, 5 / 37.5, 2.5 / 37.5, 1.25 / 37.5],
            [-0.5, 0.0, 0.0, 0.5],
        ],
        index=signals.index,
        columns=signals.columns,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_inv_vol_split_short_weights_are_negative(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities)
    shorts = result.to_numpy()[signals.eq(-1).to_numpy()]
    assert len(shorts) > 0
    assert (shorts < 0).all()


def test_inv_vol_split_signs_match_signals(config, signals, volatilities):
    _assert_signs_match(
        inverse_volatility_split_ls_weights(config, signals, volatilities), signals
    )


def test_inv_vol_split_leg_sums_with_config_defaults(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities)
    _assert_split_leg_sums(result, signals, config.long_exposure, config.short_exposure)


def test_inv_vol_split_long_only_row_keeps_longs(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(
        config, signals, volatilities, long_exposure=0.5, short_exposure=0.5
    )
    np.testing.assert_allclose(result.iloc[1].to_numpy(), [1 / 3, 1 / 6, 0.0, 0.0])


def test_inv_vol_split_short_only_row_keeps_shorts(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(
        config, signals, volatilities, long_exposure=0.5, short_exposure=0.5
    )
    np.testing.assert_allclose(
        result.iloc[2].to_numpy(), [0.0, -5 / 17.5, -2.5 / 17.5, -1.25 / 17.5]
    )


def test_inv_vol_split_flat_row_is_all_zero(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities)
    assert (result.iloc[3] == 0).all()


def test_inv_vol_split_proportional_to_inverse_vol_within_leg(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities)
    risk_contrib = result.abs() * volatilities
    for date, row in risk_contrib.iterrows():
        sig = signals.loc[date]
        for mask in (sig.eq(1), sig.eq(-1)):
            leg = row[mask]
            if len(leg):
                np.testing.assert_allclose(leg.to_numpy(), leg.iloc[0])


def test_inv_vol_split_custom_asymmetric_exposures(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(
        config, signals, volatilities, long_exposure=1.0, short_exposure=0.25
    )
    _assert_split_leg_sums(result, signals, 1.0, 0.25)


def test_inv_vol_split_only_long_override(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities, long_exposure=0.8)
    _assert_split_leg_sums(result, signals, 0.8, config.short_exposure)


def test_inv_vol_split_only_short_override(config, signals, volatilities):
    result = inverse_volatility_split_ls_weights(config, signals, volatilities, short_exposure=0.1)
    _assert_split_leg_sums(result, signals, config.long_exposure, 0.1)


def test_inv_vol_split_explicit_default_matches_implicit(config, signals, volatilities):
    implicit = inverse_volatility_split_ls_weights(config, signals, volatilities)
    explicit = inverse_volatility_split_ls_weights(
        config,
        signals,
        volatilities,
        long_exposure=config.long_exposure,
        short_exposure=config.short_exposure,
    )
    pd.testing.assert_frame_equal(implicit, explicit)


def test_inv_vol_split_equal_vols_reduce_to_equal_split(config, signals):
    vols = pd.DataFrame(0.3, index=signals.index, columns=signals.columns)
    result = inverse_volatility_split_ls_weights(config, signals, vols)
    expected = equal_split_ls_weights(config, signals)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_inv_vol_split_scale_invariant_in_vol(config, signals, volatilities):
    base = inverse_volatility_split_ls_weights(config, signals, volatilities)
    scaled = inverse_volatility_split_ls_weights(config, signals, volatilities * 2.5)
    pd.testing.assert_frame_equal(base, scaled)


def test_inv_vol_split_inputs_not_mutated(config, signals, volatilities):
    sig_orig = signals.copy(deep=True)
    vol_orig = volatilities.copy(deep=True)
    inverse_volatility_split_ls_weights(config, signals, volatilities)
    pd.testing.assert_frame_equal(signals, sig_orig)
    pd.testing.assert_frame_equal(volatilities, vol_orig)


def test_inv_vol_split_random_properties(config, random_signals, random_volatilities):
    result = inverse_volatility_split_ls_weights(
        config, random_signals, random_volatilities, long_exposure=0.6, short_exposure=0.4
    )
    _assert_frame_shape(result, random_signals)
    _assert_signs_match(result, random_signals)
    _assert_split_leg_sums(result, random_signals, 0.6, 0.4)


# ---------------------------------------------------------------------------
# Cross-scheme consistency
# ---------------------------------------------------------------------------


def test_split_schemes_match_total_schemes_on_dollar_neutral_rows(config, signals, volatilities):
    """On rows with both legs present, the gross exposure of the split schemes
    must equal long_exposure + short_exposure."""
    eq_split = equal_split_ls_weights(config, signals, long_exposure=0.5, short_exposure=0.5)
    iv_split = inverse_volatility_split_ls_weights(
        config, signals, volatilities, long_exposure=0.5, short_exposure=0.5
    )
    both = signals.eq(1).any(axis=1) & signals.eq(-1).any(axis=1)
    np.testing.assert_allclose(eq_split[both].abs().sum(axis=1), 1.0)
    np.testing.assert_allclose(iv_split[both].abs().sum(axis=1), 1.0)
