import numpy as np
import pandas as pd
import pytest

from src.backend.portfolio_construction.static_weights import (
    equal_weight_benchmark,
    mkt_cap_weight_benchmark,
    turnover,
)


# ---------------------------------------------------------------------------
# equal_weight_benchmark()
# ---------------------------------------------------------------------------


def test_equal_weight_returns_series_named_weights(config, tickers):
    result = equal_weight_benchmark(config, tickers)
    assert isinstance(result, pd.Series)
    assert result.name == "weights"


def test_equal_weight_index_is_tickers(config, tickers):
    result = equal_weight_benchmark(config, tickers)
    assert list(result.index) == tickers


def test_equal_weight_sums_to_config_exposure(config, tickers):
    result = equal_weight_benchmark(config, tickers)
    assert result.sum() == pytest.approx(config.tot_exposure)


def test_equal_weight_each_is_exposure_over_n(config, tickers):
    result = equal_weight_benchmark(config, tickers)
    expected = np.full(len(tickers), config.tot_exposure / len(tickers))
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_equal_weight_known_values(config):
    result = equal_weight_benchmark(config, ["A", "B", "C", "D"], tot_exposure=1.0)
    expected = pd.Series(
        [0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"], name="weights"
    )
    pd.testing.assert_series_equal(result, expected)


def test_equal_weight_custom_exposure(config, tickers):
    result = equal_weight_benchmark(config, tickers, tot_exposure=0.6)
    assert result.sum() == pytest.approx(0.6)
    np.testing.assert_allclose(result.to_numpy(), 0.6 / len(tickers))


def test_equal_weight_explicit_default_matches_implicit(config, tickers):
    implicit = equal_weight_benchmark(config, tickers)
    explicit = equal_weight_benchmark(config, tickers, tot_exposure=config.tot_exposure)
    pd.testing.assert_series_equal(implicit, explicit)


def test_equal_weight_single_ticker_gets_full_exposure(config):
    result = equal_weight_benchmark(config, ["ONLY"], tot_exposure=0.8)
    assert result.loc["ONLY"] == pytest.approx(0.8)


def test_equal_weight_all_weights_identical(config, tickers):
    result = equal_weight_benchmark(config, tickers)
    assert result.nunique() == 1


def test_equal_weight_scales_linearly_with_exposure(config, tickers):
    one = equal_weight_benchmark(config, tickers, tot_exposure=1.0)
    two = equal_weight_benchmark(config, tickers, tot_exposure=2.0)
    pd.testing.assert_series_equal(two, one * 2)


def test_equal_weight_input_list_not_mutated(config, tickers):
    original = list(tickers)
    equal_weight_benchmark(config, tickers)
    assert tickers == original


# ---------------------------------------------------------------------------
# mkt_cap_weight_benchmark()
# ---------------------------------------------------------------------------


def test_mkt_cap_returns_series_named_weights(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    assert isinstance(result, pd.Series)
    assert result.name == "weights"


def test_mkt_cap_index_preserved(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    pd.testing.assert_index_equal(result.index, mkt_caps.index)


def test_mkt_cap_sums_to_config_exposure(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    assert result.sum() == pytest.approx(config.tot_exposure)


def test_mkt_cap_known_values(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps, tot_exposure=1.0)
    expected = pd.Series([0.5, 0.25, 0.2, 0.05], index=mkt_caps.index, name="weights")
    pd.testing.assert_series_equal(result, expected)


def test_mkt_cap_proportional_to_caps(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    ratios = result / mkt_caps
    np.testing.assert_allclose(ratios.to_numpy(), ratios.iloc[0])


def test_mkt_cap_ordering_preserved(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    assert list(result.sort_values().index) == list(mkt_caps.sort_values().index)


def test_mkt_cap_custom_exposure(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps, tot_exposure=0.5)
    assert result.sum() == pytest.approx(0.5)
    expected = pd.Series(
        [0.25, 0.125, 0.1, 0.025], index=mkt_caps.index, name="weights"
    )
    pd.testing.assert_series_equal(result, expected)


def test_mkt_cap_explicit_default_matches_implicit(config, mkt_caps):
    implicit = mkt_cap_weight_benchmark(config, mkt_caps)
    explicit = mkt_cap_weight_benchmark(
        config, mkt_caps, tot_exposure=config.tot_exposure
    )
    pd.testing.assert_series_equal(implicit, explicit)


def test_mkt_cap_scale_invariant(config, mkt_caps):
    """Weights depend only on the relative sizes of the market caps."""
    base = mkt_cap_weight_benchmark(config, mkt_caps)
    scaled = mkt_cap_weight_benchmark(config, mkt_caps * 1e6)
    pd.testing.assert_series_equal(base, scaled)


def test_mkt_cap_equal_caps_reduce_to_equal_weight(config, tickers):
    caps = pd.Series(100.0, index=tickers)
    result = mkt_cap_weight_benchmark(config, caps)
    expected = equal_weight_benchmark(config, tickers)
    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_mkt_cap_all_weights_non_negative(config, mkt_caps):
    result = mkt_cap_weight_benchmark(config, mkt_caps)
    assert (result >= 0).all()


def test_mkt_cap_single_ticker_gets_full_exposure(config):
    caps = pd.Series([123.0], index=["ONLY"])
    result = mkt_cap_weight_benchmark(config, caps, tot_exposure=0.9)
    assert result.loc["ONLY"] == pytest.approx(0.9)


def test_mkt_cap_input_not_mutated(config, mkt_caps):
    original = mkt_caps.copy(deep=True)
    mkt_cap_weight_benchmark(config, mkt_caps)
    pd.testing.assert_series_equal(mkt_caps, original)


def test_mkt_cap_random_caps_sum_to_exposure(config, rng, tickers):
    caps = pd.Series(rng.uniform(1, 1e4, size=len(tickers)), index=tickers)
    result = mkt_cap_weight_benchmark(config, caps, tot_exposure=0.75)
    assert result.sum() == pytest.approx(0.75)
    np.testing.assert_allclose(result.to_numpy(), 0.75 * caps.to_numpy() / caps.sum())


# ---------------------------------------------------------------------------
# turnover()
#
# Turnover is the L1 size of each rebalance: the sum of absolute weight changes
# from one row to the next, with the first row measured against an empty book,
# so building the initial portfolio is charged like any other trade. It is
# reported per date, aligned to the weights it was computed from.
# ---------------------------------------------------------------------------


@pytest.fixture
def weight_path(dates, tickers) -> pd.DataFrame:
    """A four-name book that is built, held, rebalanced, then partly unwound."""
    data = [
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
    ]
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_turnover_returns_one_value_per_date(config, weight_path):
    result = turnover(config, weight_path)
    assert isinstance(result, pd.Series)
    assert len(result) == len(weight_path)


def test_turnover_is_labelled_by_the_weight_dates(config, weight_path):
    result = turnover(config, weight_path)
    assert list(result.index) == list(weight_path.index)


def test_turnover_known_values(config, weight_path):
    # Build 1.0 -> hold -> rotate 0.5 out of two names into the other two
    # (1.0 traded) -> hold -> liquidate 1.0 -> hold.
    result = turnover(config, weight_path)
    np.testing.assert_allclose(result.to_numpy(), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_turnover_charges_the_initial_build_from_cash(config, tickers, dates):
    weights = pd.DataFrame(
        [[0.4, 0.3, 0.2, 0.1]] * len(dates), index=dates, columns=tickers
    )
    result = turnover(config, weights)
    assert result.iloc[0] == pytest.approx(weights.iloc[0].abs().sum())


def test_turnover_is_zero_while_the_book_is_unchanged(config, tickers, dates):
    weights = pd.DataFrame(0.25, index=dates, columns=tickers)
    result = turnover(config, weights)
    assert result.iloc[1:].eq(0.0).all()


def test_turnover_is_zero_for_an_empty_book(config, tickers, dates):
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    assert turnover(config, weights).eq(0.0).all()


def test_turnover_counts_a_long_short_flip_in_full(config, tickers, dates):
    # Going from -0.5 to +0.5 trades 1.0, not 0.0.
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    weights.iloc[:3, 0] = -0.5
    weights.iloc[3:, 0] = 0.5
    result = turnover(config, weights)
    assert result.iloc[3] == pytest.approx(1.0)


def test_turnover_is_never_negative(config, weight_path):
    assert (turnover(config, weight_path) >= 0).all()


def test_turnover_ignores_the_direction_of_a_trade(config, tickers, dates):
    buying = pd.DataFrame(0.0, index=dates, columns=tickers)
    buying.iloc[3:, 0] = 0.2
    selling = pd.DataFrame(0.0, index=dates, columns=tickers)
    selling.iloc[:3, 0] = 0.2
    assert turnover(config, buying).iloc[3] == pytest.approx(
        turnover(config, selling).iloc[3]
    )


@pytest.mark.parametrize("factor", [2.0, 0.5, -1.0])
def test_turnover_scales_with_the_book(config, weight_path, factor):
    base = turnover(config, weight_path)
    scaled = turnover(config, weight_path * factor)
    np.testing.assert_allclose(scaled.to_numpy(), base.to_numpy() * abs(factor))


def test_turnover_is_invariant_to_column_order(config, weight_path):
    reordered = weight_path[list(reversed(weight_path.columns))]
    np.testing.assert_allclose(
        turnover(config, reordered).to_numpy(), turnover(config, weight_path).to_numpy()
    )


def test_turnover_single_row_is_just_the_initial_build(config, tickers, dates):
    weights = pd.DataFrame([[0.5, 0.5, 0.0, 0.0]], index=dates[:1], columns=tickers)
    result = turnover(config, weights)
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(1.0)


def test_turnover_does_not_mutate_its_input(config, weight_path):
    original = weight_path.copy(deep=True)
    turnover(config, weight_path)
    pd.testing.assert_frame_equal(weight_path, original)


def test_turnover_total_is_the_sum_of_each_rebalance(config, weight_path):
    assert turnover(config, weight_path).sum() == pytest.approx(3.0)


def test_turnover_of_random_weights_matches_a_manual_l1_diff(config, rng, tickers):
    index = pd.bdate_range("2023-01-02", periods=20)
    weights = pd.DataFrame(
        rng.uniform(-1, 1, size=(len(index), len(tickers))),
        index=index,
        columns=tickers,
    )
    expected = np.abs(np.diff(weights.to_numpy(), axis=0, prepend=0.0)).sum(axis=1)
    np.testing.assert_allclose(turnover(config, weights).to_numpy(), expected)


def test_turnover_keeps_a_usable_datetime_index(config, weight_path):
    result = turnover(config, weight_path)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.resample("W").sum().sum() == pytest.approx(result.sum())
