import numpy as np
import pandas as pd
import pytest

from src.backend.portfolio_construction.static_weights import (
    equal_weight_benchmark,
    mkt_cap_weight_benchmark,
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
    expected = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"], name="weights")
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
    expected = pd.Series([0.25, 0.125, 0.1, 0.025], index=mkt_caps.index, name="weights")
    pd.testing.assert_series_equal(result, expected)


def test_mkt_cap_explicit_default_matches_implicit(config, mkt_caps):
    implicit = mkt_cap_weight_benchmark(config, mkt_caps)
    explicit = mkt_cap_weight_benchmark(config, mkt_caps, tot_exposure=config.tot_exposure)
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
