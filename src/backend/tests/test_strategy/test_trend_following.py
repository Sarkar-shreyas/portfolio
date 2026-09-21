import numpy as np
import pandas as pd
import pytest

from src.backend.analysis.returns import sma_returns, ema_returns
from src.backend.strategy.trend_following import sma_crossover, ema_crossover


# Both crossover functions share identical semantics and differ only in the
# column names they read, so most tests are parametrised over the pair.
CROSSOVERS = [
    pytest.param(sma_crossover, "sma_frame", id="sma"),
    pytest.param(ema_crossover, "ema_frame", id="ema"),
]

PREFIXES = [
    pytest.param(sma_crossover, "sma", id="sma"),
    pytest.param(ema_crossover, "ema", id="ema"),
]


def _assert_signal_values(signal: pd.Series):
    """A signal may only ever contain -1, 0 or 1 (never NaN)."""
    assert not signal.isnull().any()
    assert set(np.unique(signal.to_numpy())) <= {-1, 0, 1}


def _expected_signal(frame: pd.DataFrame) -> pd.Series:
    """Reference implementation: 1 if short > long, -1 if short < long, else 0."""
    short, long = frame.iloc[:, 0], frame.iloc[:, 1]
    values = np.where(short > long, 1, np.where(short < long, -1, 0))
    return pd.Series(values, index=frame.index, name="signal")


# ---------------------------------------------------------------------------
# Output shape / metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_returns_series_named_signal(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    result = func(config, frame)
    assert isinstance(result, pd.Series)
    assert result.name == "signal"


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_index_preserved(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    result = func(config, frame)
    pd.testing.assert_index_equal(result.index, frame.index)


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_only_valid_signal_values(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    _assert_signal_values(func(config, frame))


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_input_not_mutated(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    original = frame.copy(deep=True)
    func(config, frame)
    pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# Signal semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_full_expected_signal(
    func, frame_fixture, config, expected_crossover_signal, request
):
    frame = request.getfixturevalue(frame_fixture)
    result = func(config, frame)
    pd.testing.assert_series_equal(
        result, expected_crossover_signal, check_dtype=False
    )


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_long_where_short_above_long(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    short, long = frame.iloc[:, 0], frame.iloc[:, 1]
    result = func(config, frame)
    mask = short > long
    assert mask.any()
    assert (result[mask] == 1).all()


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_short_where_short_below_long(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    short, long = frame.iloc[:, 0], frame.iloc[:, 1]
    result = func(config, frame)
    mask = short < long
    assert mask.any()
    assert (result[mask] == -1).all()


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_flat_where_equal(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    short, long = frame.iloc[:, 0], frame.iloc[:, 1]
    result = func(config, frame)
    mask = short == long
    assert mask.any()
    assert (result[mask] == 0).all()


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_flat_during_warmup_nans(func, frame_fixture, config, request):
    """Rows where either average is NaN carry no information -> signal 0.

    In particular the raw moving-average level must never leak into the
    signal series.
    """
    frame = request.getfixturevalue(frame_fixture)
    result = func(config, frame)
    mask = frame.isnull().any(axis=1)
    assert mask.any()
    assert (result[mask] == 0).all()


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_all_nan_long_gives_all_zero(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture).copy()
    frame.iloc[:, 1] = np.nan
    result = func(config, frame)
    assert (result == 0).all()


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_signal_is_antisymmetric_under_column_swap(
    func, frame_fixture, config, request
):
    """Swapping the short and long columns should exactly negate the signal."""
    frame = request.getfixturevalue(frame_fixture)
    swapped = frame.copy()
    swapped.iloc[:, 0] = frame.iloc[:, 1].to_numpy()
    swapped.iloc[:, 1] = frame.iloc[:, 0].to_numpy()
    result = func(config, frame)
    result_swapped = func(config, swapped)
    pd.testing.assert_series_equal(result_swapped, -result, check_dtype=False)


@pytest.mark.parametrize("func, frame_fixture", CROSSOVERS)
def test_extra_columns_are_ignored(func, frame_fixture, config, request):
    frame = request.getfixturevalue(frame_fixture)
    with_extra = frame.copy()
    with_extra["close"] = 1e6
    with_extra["noise"] = -1e6
    pd.testing.assert_series_equal(func(config, with_extra), func(config, frame))


def test_sma_and_ema_agree_on_identical_inputs(config, sma_frame, ema_frame):
    """The two functions implement the same rule under different column names."""
    sma_sig = sma_crossover(config, sma_frame)
    ema_sig = ema_crossover(config, ema_frame)
    pd.testing.assert_series_equal(sma_sig, ema_sig)


# ---------------------------------------------------------------------------
# Trending inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func, prefix", PREFIXES)
def test_monotonic_uptrend_is_always_long(func, prefix, config):
    """In a clean uptrend the short average sits above the long one."""
    idx = pd.bdate_range("2024-01-01", periods=50)
    prices = pd.Series(np.linspace(100, 200, 50), index=idx)
    frame = pd.DataFrame(
        {f"{prefix}_short": prices.rolling(5).mean(), f"{prefix}_long": prices.rolling(20).mean()}
    )
    result = func(config, frame)
    valid = frame.notnull().all(axis=1)
    assert (result[valid] == 1).all()
    assert (result[~valid] == 0).all()


@pytest.mark.parametrize("func, prefix", PREFIXES)
def test_monotonic_downtrend_is_always_short(func, prefix, config):
    idx = pd.bdate_range("2024-01-01", periods=50)
    prices = pd.Series(np.linspace(200, 100, 50), index=idx)
    frame = pd.DataFrame(
        {f"{prefix}_short": prices.rolling(5).mean(), f"{prefix}_long": prices.rolling(20).mean()}
    )
    result = func(config, frame)
    valid = frame.notnull().all(axis=1)
    assert (result[valid] == -1).all()
    assert (result[~valid] == 0).all()


@pytest.mark.parametrize("func, prefix", PREFIXES)
def test_random_walk_matches_reference(func, prefix, config, price_series):
    frame = pd.DataFrame(
        {
            f"{prefix}_short": price_series.rolling(5).mean(),
            f"{prefix}_long": price_series.rolling(21).mean(),
        }
    )
    result = func(config, frame)
    pd.testing.assert_series_equal(result, _expected_signal(frame), check_dtype=False)
    # Both regimes should appear in a year of random-walk data.
    assert (result == 1).any() and (result == -1).any()


# ---------------------------------------------------------------------------
# Integration with the analysis helpers that produce the inputs
# ---------------------------------------------------------------------------


def test_sma_crossover_from_analysis_output(config, price_series):
    """Signals generated from ``sma_returns`` output match the reference rule.

    ``sma_returns`` drops NaNs, so after concatenation the long average has
    a leading NaN block relative to the short one - that block must be 0.
    """
    short = sma_returns(config, price_series, period=5)
    long = sma_returns(config, price_series, period=21)
    frame = pd.concat({"sma_short": short, "sma_long": long}, axis=1)
    assert frame["sma_long"].isnull().any()

    result = sma_crossover(config, frame)
    pd.testing.assert_series_equal(result, _expected_signal(frame), check_dtype=False)
    _assert_signal_values(result)


def test_ema_crossover_from_analysis_output(config, price_series):
    short = ema_returns(config, price_series, span=5)
    long = ema_returns(config, price_series, span=21)
    frame = pd.concat({"ema_short": short, "ema_long": long}, axis=1)

    result = ema_crossover(config, frame)
    pd.testing.assert_series_equal(result, _expected_signal(frame), check_dtype=False)
    _assert_signal_values(result)
