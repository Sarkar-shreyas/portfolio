import numpy as np
import pandas as pd
import pytest

from src.backend.analysis.returns import simple_rsi, ewm_rsi
from src.backend.strategy.mean_reversion import rsi_mean_reversion


def _assert_signal_values(signal: pd.Series):
    """A signal may only ever contain -1, 0 or 1 (never NaN)."""
    assert not signal.isnull().any()
    assert set(np.unique(signal.to_numpy())) <= {-1, 0, 1}


def _expected_signal(rsi: pd.Series, overbought: float, oversold: float) -> pd.Series:
    """Reference implementation: 1 if rsi < oversold, -1 if rsi > overbought, else 0."""
    values = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
    return pd.Series(values, index=rsi.index, name="signal")


# ---------------------------------------------------------------------------
# Output shape / metadata
# ---------------------------------------------------------------------------


def test_returns_series_named_signal(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series)
    assert isinstance(result, pd.Series)
    assert result.name == "signal"


def test_index_preserved(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series)
    pd.testing.assert_index_equal(result.index, rsi_series.index)


def test_only_valid_signal_values(config, rsi_series):
    _assert_signal_values(rsi_mean_reversion(config, rsi_series))


def test_input_not_mutated(config, rsi_series):
    original = rsi_series.copy(deep=True)
    rsi_mean_reversion(config, rsi_series)
    pd.testing.assert_series_equal(rsi_series, original)


def test_works_with_range_index(config):
    """The signal must not depend on the index being a DatetimeIndex."""
    rsi = pd.Series([25.0, 50.0, 75.0])
    result = rsi_mean_reversion(config, rsi)
    expected = pd.Series([1, 0, -1], name="signal")
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# Signal semantics with config defaults
# ---------------------------------------------------------------------------


def test_full_expected_signal(config, rsi_series, expected_rsi_signal):
    result = rsi_mean_reversion(config, rsi_series)
    pd.testing.assert_series_equal(result, expected_rsi_signal, check_dtype=False)


def test_long_when_oversold(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series)
    mask = rsi_series < config.rsi_oversold
    assert mask.any()
    assert (result[mask] == 1).all()


def test_short_when_overbought(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series)
    mask = rsi_series > config.rsi_overbought
    assert mask.any()
    assert (result[mask] == -1).all()


def test_flat_in_neutral_zone(config, rsi_series):
    """Neutral RSI rows must be 0 - the raw RSI level must not leak through."""
    result = rsi_mean_reversion(config, rsi_series)
    mask = (rsi_series >= config.rsi_oversold) & (rsi_series <= config.rsi_overbought)
    assert mask.any()
    assert (result[mask] == 0).all()


def test_flat_on_nan(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series)
    mask = rsi_series.isnull()
    assert mask.any()
    assert (result[mask] == 0).all()


def test_thresholds_are_strict(config):
    """RSI exactly on a threshold is neither oversold nor overbought."""
    rsi = pd.Series([float(config.rsi_oversold), float(config.rsi_overbought)])
    result = rsi_mean_reversion(config, rsi)
    assert (result == 0).all()


def test_all_neutral_gives_all_zero(config):
    rsi = pd.Series(np.linspace(config.rsi_oversold, config.rsi_overbought, 20))
    result = rsi_mean_reversion(config, rsi)
    assert (result == 0).all()
    _assert_signal_values(result)


def test_all_nan_gives_all_zero(config):
    rsi = pd.Series([np.nan] * 5)
    result = rsi_mean_reversion(config, rsi)
    assert (result == 0).all()


def test_all_oversold_gives_all_long(config):
    rsi = pd.Series([5.0, 10.0, 15.0, 29.9])
    result = rsi_mean_reversion(config, rsi)
    assert (result == 1).all()


def test_all_overbought_gives_all_short(config):
    rsi = pd.Series([70.1, 80.0, 95.0, 100.0])
    result = rsi_mean_reversion(config, rsi)
    assert (result == -1).all()


# ---------------------------------------------------------------------------
# Threshold overrides
# ---------------------------------------------------------------------------


def test_defaults_come_from_config(rsi_series):
    """Passing the config thresholds explicitly must equal omitting them."""
    from src.backend.config import TestConfig

    config = TestConfig()
    implicit = rsi_mean_reversion(config, rsi_series)
    explicit = rsi_mean_reversion(
        config,
        rsi_series,
        rsi_overbought=config.rsi_overbought,
        rsi_oversold=config.rsi_oversold,
    )
    pd.testing.assert_series_equal(implicit, explicit)


def test_custom_thresholds_override_config(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series, rsi_overbought=80, rsi_oversold=20)
    expected = _expected_signal(rsi_series, overbought=80, oversold=20)
    pd.testing.assert_series_equal(result, expected, check_dtype=False)
    # 25.0 is no longer oversold and 75.0 is no longer overbought.
    assert result.iloc[1] == 0
    assert result.iloc[3] == 0


def test_only_overbought_override(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series, rsi_overbought=60)
    expected = _expected_signal(
        rsi_series, overbought=60, oversold=config.rsi_oversold
    )
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_only_oversold_override(config, rsi_series):
    result = rsi_mean_reversion(config, rsi_series, rsi_oversold=40)
    expected = _expected_signal(
        rsi_series, overbought=config.rsi_overbought, oversold=40
    )
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_wider_band_never_increases_active_signals(config, rsi_series):
    narrow = rsi_mean_reversion(config, rsi_series, rsi_overbought=60, rsi_oversold=40)
    wide = rsi_mean_reversion(config, rsi_series, rsi_overbought=90, rsi_oversold=10)
    assert wide.abs().sum() <= narrow.abs().sum()
    # Anything active under the wide band is also active under the narrow band.
    assert ((wide != 0) <= (narrow != 0)).all()


# ---------------------------------------------------------------------------
# Integration with the analysis helpers that produce the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rsi_func", [simple_rsi, ewm_rsi], ids=["simple", "ewm"])
def test_signal_from_analysis_rsi(rsi_func, config, returns_series):
    rsi = rsi_func(config, returns_series)
    assert rsi.between(0, 100).all()

    result = rsi_mean_reversion(config, rsi)

    expected = _expected_signal(rsi, config.rsi_overbought, config.rsi_oversold)
    pd.testing.assert_series_equal(result, expected, check_dtype=False)
    _assert_signal_values(result)
    pd.testing.assert_index_equal(result.index, rsi.index)
