"""Tests for ``src.backend.data.clean``.

``clean_timeseries`` has two distinct branches, decided by whether the incoming
frame already carries a ``DatetimeIndex``:

* **Not** a ``DatetimeIndex`` — the index is treated as ``YYYYMMDD`` strings,
  promoted to a parsed ``Date`` index, and the frame is narrowed to the OHLCV
  columns (plus ``return`` when present).
* Already a ``DatetimeIndex`` — only column lowercasing and ``dropna`` happen;
  no column selection.
"""

import numpy as np
import pandas as pd
import pytest

from src.backend.data.clean import clean_timeseries

OHLCV = ["datetime", "open", "close", "high", "low", "volume"]


# ---------------------------------------------------------------------------
# clean_timeseries() - string-index branch
# ---------------------------------------------------------------------------


def test_promotes_string_index_to_named_datetime_index(raw_bar_frame):
    result = clean_timeseries(raw_bar_frame)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "Date"


def test_parses_yyyymmdd_index_values(raw_bar_frame):
    result = clean_timeseries(raw_bar_frame)
    expected = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    pd.testing.assert_index_equal(
        result.index, pd.DatetimeIndex(expected, name="Date")
    )


def test_selects_ohlcv_columns_in_fixed_order(raw_bar_frame):
    result = clean_timeseries(raw_bar_frame)
    assert list(result.columns) == OHLCV


def test_lowercases_column_names(raw_bar_dict):
    upper = {
        key: {field.upper(): value for field, value in row.items()}
        for key, row in raw_bar_dict.items()
    }
    result = clean_timeseries(pd.DataFrame.from_dict(upper, orient="index"))
    assert list(result.columns) == OHLCV


def test_drops_columns_outside_the_ohlcv_set(raw_bar_frame):
    extra = raw_bar_frame.copy()
    extra["wap"] = 1.0
    extra["barCount"] = 5
    result = clean_timeseries(extra)
    assert list(result.columns) == OHLCV


def test_preserves_values(raw_bar_frame):
    result = clean_timeseries(raw_bar_frame)
    assert result.loc["2023-01-04", "close"] == 103.5
    assert result.loc["2023-01-05", "volume"] == 900
    assert len(result) == 3


def test_keeps_return_column_when_present(raw_bar_frame):
    with_return = raw_bar_frame.copy()
    with_return["return"] = [np.nan, 0.0248, -0.0097]
    result = clean_timeseries(with_return)
    assert list(result.columns) == OHLCV + ["return"]


def test_drops_rows_with_missing_return(raw_bar_frame):
    with_return = raw_bar_frame.copy()
    with_return["return"] = [np.nan, 0.0248, -0.0097]
    result = clean_timeseries(with_return)
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp("2023-01-04")


def test_missing_return_column_does_not_trigger_return_branch(raw_bar_frame):
    # NaNs elsewhere survive when there is no ``return`` column to filter on.
    frame = raw_bar_frame.copy()
    frame.loc["20230104", "volume"] = np.nan
    result = clean_timeseries(frame)
    assert len(result) == 3
    assert np.isnan(result.loc["2023-01-04", "volume"])


def test_raises_when_index_is_not_yyyymmdd(raw_bar_frame):
    frame = raw_bar_frame.copy()
    frame.index = ["2023-01-03", "2023-01-04", "2023-01-05"]
    with pytest.raises(ValueError):
        clean_timeseries(frame)


def test_raises_when_ohlcv_columns_are_missing(raw_bar_frame):
    with pytest.raises(KeyError):
        clean_timeseries(raw_bar_frame.drop(columns=["volume"]))


# ---------------------------------------------------------------------------
# clean_timeseries() - DatetimeIndex branch
# ---------------------------------------------------------------------------


def test_datetime_index_is_left_untouched(datetime_indexed_frame):
    result = clean_timeseries(datetime_indexed_frame)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name is None


def test_datetime_index_branch_lowercases_columns(datetime_indexed_frame):
    result = clean_timeseries(datetime_indexed_frame)
    assert list(result.columns) == ["open", "close", "volume"]


def test_datetime_index_branch_drops_na_rows(datetime_indexed_frame):
    result = clean_timeseries(datetime_indexed_frame)
    assert len(result) == 2
    assert pd.Timestamp("2023-01-04") not in result.index


def test_datetime_index_branch_keeps_non_ohlcv_columns():
    frame = pd.DataFrame(
        {"Close": [1.0, 2.0], "Beta": [0.9, 1.1]},
        index=pd.to_datetime(["2023-01-03", "2023-01-04"]),
    )
    result = clean_timeseries(frame)
    assert list(result.columns) == ["close", "beta"]


def test_datetime_index_branch_is_idempotent(datetime_indexed_frame):
    once = clean_timeseries(datetime_indexed_frame)
    twice = clean_timeseries(once)
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------------------------------
# clean_timeseries() - copy semantics
# ---------------------------------------------------------------------------


def test_does_not_mutate_input_frame(raw_bar_frame):
    before = raw_bar_frame.copy(deep=True)
    clean_timeseries(raw_bar_frame)
    pd.testing.assert_frame_equal(raw_bar_frame, before)


def test_does_not_mutate_datetime_indexed_input(datetime_indexed_frame):
    before = datetime_indexed_frame.copy(deep=True)
    clean_timeseries(datetime_indexed_frame)
    pd.testing.assert_frame_equal(datetime_indexed_frame, before)


def test_returns_a_new_object(datetime_indexed_frame):
    assert clean_timeseries(datetime_indexed_frame) is not datetime_indexed_frame
