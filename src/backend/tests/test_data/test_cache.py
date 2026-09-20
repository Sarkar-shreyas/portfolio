"""Tests for ``src.backend.data.cache``.

Every write lands in the temp ``cache_dir`` carried by the ``config`` fixture.
"""

import json

import pandas as pd
import pytest

from src.backend.data.cache import save_df_as_csv, save_dict_as_df, save_dict_as_json


# ---------------------------------------------------------------------------
# save_df_as_csv()
# ---------------------------------------------------------------------------


def test_save_df_as_csv_creates_the_file(config, datetime_indexed_frame):
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    assert (config.cache_dir / "prices.csv").exists()


def test_save_df_as_csv_round_trips_values(config, datetime_indexed_frame):
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    loaded = pd.read_csv(config.cache_dir / "prices.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(loaded, datetime_indexed_frame)


def test_save_df_as_csv_writes_the_index(config, datetime_indexed_frame):
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    first_line = (config.cache_dir / "prices.csv").read_text().splitlines()[0]
    assert first_line.startswith(",") or first_line.split(",")[0] == ""


def test_save_df_as_csv_does_not_overwrite_existing_file(config, datetime_indexed_frame):
    target = config.cache_dir / "prices.csv"
    target.write_text("sentinel\n")
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    assert target.read_text() == "sentinel\n"


def test_save_df_as_csv_reports_when_skipping(config, datetime_indexed_frame, capsys):
    (config.cache_dir / "prices.csv").write_text("sentinel\n")
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    assert "File already exists." in capsys.readouterr().out


def test_save_df_as_csv_is_quiet_on_a_fresh_write(config, datetime_indexed_frame, capsys):
    save_df_as_csv(config, datetime_indexed_frame, "prices.csv")
    assert capsys.readouterr().out == ""


def test_save_df_as_csv_keeps_distinct_filenames_separate(config, datetime_indexed_frame):
    save_df_as_csv(config, datetime_indexed_frame, "a.csv")
    save_df_as_csv(config, datetime_indexed_frame, "b.csv")
    assert (config.cache_dir / "a.csv").exists()
    assert (config.cache_dir / "b.csv").exists()


def test_save_df_as_csv_handles_empty_frame(config):
    save_df_as_csv(config, pd.DataFrame(), "empty.csv")
    assert (config.cache_dir / "empty.csv").exists()


# ---------------------------------------------------------------------------
# save_dict_as_df()
# ---------------------------------------------------------------------------


def test_save_dict_as_df_creates_the_file(config, raw_bar_dict):
    save_dict_as_df(config, raw_bar_dict, "NVDA_data.csv")
    assert (config.cache_dir / "NVDA_data.csv").exists()


def test_save_dict_as_df_writes_cleaned_columns(config, raw_bar_dict):
    save_dict_as_df(config, raw_bar_dict, "NVDA_data.csv")
    loaded = pd.read_csv(config.cache_dir / "NVDA_data.csv")
    assert list(loaded.columns) == [
        "Date",
        "datetime",
        "open",
        "close",
        "high",
        "low",
        "volume",
    ]


def test_save_dict_as_df_writes_parsed_dates(config, raw_bar_dict):
    save_dict_as_df(config, raw_bar_dict, "NVDA_data.csv")
    loaded = pd.read_csv(
        config.cache_dir / "NVDA_data.csv", index_col="Date", parse_dates=True
    )
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index[0] == pd.Timestamp("2023-01-03")


def test_save_dict_as_df_preserves_values(config, raw_bar_dict):
    save_dict_as_df(config, raw_bar_dict, "NVDA_data.csv")
    loaded = pd.read_csv(
        config.cache_dir / "NVDA_data.csv", index_col="Date", parse_dates=True
    )
    assert loaded.loc["2023-01-04", "close"] == 103.5
    assert len(loaded) == 3


def test_save_dict_as_df_overwrites_existing_file(config, raw_bar_dict):
    target = config.cache_dir / "NVDA_data.csv"
    target.write_text("stale\n")
    save_dict_as_df(config, raw_bar_dict, "NVDA_data.csv")
    assert target.read_text() != "stale\n"


def test_save_dict_as_df_rejects_unparseable_keys(config, raw_bar_dict):
    bad = {"not-a-date": next(iter(raw_bar_dict.values()))}
    with pytest.raises(ValueError):
        save_dict_as_df(config, bad, "NVDA_data.csv")


# ---------------------------------------------------------------------------
# save_dict_as_json()
# ---------------------------------------------------------------------------


def test_save_dict_as_json_creates_the_file(config):
    save_dict_as_json(config, {"NetLiquidation": {"value": "150000.00"}}, "acc.json")
    assert (config.cache_dir / "acc.json").exists()


def test_save_dict_as_json_round_trips(config, sample_account_rows):
    payload = {
        tag: {"value": value, "currency": currency}
        for tag, (value, currency) in sample_account_rows.items()
    }
    save_dict_as_json(config, payload, "acc_summary.json")
    with open(config.cache_dir / "acc_summary.json") as file:
        assert json.load(file) == payload


def test_save_dict_as_json_handles_empty_dict(config):
    save_dict_as_json(config, {}, "empty.json")
    assert (config.cache_dir / "empty.json").read_text() == "{}"


def test_save_dict_as_json_overwrites_existing_file(config):
    target = config.cache_dir / "acc.json"
    target.write_text('{"stale": true}')
    save_dict_as_json(config, {"fresh": 1}, "acc.json")
    assert json.loads(target.read_text()) == {"fresh": 1}


def test_save_dict_as_json_rejects_non_serialisable_values(config):
    with pytest.raises(TypeError):
        save_dict_as_json(config, {"ts": pd.Timestamp("2023-01-03")}, "bad.json")
