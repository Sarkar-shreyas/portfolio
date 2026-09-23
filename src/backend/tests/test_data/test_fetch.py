"""Tests for ``src.backend.data.fetch``.

No test opens a socket. The IBKR side is driven by ``FakeIBApp`` (see
``helpers``), the AlphaVantage side by a stubbed ``requests.get``, and the
``time`` module inside ``fetch`` is swapped for a non-sleeping fake by the
autouse ``fake_time`` fixture.
"""

import json

import pandas as pd
import pytest

from src.backend.data import fetch as fetch_module
from src.backend.data.fetch import (
    ACCOUNT_SUMMARY_TAGS,
    get_account_summary,
    get_benchmark_data,
    get_equity_data,
    get_fama_factors,
    get_market_cap,
    get_portfolio_data,
    read_json,
)
from src.backend.tests.test_data.helpers import make_contract, make_contract_details

SPX_CON_ID = 416904


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_requests(monkeypatch):
    """Replace ``fetch.requests.get`` with a recorder returning canned payloads.

    The returned object exposes ``urls`` (every URL requested, in order) and
    ``payloads`` (symbol -> JSON body, with ``default`` as the fallback).
    """

    class _Recorder:
        def __init__(self):
            self.urls: list[str] = []
            self.payloads: dict = {}
            self.default = {"MarketCapitalization": "3000000000000"}
            self.error: Exception | None = None

        def get(self, url):
            self.urls.append(url)
            if self.error is not None:
                raise self.error
            symbol = url.split("symbol=")[1].split("&")[0]
            return _FakeResponse(self.payloads.get(symbol, self.default))

    recorder = _Recorder()
    monkeypatch.setattr(fetch_module.requests, "get", recorder.get)
    return recorder


# ---------------------------------------------------------------------------
# read_json()
# ---------------------------------------------------------------------------


def test_read_json_returns_file_contents(tmp_path):
    path = tmp_path / "acc.json"
    payload = {"NetLiquidation": {"value": "150000.00", "currency": "USD"}}
    path.write_text(json.dumps(payload))
    assert read_json(str(path)) == payload


def test_read_json_handles_empty_object(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    assert read_json(str(path)) == {}


def test_read_json_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(str(tmp_path / "nope.json"))


def test_read_json_raises_for_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        read_json(str(path))


# ---------------------------------------------------------------------------
# get_equity_data()
# ---------------------------------------------------------------------------


def test_get_equity_data_returns_cached_csv_without_connecting(dev_config, app):
    # ISO date strings rather than YYYYMMDD, so the CSV round trip is lossless
    # and the comparison is not really testing read_csv's dtype inference.
    cached = pd.DataFrame({"datetime": ["2023-01-03"], "close": [101.0]})
    cached.to_csv(dev_config.cache_dir / "NVDA_data.csv", index=False)

    result = get_equity_data(app, dev_config, conid=4815747, ticker="NVDA")

    pd.testing.assert_frame_equal(result, cached)
    assert app.calls == []


def test_get_equity_data_announces_a_cache_hit(dev_config, app, capsys):
    pd.DataFrame({"close": [101.0]}).to_csv(dev_config.cache_dir / "NVDA_data.csv")
    get_equity_data(app, dev_config, conid=4815747, ticker="NVDA")
    assert "Loading existing NVDA data." in capsys.readouterr().out


def test_get_equity_data_cache_lookup_is_per_ticker(dev_config, app):
    pd.DataFrame({"close": [1.0]}).to_csv(dev_config.cache_dir / "AMD_data.csv")
    get_equity_data(app, dev_config, conid=4815747, ticker="NVDA")
    # The AMD cache must not satisfy an NVDA request.
    assert "reqHistoricalData" in app.call_names()


def test_get_equity_data_builds_frame_from_bars(dev_config, app):
    result = get_equity_data(app, dev_config, conid=4815747, ticker="NVDA")
    assert list(result.columns) == [
        "datetime",
        "open",
        "close",
        "high",
        "low",
        "volume",
    ]
    assert len(result) == 3
    assert result["close"].tolist() == [101.0, 103.5, 102.5]


def test_get_equity_data_connects_with_config_credentials(dev_config, app):
    get_equity_data(app, dev_config, conid=4815747)
    assert app.call_args("connect") == {
        "host": dev_config.IB_HOST,
        "port": dev_config.IB_PORT,
        "clientId": dev_config.IB_CLIENT_ID,
    }


def test_get_equity_data_requests_contract_details_for_the_conid(dev_config, app):
    get_equity_data(app, dev_config, conid=4815747, reqId=3)
    args = app.call_args("reqContractDetails")
    assert args["reqId"] == 1003  # reqId + 1000
    assert args["contract"].conId == 4815747


def test_get_equity_data_sends_expected_historical_request(dev_config, app):
    get_equity_data(app, dev_config, conid=4815747, dur="2 Y", reqId=5)
    args = app.call_args("reqHistoricalData")
    assert args["reqId"] == 5
    assert args["durationStr"] == "2 Y"
    assert args["barSizeSetting"] == "1 day"
    assert args["whatToShow"] == "ADJUSTED_LAST"
    assert args["endDateTime"] == ""
    assert args["useRTH"] == 1
    assert args["formatDate"] == 1
    assert args["keepUpToDate"] is False


def test_get_equity_data_uses_the_resolved_contract(dev_config, app_factory, sample_bars):
    resolved = make_contract(con_id=4815747, symbol="NVDA")
    app = app_factory(bars=sample_bars, contract_details=make_contract_details(resolved))
    get_equity_data(app, dev_config, conid=4815747)
    assert app.call_args("reqHistoricalData")["contract"] is resolved


def test_get_equity_data_resets_the_finished_flag(dev_config, app):
    app.hd_finished[1] = True
    get_equity_data(app, dev_config, conid=4815747, reqId=1)
    # Reset to False at entry, then set back to True by historicalDataEnd.
    assert app.hd_finished[1] is True
    assert app.cd_finished[1001] is True


def test_get_equity_data_disconnects_after_success(dev_config, app):
    get_equity_data(app, dev_config, conid=4815747)
    assert "disconnect" in app.call_names()
    assert app.connected is False


def test_get_equity_data_call_order(dev_config, app):
    get_equity_data(app, dev_config, conid=4815747)
    names = app.call_names()
    assert names.index("connect") < names.index("reqContractDetails")
    assert names.index("reqContractDetails") < names.index("reqHistoricalData")
    assert names.index("reqHistoricalData") < names.index("disconnect")


def test_get_equity_data_raises_when_connection_fails(dev_config, app_factory):
    app = app_factory(connect_error=ConnectionRefusedError("gateway down"))
    with pytest.raises(RuntimeError, match="Could not connect to IB Gateway"):
        get_equity_data(app, dev_config, conid=4815747)
    assert "reqContractDetails" not in app.call_names()


def test_get_equity_data_raises_when_contract_unresolved(dev_config, app_factory, sample_bars):
    app = app_factory(
        bars=sample_bars, contract_details=make_contract_details(contract=None)
    )
    with pytest.raises(RuntimeError, match="Could not retrieve contract"):
        get_equity_data(app, dev_config, conid=999)
    assert "disconnect" in app.call_names()
    assert "reqHistoricalData" not in app.call_names()


def test_get_equity_data_raises_when_no_contract_details_arrive(dev_config, app_factory):
    # contractDetailsEnd fires without a preceding contractDetails callback, so
    # the stored entry is still None and attribute access blows up.
    app = app_factory(send_contract_details=False)
    with pytest.raises(AttributeError):
        get_equity_data(app, dev_config, conid=4815747)


def test_get_equity_data_returns_empty_frame_when_no_bars(dev_config, app_factory):
    # historicalData never fires for an empty result set, so app.data has no
    # entry for reqId; the ``.get(reqId, [])`` fallback yields an empty frame
    # rather than raising.
    app = app_factory(bars=[])
    result = get_equity_data(app, dev_config, conid=4815747, reqId=1)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_equity_data_still_disconnects_when_no_bars(dev_config, app_factory):
    app = app_factory(bars=[])
    get_equity_data(app, dev_config, conid=4815747, reqId=1)
    assert "disconnect" in app.call_names()
    assert app.connected is False


# ---------------------------------------------------------------------------
# get_account_summary()
# ---------------------------------------------------------------------------


def test_get_account_summary_returns_cached_json_without_connecting(dev_config, app):
    payload = {"NetLiquidation": {"value": "150000.00", "currency": "USD"}}
    (dev_config.cache_dir / "acc_summary.json").write_text(json.dumps(payload))

    result = get_account_summary(app, dev_config)

    assert result == payload
    assert app.calls == []


def test_get_account_summary_returns_collected_tags(dev_config, app, sample_account_rows):
    result = get_account_summary(app, dev_config)
    assert set(result) == set(sample_account_rows)
    assert result["NetLiquidation"] == {"value": "150000.00", "currency": "USD"}


def test_get_account_summary_connects_with_config_credentials(dev_config, app):
    get_account_summary(app, dev_config)
    assert app.call_args("connect") == {
        "host": dev_config.IB_HOST,
        "port": dev_config.IB_PORT,
        "clientId": dev_config.IB_CLIENT_ID,
    }


def test_get_account_summary_requests_the_module_tag_list(dev_config, app):
    get_account_summary(app, dev_config, name="All", reqId=4)
    args = app.call_args("reqAccountSummary")
    assert args["reqId"] == 4
    assert args["groupName"] == "All"
    assert args["tags"] == ACCOUNT_SUMMARY_TAGS


def test_get_account_summary_passes_through_group_name(dev_config, app):
    get_account_summary(app, dev_config, name="DU1234567")
    assert app.call_args("reqAccountSummary")["groupName"] == "DU1234567"


def test_get_account_summary_sets_finished_flag(dev_config, app):
    get_account_summary(app, dev_config, reqId=2)
    assert app.ad_finished[2] is True


def test_get_account_summary_disconnects_after_success(dev_config, app):
    get_account_summary(app, dev_config)
    assert "disconnect" in app.call_names()
    assert app.connected is False


def test_get_account_summary_returns_the_live_app_dict(dev_config, app):
    result = get_account_summary(app, dev_config)
    assert result is app.account


def test_get_account_summary_raises_when_connection_fails(dev_config, app_factory):
    app = app_factory(connect_error=ConnectionRefusedError("gateway down"))
    with pytest.raises(RuntimeError, match="Could not connect to IB Gateway"):
        get_account_summary(app, dev_config)
    assert "reqAccountSummary" not in app.call_names()


def test_get_account_summary_returns_empty_when_no_tags(dev_config, app_factory):
    app = app_factory(account_rows={})
    assert get_account_summary(app, dev_config) == {}


# ---------------------------------------------------------------------------
# get_portfolio_data()
# ---------------------------------------------------------------------------


def test_get_portfolio_data_returns_cached_csv_without_connecting(dev_config, app):
    cached = pd.DataFrame(
        {"ticker": ["NVDA", "AAPL"], "position": [100.0, 50.0], "id": [4815747, 265598]}
    )
    cached.to_csv(dev_config.cache_dir / "portfolio.csv", index=False)

    result = get_portfolio_data(app, dev_config)

    assert set(result) == {"NVDA", "AAPL"}
    assert result["NVDA"]["position"] == 100.0
    assert app.calls == []


def test_get_portfolio_data_returns_collected_positions(dev_config, app):
    result = get_portfolio_data(app, dev_config)
    assert set(result) == {"NVDA", "AAPL"}
    assert result["NVDA"]["position"] == 100.0
    assert result["AAPL"]["marketPrice"] == 190.0


def test_get_portfolio_data_maps_contract_metadata(dev_config, app):
    result = get_portfolio_data(app, dev_config)
    assert result["AAPL"]["id"] == 265598
    assert result["AAPL"]["secType"] == "STK"
    assert result["AAPL"]["currency"] == "USD"


def test_get_portfolio_data_connects_with_config_credentials(dev_config, app):
    get_portfolio_data(app, dev_config)
    assert app.call_args("connect") == {
        "host": dev_config.IB_HOST,
        "port": dev_config.IB_PORT,
        "clientId": dev_config.IB_CLIENT_ID,
    }


def test_get_portfolio_data_subscribes_then_unsubscribes(dev_config, app):
    get_portfolio_data(app, dev_config)
    subscriptions = [
        kwargs["subscribe"]
        for name, kwargs in app.calls
        if name == "reqAccountUpdates"
    ]
    assert subscriptions == [True, False]


def test_get_portfolio_data_unsubscribes_before_disconnecting(dev_config, app):
    get_portfolio_data(app, dev_config)
    names = app.call_names()
    assert names.index("reqAccountUpdates") < names.index("disconnect")
    assert names[-1] == "disconnect"


def test_get_portfolio_data_sets_finished_flag(dev_config, app):
    get_portfolio_data(app, dev_config)
    assert app.pd_finished is True


def test_get_portfolio_data_returns_the_live_app_dict(dev_config, app):
    assert get_portfolio_data(app, dev_config) is app.positions


def test_get_portfolio_data_raises_when_connection_fails(dev_config, app_factory):
    app = app_factory(connect_error=ConnectionRefusedError("gateway down"))
    with pytest.raises(RuntimeError, match="Could not connect to IB Gateway"):
        get_portfolio_data(app, dev_config)
    assert "reqAccountUpdates" not in app.call_names()


def test_get_portfolio_data_returns_empty_for_flat_account(dev_config, app_factory):
    app = app_factory(portfolio_rows=[])
    assert get_portfolio_data(app, dev_config) == {}


# ---------------------------------------------------------------------------
# get_market_cap()
# ---------------------------------------------------------------------------


def test_get_market_cap_returns_empty_for_no_symbols(dev_config, fake_requests):
    assert get_market_cap(dev_config, []) == {}
    assert fake_requests.urls == []


def test_get_market_cap_defaults_to_no_symbols(dev_config, fake_requests):
    assert get_market_cap(dev_config) == {}


def test_get_market_cap_raises_without_an_api_key(dev_config, fake_requests):
    dev_config.ALPHA_VANTAGE_KEY = ""
    with pytest.raises(ValueError, match="Could not retrieve API key"):
        get_market_cap(dev_config, ["NVDA"])


def test_get_market_cap_empty_symbols_short_circuits_key_check(dev_config):
    dev_config.ALPHA_VANTAGE_KEY = ""
    assert get_market_cap(dev_config, []) == {}


def test_get_market_cap_returns_int_market_caps(dev_config, fake_requests):
    fake_requests.payloads = {
        "NVDA": {"MarketCapitalization": "3000000000000"},
        "AAPL": {"MarketCapitalization": "2800000000000"},
    }
    result = get_market_cap(dev_config, ["NVDA", "AAPL"])
    assert result == {"NVDA": 3_000_000_000_000, "AAPL": 2_800_000_000_000}
    assert all(isinstance(value, int) for value in result.values())


def test_get_market_cap_normalises_symbol_keys(dev_config, fake_requests):
    fake_requests.payloads = {" nvda ": {"MarketCapitalization": "100"}}
    assert get_market_cap(dev_config, [" nvda "]) == {"NVDA": 100}


def test_get_market_cap_includes_key_and_function_in_url(dev_config, fake_requests):
    get_market_cap(dev_config, ["NVDA"])
    url = fake_requests.urls[0]
    assert "function=OVERVIEW" in url
    assert "symbol=NVDA" in url
    assert f"apikey={dev_config.ALPHA_VANTAGE_KEY}" in url


def test_get_market_cap_requests_each_symbol_once(dev_config, fake_requests):
    get_market_cap(dev_config, ["NVDA", "AAPL", "AMD"])
    assert len(fake_requests.urls) == 3


def test_get_market_cap_throttles_between_requests(dev_config, fake_requests, fake_time):
    get_market_cap(dev_config, ["NVDA", "AAPL"])
    assert fake_time.slept == [1, 1]


def test_get_market_cap_raises_when_the_request_fails(dev_config, fake_requests):
    fake_requests.error = RuntimeError("network down")
    with pytest.raises(RuntimeError, match="Error retrieving info for NVDA"):
        get_market_cap(dev_config, ["NVDA"])


def test_get_market_cap_raises_on_missing_market_cap_field(dev_config, fake_requests):
    fake_requests.payloads = {"NVDA": {"Note": "rate limit reached"}}
    with pytest.raises(RuntimeError, match="Error retrieving info for NVDA"):
        get_market_cap(dev_config, ["NVDA"])


def test_get_market_cap_raises_on_non_numeric_market_cap(dev_config, fake_requests):
    fake_requests.payloads = {"NVDA": {"MarketCapitalization": "None"}}
    with pytest.raises(RuntimeError, match="Error retrieving info for NVDA"):
        get_market_cap(dev_config, ["NVDA"])


# ---------------------------------------------------------------------------
# get_benchmark_data()
# ---------------------------------------------------------------------------


def test_get_benchmark_data_returns_cached_csv_without_connecting(dev_config, app):
    cached = pd.DataFrame({"datetime": ["2023-01-03"], "close": [3850.0]})
    cached.to_csv(dev_config.cache_dir / "SPX_data.csv", index=False)

    result = get_benchmark_data(app, dev_config, reqId=1)

    pd.testing.assert_frame_equal(result, cached)
    assert app.calls == []


def test_get_benchmark_data_returns_empty_frame_when_no_bars(dev_config, app_factory):
    # Same fallback as get_equity_data: no bars means no app.data entry for
    # reqId, and ``.get(reqId, [])`` yields an empty frame.
    app = app_factory(bars=[])
    result = get_benchmark_data(app, dev_config, reqId=1)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_benchmark_data_builds_frame_from_bars(dev_config, app):
    result = get_benchmark_data(app, dev_config, reqId=1)
    assert list(result.columns) == [
        "datetime",
        "open",
        "close",
        "high",
        "low",
        "volume",
    ]
    assert len(result) == 3


def test_get_benchmark_data_requests_the_spx_conid(dev_config, app):
    get_benchmark_data(app, dev_config, reqId=2)
    args = app.call_args("reqContractDetails")
    assert args["reqId"] == 1002
    assert args["contract"].conId == SPX_CON_ID


def test_get_benchmark_data_requests_five_years_of_daily_bars(dev_config, app):
    get_benchmark_data(app, dev_config, reqId=2)
    args = app.call_args("reqHistoricalData")
    assert args["reqId"] == 2
    assert args["durationStr"] == "5 Y"
    assert args["barSizeSetting"] == "1 day"
    assert args["whatToShow"] == "ADJUSTED_LAST"


def test_get_benchmark_data_connects_with_config_credentials(dev_config, app):
    get_benchmark_data(app, dev_config, reqId=1)
    assert app.call_args("connect") == {
        "host": dev_config.IB_HOST,
        "port": dev_config.IB_PORT,
        "clientId": dev_config.IB_CLIENT_ID,
    }


def test_get_benchmark_data_disconnects_after_success(dev_config, app):
    get_benchmark_data(app, dev_config, reqId=1)
    assert "disconnect" in app.call_names()
    assert app.connected is False


def test_get_benchmark_data_raises_when_connection_fails(dev_config, app_factory):
    app = app_factory(connect_error=ConnectionRefusedError("gateway down"))
    with pytest.raises(RuntimeError, match="Could not connect to IB Gateway"):
        get_benchmark_data(app, dev_config, reqId=1)


def test_get_benchmark_data_raises_when_contract_unresolved(dev_config, app_factory, sample_bars):
    app = app_factory(
        bars=sample_bars, contract_details=make_contract_details(contract=None)
    )
    with pytest.raises(RuntimeError, match="Could not retrieve contract"):
        get_benchmark_data(app, dev_config, reqId=1)
    assert "reqHistoricalData" not in app.call_names()


# ---------------------------------------------------------------------------
# get_fama_factors()
# ---------------------------------------------------------------------------


def test_get_fama_factors_loads_from_project_root(dev_config, fama_csv):
    result = get_fama_factors(dev_config)
    assert len(result) == 5
    assert isinstance(result.index, pd.DatetimeIndex)


def test_get_fama_factors_renames_mkt_rf(dev_config, fama_csv):
    result = get_fama_factors(dev_config)
    assert "Mkt" in result.columns
    assert "Mkt-RF" not in result.columns


def test_get_fama_factors_keeps_remaining_factor_columns(dev_config, fama_csv):
    result = get_fama_factors(dev_config)
    assert list(result.columns) == ["Mkt", "SMB", "HML", "RMW", "CMA", "RF"]


def test_get_fama_factors_writes_the_cache(dev_config, fama_csv):
    get_fama_factors(dev_config)
    assert (dev_config.cache_dir / "FF_five_factor.csv").exists()


def test_get_fama_factors_honours_custom_save_filename(dev_config, fama_csv):
    get_fama_factors(dev_config, save_filename="ff3.csv")
    assert (dev_config.cache_dir / "ff3.csv").exists()


def test_get_fama_factors_skips_saving_when_save_is_false(dev_config, fama_csv):
    get_fama_factors(dev_config, save=False)
    assert not (dev_config.cache_dir / "FF_five_factor.csv").exists()


def test_get_fama_factors_prefers_the_cache(dev_config, fama_csv, capsys):
    cached = pd.DataFrame(
        {"Mkt": [9.99], "SMB": [0.0]},
        index=pd.DatetimeIndex(["2021-01-04"], name="Date"),
    )
    cached.to_csv(dev_config.cache_dir / "FF_five_factor.csv")

    result = get_fama_factors(dev_config)

    assert "Loading cached Fama-French data." in capsys.readouterr().out
    assert result["Mkt"].tolist() == [9.99]


def test_get_fama_factors_cache_round_trips(dev_config, fama_csv):
    first = get_fama_factors(dev_config)
    second = get_fama_factors(dev_config)
    pd.testing.assert_frame_equal(first, second)


def test_get_fama_factors_filters_by_start_date(dev_config, fama_csv):
    result = get_fama_factors(dev_config, start_date="2023-01-05", save=False)
    assert result.index.min() == pd.Timestamp("2023-01-05")
    assert len(result) == 3


def test_get_fama_factors_filters_by_end_date(dev_config, fama_csv):
    result = get_fama_factors(dev_config, end_date="2023-01-04", save=False)
    assert result.index.max() == pd.Timestamp("2023-01-04")
    assert len(result) == 2


def test_get_fama_factors_filters_by_both_dates(dev_config, fama_csv):
    result = get_fama_factors(
        dev_config, start_date="2023-01-04", end_date="2023-01-06", save=False
    )
    assert len(result) == 3
    assert result.index.min() == pd.Timestamp("2023-01-04")
    assert result.index.max() == pd.Timestamp("2023-01-06")


def test_get_fama_factors_returns_everything_without_dates(dev_config, fama_csv):
    result = get_fama_factors(dev_config, save=False)
    assert len(result) == 5


def test_get_fama_factors_selects_requested_columns(dev_config, fama_csv):
    result = get_fama_factors(
        dev_config, start_date="2023-01-03", cols=["Mkt-RF", "SMB"], save=False
    )
    assert list(result.columns) == ["Mkt", "SMB"]


def test_get_fama_factors_applies_cols_when_no_dates_given(dev_config, fama_csv):
    result = get_fama_factors(dev_config, cols=["Mkt-RF"], save=False)
    assert list(result.columns) == ["Mkt"]
    assert len(result) == 5


def test_get_fama_factors_applies_cols_with_end_date_only(dev_config, fama_csv):
    result = get_fama_factors(
        dev_config, end_date="2023-01-04", cols=["Mkt-RF", "RF"], save=False
    )
    assert list(result.columns) == ["Mkt", "RF"]
    assert len(result) == 2


def test_get_fama_factors_cols_order_is_preserved(dev_config, fama_csv):
    result = get_fama_factors(dev_config, cols=["RF", "SMB", "Mkt-RF"], save=False)
    assert list(result.columns) == ["RF", "SMB", "Mkt"]


def test_get_fama_factors_cols_subset_excluding_market(dev_config, fama_csv):
    # No "Mkt-RF" in cols, so the rename must not fire.
    result = get_fama_factors(dev_config, cols=["SMB", "HML"], save=False)
    assert list(result.columns) == ["SMB", "HML"]
    assert "Mkt" not in result.columns


def test_get_fama_factors_raises_for_unknown_column(dev_config, fama_csv):
    with pytest.raises(KeyError):
        get_fama_factors(dev_config, cols=["NotAFactor"], save=False)


def test_get_fama_factors_saved_cache_reflects_cols_subset(dev_config, fama_csv):
    get_fama_factors(dev_config, cols=["Mkt-RF", "SMB"])
    cached = pd.read_csv(
        dev_config.cache_dir / "FF_five_factor.csv", index_col=0, parse_dates=True
    )
    assert list(cached.columns) == ["Mkt", "SMB"]


def test_get_fama_factors_cached_read_does_not_rerename(dev_config, fama_csv):
    # First call renames Mkt-RF -> Mkt and caches that. The second call reads
    # the cache, where cols no longer contains "Mkt-RF", so the rename is
    # skipped and the column name stays stable.
    first = get_fama_factors(dev_config)
    second = get_fama_factors(dev_config)
    assert list(first.columns) == list(second.columns)
    assert "Mkt" in second.columns


def test_get_fama_factors_cached_read_rejects_prerename_cols(dev_config, fama_csv):
    # Known sharp edge: the cache stores the renamed "Mkt" column, so passing
    # the original "Mkt-RF" name on a later call no longer resolves.
    get_fama_factors(dev_config)
    with pytest.raises(KeyError):
        get_fama_factors(dev_config, cols=["Mkt-RF", "SMB"], save=False)


def test_get_fama_factors_does_not_mutate_the_source_frame(dev_config, fama_csv):
    first = get_fama_factors(dev_config, start_date="2023-01-04", save=False)
    second = get_fama_factors(dev_config, start_date="2023-01-04", save=False)
    pd.testing.assert_frame_equal(first, second)


def test_get_fama_factors_raises_when_no_file_is_found(dev_config):
    with pytest.raises(RuntimeError, match="Could not find fama french CSV file"):
        get_fama_factors(dev_config)


def test_get_fama_factors_raises_for_a_wrong_given_filename(dev_config, fama_csv):
    with pytest.raises(RuntimeError):
        get_fama_factors(dev_config, given_filename="does_not_exist.csv")


def test_get_fama_factors_honours_custom_given_filename(dev_config, fama_csv):
    renamed = dev_config.root_dir / "ff_custom.csv"
    (dev_config.root_dir / fama_csv).rename(renamed)
    result = get_fama_factors(dev_config, given_filename="ff_custom.csv", save=False)
    assert len(result) == 5


def test_get_fama_factors_values_match_the_source_file(dev_config, fama_csv):
    result = get_fama_factors(dev_config, save=False)
    assert result.loc["2023-01-03", "Mkt"] == pytest.approx(-0.67)
    assert result.loc["2023-01-06", "SMB"] == pytest.approx(0.31)
    assert result["RF"].tolist() == pytest.approx([0.01] * 5)
