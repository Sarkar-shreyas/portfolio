"""Tests for ``src.backend.data.ib_client``.

Each ``EWrapper`` callback is exercised directly on a real ``IBApp`` instance;
constructing one performs no I/O, so no mocking is required here.
"""

import pytest

from src.backend.data.ib_client import Account, IBApp, Position, run_loop

from src.backend.tests.test_data.helpers import (
    FakeIBApp,
    make_bar,
    make_contract,
    make_contract_details,
)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def test_position_stores_all_fields():
    position = Position(
        con_id=4815747,
        symbol="NVDA",
        quantity=100.0,
        average_cost=110.0,
        market_price=125.5,
        market_value=12550.0,
        unrealized_pnl=1550.0,
        realized_pnl=0.0,
    )
    assert position.con_id == 4815747
    assert position.symbol == "NVDA"
    assert position.quantity == 100.0
    assert position.average_cost == 110.0
    assert position.market_price == 125.5
    assert position.market_value == 12550.0
    assert position.unrealized_pnl == 1550.0
    assert position.realized_pnl == 0.0


def test_position_equality_is_by_value():
    fields = dict(
        con_id=1,
        symbol="NVDA",
        quantity=1.0,
        average_cost=1.0,
        market_price=1.0,
        market_value=1.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
    )
    assert Position(**fields) == Position(**fields)


def test_position_requires_all_fields():
    with pytest.raises(TypeError):
        Position(con_id=1, symbol="NVDA")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


def test_account_stores_all_fields():
    account = Account(
        net_liquidation=150000.0,
        cash=25000.0,
        buying_power=600000.0,
        equity_with_loan=150000.0,
        gross_position_value=125000.0,
        maintenance_margin=30000.0,
        available_funds=120000.0,
    )
    assert account.net_liquidation == 150000.0
    assert account.cash == 25000.0
    assert account.buying_power == 600000.0
    assert account.equity_with_loan == 150000.0
    assert account.gross_position_value == 125000.0
    assert account.maintenance_margin == 30000.0
    assert account.available_funds == 120000.0


def test_account_requires_all_fields():
    with pytest.raises(TypeError):
        Account(net_liquidation=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# IBApp.__init__
# ---------------------------------------------------------------------------


def test_init_starts_with_empty_containers():
    app = IBApp()
    assert app.data == {}
    assert app.positions == {}
    assert app.account == {}
    assert app.hd_finished == {}
    assert app.ad_finished == {}
    assert app.contract_details == {}
    assert app.cd_finished == {}


def test_init_portfolio_flag_starts_false():
    assert IBApp().pd_finished is False


def test_init_instances_do_not_share_state():
    first, second = IBApp(), IBApp()
    first.data[1] = ["bar"]
    assert second.data == {}


# ---------------------------------------------------------------------------
# IBApp.contractDetails
# ---------------------------------------------------------------------------


def test_contract_details_stores_payload_under_req_id():
    app = IBApp()
    details = make_contract_details(make_contract())
    app.contractDetails(1001, details)
    assert app.contract_details[1001] is details


def test_contract_details_keeps_requests_separate():
    app = IBApp()
    nvda = make_contract_details(make_contract(symbol="NVDA"))
    aapl = make_contract_details(make_contract(con_id=265598, symbol="AAPL"))
    app.contractDetails(1001, nvda)
    app.contractDetails(1002, aapl)
    assert app.contract_details[1001].contract.symbol == "NVDA"
    assert app.contract_details[1002].contract.symbol == "AAPL"


def test_contract_details_overwrites_previous_payload_for_same_req_id():
    app = IBApp()
    first = make_contract_details(make_contract(symbol="NVDA"))
    second = make_contract_details(make_contract(symbol="AMD"))
    app.contractDetails(1001, first)
    app.contractDetails(1001, second)
    assert app.contract_details[1001] is second


# ---------------------------------------------------------------------------
# IBApp.contractDetailsEnd
# ---------------------------------------------------------------------------


def test_contract_details_end_sets_flag():
    app = IBApp()
    app.contractDetailsEnd(1001)
    assert app.cd_finished[1001] is True


def test_contract_details_end_only_flags_its_own_req_id():
    app = IBApp()
    app.cd_finished[1002] = False
    app.contractDetailsEnd(1001)
    assert app.cd_finished[1001] is True
    assert app.cd_finished[1002] is False


# ---------------------------------------------------------------------------
# IBApp.historicalData
# ---------------------------------------------------------------------------


def test_historical_data_maps_bar_fields():
    app = IBApp()
    app.historicalData(1, make_bar("20230103", 100.0, 102.0, 99.0, 101.0, 1_000))
    assert app.data[1] == [
        {
            "datetime": "20230103",
            "open": 100.0,
            "close": 101.0,
            "high": 102.0,
            "low": 99.0,
            "volume": 1_000,
        }
    ]


def test_historical_data_appends_in_arrival_order(sample_bars):
    app = IBApp()
    for bar in sample_bars:
        app.historicalData(1, bar)
    assert [row["datetime"] for row in app.data[1]] == [
        "20230103",
        "20230104",
        "20230105",
    ]


def test_historical_data_creates_list_on_first_bar():
    app = IBApp()
    assert 1 not in app.data
    app.historicalData(1, make_bar("20230103", 1.0, 1.0, 1.0, 1.0, 1))
    assert isinstance(app.data[1], list)


def test_historical_data_keeps_requests_separate(sample_bars):
    app = IBApp()
    app.historicalData(1, sample_bars[0])
    app.historicalData(2, sample_bars[1])
    app.historicalData(2, sample_bars[2])
    assert len(app.data[1]) == 1
    assert len(app.data[2]) == 2


# ---------------------------------------------------------------------------
# IBApp.historicalDataEnd
# ---------------------------------------------------------------------------


def test_historical_data_end_sets_flag():
    app = IBApp()
    app.hd_finished[1] = False
    app.historicalDataEnd(1, "20230103", "20230105")
    assert app.hd_finished[1] is True


def test_historical_data_end_only_flags_its_own_req_id():
    app = IBApp()
    app.hd_finished[1] = False
    app.hd_finished[2] = False
    app.historicalDataEnd(1, "", "")
    assert app.hd_finished[1] is True
    assert app.hd_finished[2] is False


def test_historical_data_end_announces_completion(capsys):
    IBApp().historicalDataEnd(1, "", "")
    assert "Finished retrieving Historical Market Data" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# IBApp.updatePortfolio
# ---------------------------------------------------------------------------


def test_update_portfolio_maps_contract_and_pnl_fields():
    app = IBApp()
    contract = make_contract(con_id=4815747, symbol="NVDA")
    app.updatePortfolio(
        contract=contract,
        position=100.0,
        marketPrice=125.5,
        marketValue=12550.0,
        averageCost=110.0,
        unrealizedPNL=1550.0,
        realizedPNL=25.0,
        accountName="DU1234567",
    )
    assert app.positions["NVDA"] == {
        "id": 4815747,
        "secType": "STK",
        "primaryExchange": "NASDAQ",
        "currency": "USD",
        "position": 100.0,
        "marketPrice": 125.5,
        "marketValue": 12550.0,
        "averageCost": 110.0,
        "unrealizedPNL": 1550.0,
        "realizedPNL": 25.0,
    }


def test_update_portfolio_is_keyed_by_symbol():
    app = IBApp()
    app.updatePortfolio(
        make_contract(symbol="NVDA"), 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, "DU1"
    )
    app.updatePortfolio(
        make_contract(con_id=265598, symbol="AAPL"),
        2.0,
        2.0,
        2.0,
        2.0,
        0.0,
        0.0,
        "DU1",
    )
    assert set(app.positions) == {"NVDA", "AAPL"}


def test_update_portfolio_overwrites_on_refresh():
    app = IBApp()
    contract = make_contract(symbol="NVDA")
    app.updatePortfolio(contract, 100.0, 125.5, 12550.0, 110.0, 1550.0, 0.0, "DU1")
    app.updatePortfolio(contract, 150.0, 130.0, 19500.0, 112.0, 2700.0, 0.0, "DU1")
    assert len(app.positions) == 1
    assert app.positions["NVDA"]["position"] == 150.0
    assert app.positions["NVDA"]["marketPrice"] == 130.0


def test_update_portfolio_does_not_record_account_name():
    app = IBApp()
    app.updatePortfolio(
        make_contract(symbol="NVDA"), 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, "DU1234567"
    )
    assert "accountName" not in app.positions["NVDA"]


# ---------------------------------------------------------------------------
# IBApp.accountDownloadEnd
# ---------------------------------------------------------------------------


def test_account_download_end_sets_flag():
    app = IBApp()
    app.accountDownloadEnd("DU1234567")
    assert app.pd_finished is True


def test_account_download_end_announces_completion(capsys):
    IBApp().accountDownloadEnd("DU1234567")
    assert "Finished retrieving Portfolio data" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# IBApp.accountSummary
# ---------------------------------------------------------------------------


def test_account_summary_stores_value_and_currency_by_tag():
    app = IBApp()
    app.accountSummary(1, "DU1234567", "NetLiquidation", "150000.00", "USD")
    assert app.account["NetLiquidation"] == {"value": "150000.00", "currency": "USD"}


def test_account_summary_accumulates_tags(sample_account_rows):
    app = IBApp()
    for tag, (value, currency) in sample_account_rows.items():
        app.accountSummary(1, "DU1234567", tag, value, currency)
    assert set(app.account) == set(sample_account_rows)


def test_account_summary_overwrites_repeated_tag():
    app = IBApp()
    app.accountSummary(1, "DU1234567", "NetLiquidation", "150000.00", "USD")
    app.accountSummary(1, "DU1234567", "NetLiquidation", "151000.00", "USD")
    assert app.account["NetLiquidation"]["value"] == "151000.00"


def test_account_summary_does_not_record_req_id_or_account():
    app = IBApp()
    app.accountSummary(9, "DU1234567", "BuyingPower", "600000.00", "USD")
    assert app.account["BuyingPower"] == {"value": "600000.00", "currency": "USD"}


# ---------------------------------------------------------------------------
# IBApp.accountSummaryEnd
# ---------------------------------------------------------------------------


def test_account_summary_end_sets_flag():
    app = IBApp()
    app.ad_finished[1] = False
    app.accountSummaryEnd(1)
    assert app.ad_finished[1] is True


def test_account_summary_end_only_flags_its_own_req_id():
    app = IBApp()
    app.ad_finished[2] = False
    app.accountSummaryEnd(1)
    assert app.ad_finished[1] is True
    assert app.ad_finished[2] is False


def test_account_summary_end_announces_completion(capsys):
    IBApp().accountSummaryEnd(1)
    assert "Finished retrieving Account data" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# IBApp.error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [2104, 2106, 2158])
def test_error_suppresses_benign_connection_notices(capsys, code):
    IBApp().error(1, code, "Market data farm connection is OK")
    assert capsys.readouterr().out == ""


def test_error_prints_real_errors(capsys):
    IBApp().error(7, 162, "Historical Market Data Service error message")
    out = capsys.readouterr().out
    assert "7" in out
    assert "162" in out
    assert "Historical Market Data Service error message" in out


def test_error_does_not_mutate_app_state():
    app = IBApp()
    app.error(1, 162, "boom")
    assert app.data == {}
    assert app.hd_finished == {}


# ---------------------------------------------------------------------------
# run_loop
# ---------------------------------------------------------------------------


def test_run_loop_invokes_app_run():
    app = FakeIBApp()
    run_loop(app)
    assert app.call_names() == ["run"]


def test_run_loop_returns_none():
    assert run_loop(FakeIBApp()) is None
