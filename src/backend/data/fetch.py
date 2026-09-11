import pandas as pd
import time
from threading import Thread
import sys
from ibapi.contract import Contract
import json
# from ibapi.account_summary_tags import AccountSummaryTags

from src.backend.data.ib_client import IBApp, run_loop
from src.backend.config import DevConfig

ACCOUNT_SUMMARY_TAGS = (
    "NetLiquidation,"
    "TotalCashValue,"
    "BuyingPower,"
    "EquityWithLoanValue,"
    "GrossPositionValue,"
    "MaintMarginReq,"
    "AvailableFunds,"
    "ExcessLiquidity"
)


def get_equity_data(
    app: IBApp,
    config: DevConfig,
    conid: int,
    ticker: str = "NVDA",
    sec_type: str = "STK",
    dur: str = "1 Y",
    reqId: int = 1,
) -> pd.DataFrame:
    app.hd_finished[reqId] = False
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
        time.sleep(1)
    except Exception as e:
        print(f"Could not connect to ibkr gateway: {e}")
        sys.exit(0)

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    req_contract = Contract()
    req_contract.conId = conid
    con_reqId = reqId + 1000
    app.contract_details[con_reqId] = None
    app.cd_finished[con_reqId] = False
    app.reqContractDetails(con_reqId, req_contract)

    while not app.cd_finished[con_reqId]:
        time.sleep(0.5)

    contract = app.contract_details[con_reqId].contract
    print(contract)

    if not contract:
        print("Error: Could not retrieve contract")
        app.disconnect()
        sys.exit(0)

    print(f"{ticker}: {contract}")
    end_time = time.strftime("%Y%m%d %H:%M:%S")
    app.reqHistoricalData(
        reqId=reqId,
        contract=contract,
        endDateTime="",
        durationStr=dur,
        barSizeSetting="1 day",
        whatToShow="ADJUSTED_LAST",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )
    while not app.hd_finished[reqId]:
        time.sleep(0.5)

    app.disconnect()

    df = pd.DataFrame(app.data[reqId])
    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"])
    df = df["datetime,open,close,high,low,volume,return".split(",")]
    df.to_csv(f"{config.cache_dir}/{ticker}_data.csv", index=False)

    return df


def get_account_summary(
    app: IBApp, config: DevConfig, name: str = "All", reqId: int = 1
) -> dict:
    app.ad_finished[reqId] = False
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
    except Exception as e:
        print("Could not connect to ibkr gateway")
        sys.exit(0)

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    app.reqAccountSummary(reqId, name, ACCOUNT_SUMMARY_TAGS)

    while not app.ad_finished[reqId]:
        time.sleep(0.5)

    app.disconnect()

    account_data = app.account
    with open(f"{config.cache_dir}/acc_summary.json", "w") as file:
        json.dump(account_data, file, indent=2)

    return account_data


def get_portfolio_data(app: IBApp, config: DevConfig) -> dict:
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
    except Exception as e:
        print("Could not connect to ibkr gateway")
        sys.exit(0)

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    app.reqAccountUpdates(True, "")

    while not app.pd_finished:
        time.sleep(0.5)

    app.reqAccountUpdates(False, "")
    app.disconnect()

    # df = pd.DataFrame(app.positions)
    # df.to_csv(f"{config.cache_dir}/portfolio.csv", index=False)

    return app.positions
