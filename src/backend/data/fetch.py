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
    ticker: str = "NVDA",
    dur: str = "1 Y",
    reqId: int = 1,
) -> pd.DataFrame:
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

    contract = Contract()
    contract.symbol = ticker
    contract.secType = config.sec_type
    contract.exchange = config.exchange
    contract.currency = config.currency

    end_time = time.strftime("%Y%m%d %H:%M:%S")
    app.reqHistoricalData(
        reqId=reqId,
        contract=contract,
        endDateTime=end_time,
        durationStr=dur,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    while not app.hd_finished:
        time.sleep(0.5)

    app.disconnect()

    df = pd.DataFrame(app.data)
    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"])
    df = df["datetime,open,close,high,low,volume,return".split(",")]
    df.to_csv(f"{config.cache_dir}/{ticker}_data.csv", index=False)

    return df


def get_account_summary(
    app: IBApp, config: DevConfig, name: str = "All", reqId: int = 1
) -> dict:
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

    while not app.ad_finished:
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
