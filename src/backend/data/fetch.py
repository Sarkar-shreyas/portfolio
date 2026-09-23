import pandas as pd
import time
from threading import Thread
import sys
from ibapi.contract import Contract
import requests
import os
from typing import Optional

# from ibapi.account_summary_tags import AccountSummaryTags
import json
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


def read_json(filepath: str) -> dict:
    """Returns a dictionary containing data from the given json file"""
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def get_equity_data(
    app: IBApp,
    config: DevConfig,
    conid: int,
    ticker: str = "NVDA",
    dur: str = "1 Y",
    reqId: int = 1,
) -> pd.DataFrame:
    """Retrieves historical equity data via the IBKR API."""
    if os.path.exists(f"{config.cache_dir}/{ticker}_data.csv"):
        print(f"Loading existing {ticker} data.")
        data = pd.read_csv(f"{config.cache_dir}/{ticker}_data.csv")
        return data
    app.hd_finished[reqId] = False
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
        time.sleep(1)
    except Exception as e:
        raise RuntimeError(f"Could not connect to IB Gateway: {e}")

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
        app.disconnect()
        raise RuntimeError(f"Could not retrieve contract for id {con_reqId}")

    print(f"{ticker}: {contract}")
    # end_time = time.strftime("%Y%m%d %H:%M:%S")
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
    time.sleep(1)
    df = pd.DataFrame(app.data.get(reqId, []))

    return df


def get_account_summary(
    app: IBApp, config: DevConfig, name: str = "All", reqId: int = 1
) -> dict:
    """Retrieves Account summary data via the IBKR API."""
    if os.path.exists(f"{config.cache_dir}/acc_summary.json"):
        data = read_json(f"{config.cache_dir}/acc_summary.json")
        print("Loading existing account data")
        return data
    app.ad_finished[reqId] = False
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
    except Exception as e:
        raise RuntimeError(f"Could not connect to IB Gateway: {e}")

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    app.reqAccountSummary(reqId, name, ACCOUNT_SUMMARY_TAGS)

    while not app.ad_finished[reqId]:
        time.sleep(0.5)

    app.disconnect()

    account_data = app.account

    return account_data


def get_portfolio_data(app: IBApp, config: DevConfig) -> dict:
    """Retrieves portfolio data using the IBKR API"""
    if os.path.exists(f"{config.cache_dir}/portfolio.csv"):
        print("Loading existing portfolio data")
        data = pd.read_csv(f"{config.cache_dir}/portfolio.csv", index_col="ticker")
        return data.to_dict(orient="index")
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
    except Exception as e:
        raise RuntimeError(f"Could not connect to IB Gateway: {e}")

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    app.reqAccountUpdates(True, "")

    while not app.pd_finished:
        time.sleep(0.5)

    app.reqAccountUpdates(False, "")
    app.disconnect()

    return app.positions


def get_market_cap(config: DevConfig, symbols: list = []) -> dict:
    """Retrieves market cap information for the given symbols via the alphavantage API."""
    if not symbols:
        return {}
    if len(config.ALPHA_VANTAGE_KEY) == 0:
        raise ValueError("Could not retrieve API key.")
    ticker_data = {}
    for symbol in symbols:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={config.ALPHA_VANTAGE_KEY}"
        try:
            print(f"Fetching data for {symbol}")
            r = requests.get(url)
            data = r.json()
            ticker_data[symbol.strip().upper()] = int(data["MarketCapitalization"])
        except Exception as e:
            raise RuntimeError(f"Error retrieving info for {symbol}: {e}")
        time.sleep(1)
    return ticker_data


def get_benchmark_data(app: IBApp, config: DevConfig, reqId: int) -> pd.DataFrame:
    """Retrieves historical S&P500 data via the IBKR API."""
    if os.path.exists(f"{config.cache_dir}/SPX_data.csv"):
        data = pd.read_csv(f"{config.cache_dir}/SPX_data.csv")
        return data
    app.hd_finished[reqId] = False
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
        time.sleep(1)
    except Exception as e:
        raise RuntimeError(f"Could not connect to IB Gateway: {e}")

    thread = Thread(target=run_loop, args=(app,))
    thread.start()
    time.sleep(1)

    req_contract = Contract()
    req_contract.conId = 416904
    con_reqId = reqId + 1000
    app.contract_details[con_reqId] = None
    app.cd_finished[con_reqId] = False
    app.reqContractDetails(con_reqId, req_contract)

    while not app.cd_finished[con_reqId]:
        time.sleep(0.5)

    contract = app.contract_details[con_reqId].contract
    print(contract)

    if not contract:
        app.disconnect()
        raise RuntimeError(f"Could not retrieve contract for id {con_reqId}")

    # end_time = time.strftime("%Y%m%d %H:%M:%S")
    app.reqHistoricalData(
        reqId=reqId,
        contract=contract,
        endDateTime="",
        durationStr="5 Y",
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
    time.sleep(1)
    print("App disconnected.")
    df = pd.DataFrame(app.data.get(reqId, []))

    return df


def get_fama_factors(
    config: DevConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_col: int = 0,
    skiprows: int = 4,
    header: int = 0,
    skipfooter: int = 2,
    engine: str = "python",
    given_filename: str = "fama_french_daily_five.csv",
    save_filename: str = "FF_five_factor.csv",
    cols: list = [],
    save: bool = True,
) -> pd.DataFrame:
    """Loads the daily fama french factors from the csv file uploaded in the project root, then saves the given date range to the cache."""
    if os.path.exists(f"{config.cache_dir}/{save_filename}"):
        print("Loading cached Fama-French data.")
        fama_data = pd.read_csv(
            f"{config.cache_dir}/{save_filename}",
            parse_dates=True,
            index_col=index_col,
            header=header,
        )
    elif os.path.exists(f"{config.root_dir}/{given_filename}"):
        fama_data = pd.read_csv(
            f"{config.root_dir}/{given_filename}",
            parse_dates=True,
            index_col=index_col,
            skiprows=skiprows,
            header=header,
            skipfooter=skipfooter,
            engine=engine,  # type: ignore
        )
    else:
        raise RuntimeError(
            "Could not find fama french CSV file. Check given_filename or file location."
        )
    if not cols:
        cols = list(fama_data.columns)
    if start_date is None and end_date is None:
        fama = fama_data.copy().loc[:, cols]
    elif end_date is None:
        fama = fama_data.copy().loc[start_date:, cols]
    elif start_date is None:
        fama = fama_data.copy().loc[:end_date, cols]
    else:
        fama = fama_data.copy().loc[start_date:end_date, cols]
    if "Mkt-RF" in cols:
        fama.rename(columns={"Mkt-RF": "Mkt"}, inplace=True)
    if save:
        fama.to_csv(f"{config.cache_dir}/{save_filename}")

    return fama
