from ibapi.client import EClient
from ibapi.contract import Contract, ContractDetails
from ibapi.wrapper import EWrapper
from ibapi.common import BarData

from dataclasses import dataclass

__all__ = ["IBApp", "run_loop"]


# Unused currently
@dataclass
class Position:
    con_id: int
    symbol: str
    quantity: float
    average_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


# Unused currently
@dataclass
class Account:
    net_liquidation: float
    cash: float
    buying_power: float
    equity_with_loan: float
    gross_position_value: float
    maintenance_margin: float
    available_funds: float


class IBApp(EWrapper, EClient):
    """
    Class instance inheriting from ibapi's EWrapper and EClient classes. Overwrites several
    methods to enable simpler data storage and track event completion.

    Parameters:
    -----------
    data: dict
        Bar data retrieved by historicalData, keyed by request ID (int)
    positions: dict
        Portfolio data retrieved by updatePortfolio, keyed by asset symbols (str)
    account: dict
        Account summary info retrieved by accountSummary, keyed by account tag (str)
    hd_finished: dict
        Boolean flag representing a historicalData request has completed successfully, keyed by request ID (int)
    pd_finished: bool
        Boolean flag representing an updatePortfolio request has completed successfully.
    ad_finished: dict
        Boolean flags representing an accountSummary request has completed successfully, keyed by request ID (int)
    contract_details: dict
        Stores contract details retrieved from contractDetails, keyed by request ID (int)
    cd_finished: dict
        Boolean flag representing a contractDetails request has completed successfully, keyed by request ID (int)
    """

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.data = {}
        self.positions = {}
        self.account = {}
        self.hd_finished = {}
        self.pd_finished = False
        self.ad_finished = {}
        self.contract_details = {}
        self.cd_finished = {}

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        if reqId not in self.contract_details:
            # print(f"Adding new contract ID {reqId}")
            self.contract_details[reqId] = None

        self.contract_details[reqId] = contractDetails

    def contractDetailsEnd(self, reqId: int):
        self.cd_finished[reqId] = True

    def historicalData(self, reqId: int, bar: BarData) -> None:
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append(
            {
                "datetime": bar.date,
                "open": bar.open,
                "close": bar.close,
                "high": bar.high,
                "low": bar.low,
                "volume": bar.volume,
            }
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print("Finished retrieving Historical Market Data")
        self.hd_finished[reqId] = True

    def updatePortfolio(
        self,
        contract: Contract,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ):
        self.positions[contract.symbol] = {
            "id": contract.conId,
            "secType": contract.secType,
            "primaryExchange": contract.primaryExchange,
            "currency": contract.currency,
            "position": position,
            "marketPrice": marketPrice,
            "marketValue": marketValue,
            "averageCost": averageCost,
            "unrealizedPNL": unrealizedPNL,
            "realizedPNL": realizedPNL,
        }

    def accountDownloadEnd(self, accountName: str):
        print("Finished retrieving Portfolio data")
        self.pd_finished = True

    def accountSummary(
        self, reqId: int, account: str, tag: str, value: str, currency: str
    ):
        self.account[tag] = {"value": value, "currency": currency}

    def accountSummaryEnd(self, reqId: int):
        print("Finished retrieving Account data")
        self.ad_finished[reqId] = True

    def error(self, reqId: int, errorCode: int, errorString: str):
        if errorCode not in [2104, 2106, 2158]:
            print(reqId, errorCode, errorString)


def run_loop(app: IBApp):
    """Helper function for easy referral to threads."""
    app.run()
