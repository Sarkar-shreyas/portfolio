from ibapi.client import EClient
from ibapi.contract import Contract, ContractDetails
from ibapi.wrapper import EWrapper
from ibapi.common import BarData

from dataclasses import dataclass


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


def run_loop(app: IBApp):
    app.run()
