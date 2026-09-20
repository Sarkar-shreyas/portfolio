"""Test doubles and synthetic-payload factories for the data test suite.

Kept out of ``conftest.py`` so that test modules can import the builders
directly (``conftest`` is loaded by pytest, not imported by name).

``FakeIBApp`` is the important piece: it subclasses the real :class:`IBApp`
rather than replacing it with a ``Mock``. ``IBApp.__init__`` performs no I/O
(``EClient.__init__`` only initialises in-memory state), so the instance keeps
the genuine attribute set *and* the genuine ``EWrapper`` callback
implementations. Only the network-facing ``EClient`` request methods are
overridden, and each one feeds canned responses straight back into those real
callbacks. That flips the ``*_finished`` flags the ``fetch`` polling loops wait
on, so the orchestration in ``fetch`` runs unmodified and to completion.
"""

import time as _real_time

from ibapi.common import BarData
from ibapi.contract import Contract, ContractDetails

from src.backend.data.ib_client import IBApp


# ---------------------------------------------------------------------------
# Synthetic IBKR payload builders
# ---------------------------------------------------------------------------


def make_bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> BarData:
    """Build a ``BarData`` instance matching what the IBKR API delivers."""
    bar = BarData()
    bar.date = date
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


def make_contract(
    con_id: int = 4815747,
    symbol: str = "NVDA",
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str = "NASDAQ",
) -> Contract:
    """Build a fully populated ``Contract``."""
    contract = Contract()
    contract.conId = con_id
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    contract.primaryExchange = primary_exchange
    return contract


def make_contract_details(contract=None) -> ContractDetails:
    """Wrap a contract in a ``ContractDetails``, as ``reqContractDetails`` returns.

    Passing ``contract=None`` explicitly is *not* the same as omitting it: use
    ``make_contract_details(contract=None)`` to model a details payload whose
    contract failed to resolve.
    """
    details = ContractDetails()
    details.contract = contract
    return details


def make_portfolio_row(
    symbol: str = "NVDA",
    con_id: int = 4815747,
    position: float = 100.0,
    market_price: float = 125.5,
    market_value: float = 12550.0,
    average_cost: float = 110.0,
    unrealized_pnl: float = 1550.0,
    realized_pnl: float = 0.0,
    account_name: str = "DU1234567",
) -> dict:
    """Keyword arguments for a single ``updatePortfolio`` callback."""
    return {
        "contract": make_contract(con_id=con_id, symbol=symbol),
        "position": position,
        "marketPrice": market_price,
        "marketValue": market_value,
        "averageCost": average_cost,
        "unrealizedPNL": unrealized_pnl,
        "realizedPNL": realized_pnl,
        "accountName": account_name,
    }


# ---------------------------------------------------------------------------
# Time control
# ---------------------------------------------------------------------------


class FakeTime:
    """Drop-in replacement for the ``time`` module used inside ``fetch``.

    ``sleep`` returns immediately but records the requested duration, so the
    polling loops in ``fetch`` run at full speed without changing their logic.
    """

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def strftime(self, fmt: str, *args) -> str:
        return _real_time.strftime(fmt, *args)


# ---------------------------------------------------------------------------
# IBApp test double
# ---------------------------------------------------------------------------


class FakeIBApp(IBApp):
    """An ``IBApp`` with its socket layer replaced by canned, synchronous replies.

    Every overridden ``req*`` method records its arguments in :attr:`calls` and
    then invokes the corresponding real ``EWrapper`` callbacks.
    """

    def __init__(
        self,
        bars: list | None = None,
        contract_details: ContractDetails | None = None,
        account_rows: dict | None = None,
        portfolio_rows: list | None = None,
        connect_error: Exception | None = None,
        send_contract_details: bool = True,
    ) -> None:
        super().__init__()
        self._bars = bars if bars is not None else []
        self._contract_details = (
            make_contract_details(make_contract())
            if contract_details is None
            else contract_details
        )
        self._account_rows = account_rows if account_rows is not None else {}
        self._portfolio_rows = portfolio_rows if portfolio_rows is not None else []
        self._connect_error = connect_error
        self._send_contract_details = send_contract_details
        self.connected = False
        self.calls: list[tuple[str, dict]] = []

    # -- assertion helpers --------------------------------------------------

    def call_names(self) -> list[str]:
        """Just the ordered method names from :attr:`calls`."""
        return [name for name, _ in self.calls]

    def call_args(self, name: str) -> dict:
        """Arguments of the first recorded call to ``name``."""
        for recorded, kwargs in self.calls:
            if recorded == name:
                return kwargs
        raise AssertionError(f"{name!r} was never called. Calls: {self.call_names()}")

    # -- EClient network layer ---------------------------------------------

    def connect(self, host: str, port: int, clientId: int) -> None:
        self.calls.append(
            ("connect", {"host": host, "port": port, "clientId": clientId})
        )
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.calls.append(("disconnect", {}))
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def run(self) -> None:
        # The real EClient.run() blocks draining the socket queue. Responses
        # here are delivered synchronously, so the reader thread has no work.
        self.calls.append(("run", {}))

    def reqContractDetails(self, reqId: int, contract: Contract) -> None:
        self.calls.append(
            ("reqContractDetails", {"reqId": reqId, "contract": contract})
        )
        if self._send_contract_details:
            self.contractDetails(reqId, self._contract_details)
        self.contractDetailsEnd(reqId)

    def reqHistoricalData(
        self,
        reqId: int,
        contract: Contract,
        endDateTime: str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: int,
        formatDate: int,
        keepUpToDate: bool,
        chartOptions: list,
    ) -> None:
        self.calls.append(
            (
                "reqHistoricalData",
                {
                    "reqId": reqId,
                    "contract": contract,
                    "endDateTime": endDateTime,
                    "durationStr": durationStr,
                    "barSizeSetting": barSizeSetting,
                    "whatToShow": whatToShow,
                    "useRTH": useRTH,
                    "formatDate": formatDate,
                    "keepUpToDate": keepUpToDate,
                    "chartOptions": chartOptions,
                },
            )
        )
        for bar in self._bars:
            self.historicalData(reqId, bar)
        self.historicalDataEnd(reqId, "", "")

    def reqAccountSummary(self, reqId: int, groupName: str, tags: str) -> None:
        self.calls.append(
            (
                "reqAccountSummary",
                {"reqId": reqId, "groupName": groupName, "tags": tags},
            )
        )
        for tag, (value, currency) in self._account_rows.items():
            self.accountSummary(reqId, "DU1234567", tag, value, currency)
        self.accountSummaryEnd(reqId)

    def reqAccountUpdates(self, subscribe: bool, acctCode: str) -> None:
        self.calls.append(
            ("reqAccountUpdates", {"subscribe": subscribe, "acctCode": acctCode})
        )
        if not subscribe:
            return
        for row in self._portfolio_rows:
            self.updatePortfolio(**row)
        self.accountDownloadEnd("DU1234567")
