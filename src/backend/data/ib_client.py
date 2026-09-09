from threading import Thread
import time
import sys

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from config import DevConfig


class IBApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.finished = False

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.finished = True

    def accountSummaryEnd(self, reqId: int):
        self.finished = True


def start_conn(app: IBApp, config: DevConfig):
    try:
        app.connect(
            host=config.IB_HOST, port=config.IB_PORT, clientId=config.IB_CLIENT_ID
        )
        print("Connected to IBKR Gateway")
        thread = Thread(target=app.run(), args=(app,))
        thread.start()
        time.sleep(1)
    except EWrapper.error:
        print("Could not connect to IBKR Gateway.")
        sys.exit(0)


def end_conn(app: IBApp):
    app.disconnect()
