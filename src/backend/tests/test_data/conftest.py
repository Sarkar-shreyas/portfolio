"""Shared fixtures for the data test suite.

Nothing here touches the network, the real IBKR gateway, or any HTTP endpoint.
All filesystem writes are confined to pytest's ``tmp_path``. The test doubles
themselves live in :mod:`src.backend.tests.test_data.helpers`.
"""

import pandas as pd
import pytest

from src.backend.config import DevConfig, TestConfig
from src.backend.data import fetch as fetch_module
from src.backend.tests.test_data.helpers import (
    FakeIBApp,
    FakeTime,
    make_bar,
    make_portfolio_row,
)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path) -> TestConfig:
    """A TestConfig whose cache/root directories point at a temp dir."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return TestConfig(root_dir=tmp_path, cache_dir=cache_dir, trial_dir=tmp_path)


@pytest.fixture
def dev_config(tmp_path) -> DevConfig:
    """A DevConfig with temp directories and deterministic connection params."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return DevConfig(
        root_dir=tmp_path,
        cache_dir=cache_dir,
        trial_dir=tmp_path,
        IB_HOST="127.0.0.1",
        IB_PORT=4002,
        IB_CLIENT_ID=7,
        ALPHA_VANTAGE_KEY="test-key",
    )


# ---------------------------------------------------------------------------
# Time control
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_time(monkeypatch) -> FakeTime:
    """Swap ``fetch``'s ``time`` reference so no test actually sleeps.

    Autouse: every polling loop in ``fetch`` sleeps between iterations, and the
    suite would otherwise spend seconds per test waiting on flags that the fake
    app has already set.
    """
    clock = FakeTime()
    monkeypatch.setattr(fetch_module, "time", clock)
    return clock


# ---------------------------------------------------------------------------
# Synthetic IBKR payloads
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bars() -> list:
    """Three consecutive daily bars."""
    return [
        make_bar("20230103", 100.0, 102.0, 99.0, 101.0, 1_000),
        make_bar("20230104", 101.0, 104.0, 100.5, 103.5, 1_200),
        make_bar("20230105", 103.5, 105.0, 102.0, 102.5, 900),
    ]


@pytest.fixture
def sample_account_rows() -> dict:
    """``accountSummary`` rows keyed by tag, mapping to ``(value, currency)``."""
    return {
        "NetLiquidation": ("150000.00", "USD"),
        "TotalCashValue": ("25000.00", "USD"),
        "BuyingPower": ("600000.00", "USD"),
        "EquityWithLoanValue": ("150000.00", "USD"),
        "GrossPositionValue": ("125000.00", "USD"),
        "MaintMarginReq": ("30000.00", "USD"),
        "AvailableFunds": ("120000.00", "USD"),
        "ExcessLiquidity": ("120000.00", "USD"),
    }


@pytest.fixture
def sample_portfolio_rows() -> list:
    """Two ``updatePortfolio`` callback payloads."""
    return [
        make_portfolio_row(symbol="NVDA", con_id=4815747),
        make_portfolio_row(
            symbol="AAPL",
            con_id=265598,
            position=50.0,
            market_price=190.0,
            market_value=9500.0,
            average_cost=175.0,
            unrealized_pnl=750.0,
        ),
    ]


@pytest.fixture
def app_factory():
    """Factory producing ``FakeIBApp`` instances with per-test canned data."""

    def _factory(**kwargs) -> FakeIBApp:
        return FakeIBApp(**kwargs)

    return _factory


@pytest.fixture
def app(sample_bars, sample_account_rows, sample_portfolio_rows) -> FakeIBApp:
    """A ``FakeIBApp`` pre-loaded with the default synthetic payloads."""
    return FakeIBApp(
        bars=sample_bars,
        account_rows=sample_account_rows,
        portfolio_rows=sample_portfolio_rows,
    )


# ---------------------------------------------------------------------------
# Timeseries fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_bar_dict() -> dict:
    """Bars keyed by ``YYYYMMDD`` date string, as collected before cleaning."""
    return {
        "20230103": {
            "datetime": "20230103",
            "open": 100.0,
            "close": 101.0,
            "high": 102.0,
            "low": 99.0,
            "volume": 1_000,
        },
        "20230104": {
            "datetime": "20230104",
            "open": 101.0,
            "close": 103.5,
            "high": 104.0,
            "low": 100.5,
            "volume": 1_200,
        },
        "20230105": {
            "datetime": "20230105",
            "open": 103.5,
            "close": 102.5,
            "high": 105.0,
            "low": 102.0,
            "volume": 900,
        },
    }


@pytest.fixture
def raw_bar_frame(raw_bar_dict) -> pd.DataFrame:
    """``raw_bar_dict`` as a DataFrame indexed by the date strings."""
    return pd.DataFrame.from_dict(raw_bar_dict, orient="index")


@pytest.fixture
def datetime_indexed_frame() -> pd.DataFrame:
    """A frame already carrying a ``DatetimeIndex``, with one NaN row."""
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 103.5],
            "Close": [101.0, float("nan"), 102.5],
            "Volume": [1_000, 1_200, 900],
        },
        index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
    )


# ---------------------------------------------------------------------------
# Fama-French fixtures
# ---------------------------------------------------------------------------

FAMA_PREAMBLE = (
    "This file was created by using the 202607 CRSP database.\n"
    "The Tbill return is the simple daily rate.\n"
    "Starting from 202406, the 1-month TBill rate is from ICE BofA.\n"
    "\n"
)

FAMA_FOOTER = "\n\n  Annual Factors: January-December\n"

FAMA_ROWS = [
    ("20230103", -0.67, 0.00, -0.33, -0.01, 0.16, 0.01),
    ("20230104", 0.79, -0.26, 0.26, -0.07, -0.20, 0.01),
    ("20230105", 0.63, -0.17, -0.10, 0.18, -0.34, 0.01),
    ("20230106", 1.12, 0.31, -0.44, 0.05, 0.22, 0.01),
    ("20230109", -0.05, 0.12, 0.19, -0.11, 0.08, 0.01),
]


@pytest.fixture
def fama_filename() -> str:
    """Name of the raw Fama-French file expected in the project root."""
    return "fama_french_daily_five.csv"


@pytest.fixture
def fama_csv(dev_config, fama_filename) -> str:
    """Write a Fama-French style CSV into the config root; return its filename.

    The layout mirrors the real Ken French download: three preamble lines then
    a blank one (hence ``skiprows=4``), a header row, ``YYYYMMDD`` integer
    dates, and a two-line footer (hence ``skipfooter=2``).
    """
    body = "\n".join(
        f"{date},{mkt:>8.2f},{smb:>8.2f},{hml:>8.2f},{rmw:>8.2f},{cma:>8.2f},{rf:>8.2f}"
        for date, mkt, smb, hml, rmw, cma, rf in FAMA_ROWS
    )
    content = FAMA_PREAMBLE + ",Mkt-RF,SMB,HML,RMW,CMA,RF\n" + body + FAMA_FOOTER
    (dev_config.root_dir / fama_filename).write_text(content)
    return fama_filename
