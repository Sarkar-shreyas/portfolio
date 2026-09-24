"""Fixtures for the end-to-end reference-workflow integration test.

Everything here is deterministic and local: the OHLCV panel is generated from
``TestConfig.random_seed`` so the whole pipeline is reproducible, and no test in
this package touches IBKR, AlphaVantage, the cache directory or any other
external source.

The fixtures deliberately start one step earlier than the backtester needs -- at
a per-ticker OHLCV panel -- so the integration test exercises the same shape of
data a notebook would load, and has to derive the close/volume panels the
library expects.

Fixtures are module-scoped because the pipeline is built once and then inspected
from several angles. Tests that perturb the inputs copy them first.
"""

import numpy as np
import pandas as pd
import pytest

from src.backend.config import TestConfig
from src.backend.backtest.costs import linear_cost
from src.backend.backtest.walk_forward import WalkForwardBacktester
from src.backend.portfolio_construction.dynamic_weights import equal_active_weights
from src.backend.strategy.trend_following import sma_crossover_portfolio


UNIVERSE = ["AAPL", "AMD", "MSFT", "NVDA"]
N_OBS = 260


@pytest.fixture(scope="module")
def config() -> TestConfig:
    """A TestConfig instance with fixed, reproducible parameters."""
    return TestConfig()


@pytest.fixture(scope="module")
def tickers() -> list[str]:
    return list(UNIVERSE)


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    """A business-day index long enough for a 21-day warm-up plus 7 OOS folds."""
    return pd.bdate_range("2022-01-03", periods=N_OBS)


@pytest.fixture(scope="module")
def ohlcv_panel(config, dates, tickers) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames, as a notebook would hold after loading data.

    Close prices follow a geometric random walk; the other bars are derived from
    close so that ``low <= open, close <= high`` holds by construction. Only
    close and volume are consumed downstream, but the full set is present so the
    workflow starts where the reference workflow says it starts.
    """
    rng = np.random.default_rng(config.random_seed)
    panel = {}
    for ticker in tickers:
        steps = rng.normal(0.0004, 0.012, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(steps))
        spread = np.abs(rng.normal(0.0, 0.004, size=len(dates))) * close
        open_ = close * (1 + rng.normal(0.0, 0.002, size=len(dates)))
        panel[ticker] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + spread,
                "low": np.minimum(open_, close) - spread,
                "close": close,
                "volume": rng.uniform(1e6, 5e6, size=len(dates)),
            },
            index=dates,
        )
    return panel


@pytest.fixture(scope="module")
def close_panel(ohlcv_panel, tickers) -> pd.DataFrame:
    """A (dates x tickers) close-price panel, as the backtester expects."""
    return pd.concat(
        {ticker: ohlcv_panel[ticker]["close"] for ticker in tickers}, axis=1
    )[tickers]


@pytest.fixture(scope="module")
def volume_panel(ohlcv_panel, tickers) -> pd.DataFrame:
    """A (dates x tickers) volume panel aligned to ``close_panel``."""
    return pd.concat(
        {ticker: ohlcv_panel[ticker]["volume"] for ticker in tickers}, axis=1
    )[tickers]


@pytest.fixture(scope="module")
def window_params() -> dict:
    """Short OOS windows so the panel yields several rebalances."""
    return {"window_type": "rolling", "window_len": 21}


@pytest.fixture(scope="module")
def signal_windows() -> tuple[int, int]:
    """The (short, long) SMA windows used throughout the workflow."""
    return (5, 21)


@pytest.fixture(scope="module")
def train_frac() -> float:
    return 0.5


@pytest.fixture(scope="module")
def make_backtester(
    config, close_panel, volume_panel, window_params, train_frac
) -> callable:
    """Factory for a backtester over the synthetic panel.

    Any keyword is forwarded to the constructor so a test can vary one input --
    the cost level, the weight function, the price data -- at a time.
    """

    def _make(**overrides) -> WalkForwardBacktester:
        kwargs = {
            "config": config,
            "price_data": close_panel,
            "volume_data": volume_panel,
            "signal_fn": sma_crossover_portfolio,
            "weight_fn": equal_active_weights,
            "cost_model": linear_cost,
            "window_params": dict(window_params),
            "train_frac": train_frac,
            "cost_bps": 10.0,
        }
        kwargs.update(overrides)
        return WalkForwardBacktester(**kwargs)

    return _make
