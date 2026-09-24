"""Shared fixtures for the backtest test suite.

Everything here is deterministic: the synthetic price and volume panels are
driven by ``TestConfig.random_seed`` so the walk-forward results are stable
across runs.

Two deliberately simple, hand-checkable stand-ins are provided for the
pluggable pieces of ``WalkForwardBacktester``:

* ``momentum_signal`` / ``constant_signal`` as ``signal_fn``
* ``make_flat_cost`` as ``cost_model``

Both follow the calling conventions the backtester uses internally, i.e.
``signal_fn(config, data, *signal_args)`` and
``cost_model(config, orders, mkt_states)``.
"""

import numpy as np
import pandas as pd
import pytest

from src.backend.config import TestConfig
from src.backend.backtest.costs import Order, MktState, sqrt_cost
from src.backend.backtest.walk_forward import WalkForwardBacktester
from src.backend.portfolio_construction import equal_active_weights


# ---------------------------------------------------------------------------
# Config / data
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> TestConfig:
    """A TestConfig instance with fixed, reproducible parameters."""
    return TestConfig()


@pytest.fixture
def rng(config) -> np.random.Generator:
    """A seeded random number generator for reproducible synthetic data."""
    return np.random.default_rng(config.random_seed)


@pytest.fixture
def tickers() -> list[str]:
    return ["AAA", "BBB", "CCC"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=120)


@pytest.fixture
def price_data(rng, dates, tickers) -> pd.DataFrame:
    """A (dates x tickers) panel of strictly positive close prices."""
    steps = rng.normal(0.0004, 0.012, size=(len(dates), len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def volume_data(rng, dates, tickers) -> pd.DataFrame:
    """Average daily volumes aligned to ``price_data``."""
    volumes = rng.uniform(1e6, 5e6, size=(len(dates), len(tickers)))
    return pd.DataFrame(volumes, index=dates, columns=tickers)


@pytest.fixture
def window_params() -> dict:
    """Short OOS windows so the 120-row panel yields several folds."""
    return {"window_type": "rolling", "window_len": 10}


# ---------------------------------------------------------------------------
# Pluggable strategy / cost stand-ins
# ---------------------------------------------------------------------------


def momentum_signal(config, data: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Sign of the trailing ``lookback``-period return, one column per ticker."""
    return np.sign(data.pct_change(lookback)).fillna(0.0)


def constant_signal(config, data: pd.DataFrame, value: int = 1) -> pd.DataFrame:
    """A constant signal, so the resulting weights are fully predictable."""
    return pd.DataFrame(value, index=data.index, columns=data.columns, dtype=float)


def make_flat_cost(bps: float):
    """Build a cost model charging a flat ``bps`` on traded notional.

    Matches the signature ``WalkForwardBacktester._compute_costs`` calls with,
    and ignores both ``config`` and any forwarded ``cost_bps`` so the charge is
    fixed at ``bps`` whatever the backtester or config say.
    """

    def flat_cost(config, orders, mkt_states, cost_bps=None):
        quantities = np.array([order.quantity for order in orders])
        prices = np.array([order.price for order in orders])
        return np.abs(quantities) * prices * bps / 10000

    return flat_cost


@pytest.fixture
def signal_fn():
    return momentum_signal


@pytest.fixture
def constant_signal_fn():
    return constant_signal


@pytest.fixture
def flat_cost_model():
    """Factory fixture: ``flat_cost_model(bps)`` -> a flat-bps cost model."""
    return make_flat_cost


@pytest.fixture
def weight_fn():
    return equal_active_weights


# ---------------------------------------------------------------------------
# Backtester builders
# ---------------------------------------------------------------------------


@pytest.fixture
def make_backtester(config, price_data, volume_data, signal_fn, weight_fn, window_params):
    """Factory returning a ``WalkForwardBacktester`` over the synthetic panel.

    Any keyword is forwarded to the constructor, so individual tests can vary
    one input (cost model, window type, train fraction, ...) at a time.
    """

    def _make(**overrides) -> WalkForwardBacktester:
        kwargs = {
            "config": config,
            "price_data": price_data,
            "volume_data": volume_data,
            "signal_fn": signal_fn,
            "weight_fn": weight_fn,
            "cost_model": sqrt_cost,
            "window_params": dict(window_params),
        }
        kwargs.update(overrides)
        return WalkForwardBacktester(**kwargs)

    return _make


@pytest.fixture
def backtester(make_backtester) -> WalkForwardBacktester:
    """A backtester with the default rolling-window / sqrt-cost setup."""
    return make_backtester()


# ---------------------------------------------------------------------------
# Order / market-state fixtures for the cost models
# ---------------------------------------------------------------------------


@pytest.fixture
def order() -> Order:
    """A single buy order: 1,000 shares at $50, i.e. $50,000 notional."""
    return Order(ticker="AAA", quantity=1000.0, price=50.0)


@pytest.fixture
def mkt_state() -> MktState:
    """Market state matching ``order``: 1,000,000 shares traded daily."""
    return MktState(price=50.0, daily_volume=1_000_000.0)


@pytest.fixture
def order_array() -> np.ndarray:
    """A mixed long/short/flat basket of orders."""
    return np.array(
        [
            Order("AAA", 1000.0, 50.0),
            Order("BBB", -400.0, 25.0),
            Order("CCC", 0.0, 10.0),
        ]
    )


@pytest.fixture
def mkt_state_array() -> np.ndarray:
    """Market states aligned to ``order_array``, with distinct volumes."""
    return np.array(
        [
            MktState(50.0, 1_000_000.0),
            MktState(25.0, 250_000.0),
            MktState(10.0, 500_000.0),
        ]
    )
