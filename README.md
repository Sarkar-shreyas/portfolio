# Portfolio risk analyser

A Python library for quantitative portfolio analysis: market data, return and risk
statistics, signal generation, portfolio construction, walk-forward backtesting with
transaction costs, and Monte Carlo simulation.
The intended use is from a Jupyter notebook: import the pieces you need and compose them
into an analysis. Every module is a set of plain functions (plus one class, the backtester),
so each stage can be used on its own or wired into the full pipeline below.

## The core workflow

```
market data
    -> returns
    -> signals
    -> portfolio weights
    -> walk-forward backtest
    -> transaction costs
    -> portfolio returns
    -> risk metrics
```

```python
import pandas as pd

from src.backend.config import DevConfig
from src.backend.analysis import ann_returns, ann_volatility, ann_sharpe, max_drawdown, est_var
from src.backend.strategy import sma_crossover_portfolio
from src.backend.portfolio_construction import equal_active_weights
from src.backend.backtest.costs import linear_cost
from src.backend.backtest.walk_forward import WalkForwardBacktester

config = DevConfig()

# close and volume are (dates x tickers) frames sharing one DatetimeIndex
backtester = WalkForwardBacktester(
    config=config,
    price_data=close,
    volume_data=volume,
    signal_fn=sma_crossover_portfolio,
    weight_fn=equal_active_weights,
    cost_model=linear_cost,
    cost_bps=10,
    window_params={"window_type": "rolling", "window_freq": "D", "window_len": 21},
    train_frac=0.5,
)

results = backtester.run(signal_args=[5, 21], portfolio_args=[])
net_returns = backtester.runs[1]["net_returns"]

ann_returns(config, net_returns)
ann_volatility(config, net_returns)
ann_sharpe(config, net_returns)
max_drawdown(config, net_returns)
est_var(config, net_returns)
```

`signal_args` and `portfolio_args` are unpacked into `signal_fn` and `weight_fn`, so
swapping a stage is a one-line change. Inverse-volatility weighting, for example, takes
a volatility frame as its extra argument:

```python
from src.backend.analysis import simple_returns, rolling_volatility
from src.backend.portfolio_construction import inverse_volatility_weighted

vols = rolling_volatility(config, simple_returns(close), 21)
backtester = WalkForwardBacktester(..., weight_fn=inverse_volatility_weighted)
results = backtester.run(signal_args=[5, 21], portfolio_args=[vols])
```

## What the backtester gives you

`run()` returns a summary dict:

| | |
|---|---|
| `cum_return`, `ann_return`, `ann_vol` | headline performance |
| `sharpe`, `sortino`, `calmar` | risk-adjusted ratios |
| `max_drawdown`, `final_equity` | drawdown and closing equity |
| `total_turnover`, `avg_gross_exposure`, `avg_net_exposure` | trading and exposure |

Each call also appends a full record to `backtester.runs[n]`, keyed by run number, so
several parameterisations can be compared without re-running:

| | |
|---|---|
| `net_returns`, `equity_curve`, `oos_weights` | the out-of-sample series |
| `target_shares`, `trade_shares`, `trade_prices` | the blotter, one row per rebalance |
| `fold_capital`, `fold_costs`, `fold_turnovers`, `cash_rebalance` | per-rebalance detail |
| `signal_fn`, `weight_fn`, `cost_model`, `cost_bps`, `window_params`, `train_frac`, `start_capital` | the settings that produced it |

`clear_runs()` drops the history when it gets large.

## Contracts

The backtester defines the shape every pluggable piece has to satisfy. The three
callables it takes are called as:

```python
signal_fn(config, price_panel, *signal_args)   -> (dates x tickers) DataFrame
weight_fn(config, signals, *portfolio_args)    -> (dates x tickers) DataFrame
cost_model(config, orders, mkt_states, cost_bps=...)  -> float | np.ndarray
```

A few invariants matter enough to state explicitly:

- **Panels, not series.** `price_data` and `volume_data` are `(dates x tickers)` frames
  on the same DatetimeIndex. A Series is promoted to a single-column frame.
- **Weights must preserve the signal index.** The backtester reads the weight row at the
  last training date. A weight function that widens or reorders the index hands it the
  wrong row.
- **Signals must be causal.** A value at date `t` may only use data up to and including
  `t`. The backtester supplies the one-bar execution gap itself, so signal functions
  should not shift their own output.
- **Execution is at the first out-of-sample close.** Holdings are carried in shares and
  drift with prices between rebalances; each fold trades only the difference between the
  drifted book and the new target. The book is marked to market on the rebalance bar
  before it is resized, so the realised gross exposure matches the target up to the
  transaction cost charged on that bar.

## Layout

| Path | Contents |
|---|---|
| `src/backend/config.py` | `Config`, `DevConfig`, `TestConfig` dataclasses holding every default constant |
| `src/backend/data/` | IBKR and AlphaVantage retrieval, Fama-French factor loading, CSV/JSON caching, timeseries cleaning |
| `src/backend/analysis/` | returns, volatility (incl. GARCH), correlation and covariance, Sharpe/Sortino/Calmar, drawdowns, VaR/CVaR, factor regressions, PCA |
| `src/backend/strategy/` | signal generation: SMA/EMA crossover and RSI mean reversion, single-asset and portfolio variants |
| `src/backend/portfolio_construction/` | equal-weight, long/short split, inverse-volatility weighting, benchmarks, turnover and exposure helpers |
| `src/backend/backtest/` | `WalkForwardBacktester`, `Order`/`MktState`, `linear_cost` and `sqrt_cost` |
| `src/backend/simulations/` | Monte Carlo VaR (single asset and portfolio), additive/multiplicative random walks, GBM |
| `src/backend/vis/` | candlestick, timeseries, histogram, QQ and heatmap helpers |
| `src/backend/tests/` | the test suite, mirroring the package layout |

Every analysis function takes a config instance as its first argument and reads its
defaults from it, so behaviour is changed by editing `config.py` or passing an override
rather than by threading constants through call sites.

## Setup

Python 3.14. From the project root:

```
pip install -r requirements.txt
```

Configuration constants live in `src/backend/config.py`. A few are read from a `.env`
file at the project root:

```
ROOT=/absolute/path/to/portfolio
IB_HOST=127.0.0.1
IB_PORT=4001
IB_CLIENT_ID=1
ALPHA_VANTAGE_KEY=your_key
```

`ROOT` is used to locate the cache and trial-data directories. The IBKR and AlphaVantage
values are only needed for live data retrieval; they can be hardcoded in `config.py`
instead, and nothing outside `src/backend/data/` requires them.

Data fetched from IBKR is cached under `src/backend/cache/`, and the fetch helpers return
the cached copy when one exists rather than reconnecting.

## Tests

```
pytest
```

780 tests, all deterministic and offline — external data sources are mocked, and
synthetic data is seeded from `TestConfig.random_seed`.

| Suite | Tests |
|---|---|
| `test_analysis/` | 107 |
| `test_backtest/` | 201 |
| `test_data/` | 165 |
| `test_portfolio_construction/` | 103 |
| `test_simulations/` | 96 |
| `test_strategy/` | 77 |
| `test_integration/` | 31 |

`test_integration/` is the one suite that composes the whole library rather than testing
a function in isolation. It runs the workflow above end to end on synthetic data and
checks the properties that only exist between modules: that index and column alignment
survives every stage, that the walk-forward process has no look-ahead, that transaction
costs reach the P&L at the right magnitude, and that the portfolio return on a
non-trading day equals the previous day's weights dotted with that day's asset returns.
Start there when changing anything that crosses a module boundary.

The root `conftest.py` exists only to anchor pytest's path insertion at the project root
so `from src.backend... import ...` resolves however pytest is invoked.

## Current limitations

- Not packaged — there is no `pyproject.toml`, so the library is used from the project
  root rather than installed. Imports are absolute from `src.backend`.
- IBKR is the only live equity data source. `yfinance` support may be added later.
- `cov_shrinkage` in `analysis/pca_risk.py` passes a covariance matrix to scikit-learn's
  `LedoitWolf`, which expects a sample matrix; its output is not currently meaningful.
  `pca_analysis` works correctly on a plain covariance matrix from `get_covariance`.
- The backtester models no margin or financing cost, so levered and short books carry
  negative cash without being charged for it.
- `window_params["window_freq"]` is accepted but unused; `window_len` is a number of
  bars, not a calendar period.
- `portfolio_construction/optimised_weights.py` is a placeholder for mean-variance and
  risk-parity construction.
