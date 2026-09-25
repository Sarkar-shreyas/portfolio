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
from src.backend.data import to_panel
from src.backend.analysis import ann_returns, ann_volatility, ann_sharpe, max_drawdown, est_var
from src.backend.strategy import sma_crossover_portfolio
from src.backend.portfolio_construction import equal_active_weights
from src.backend.backtest import WalkForwardBacktester, linear_cost

config = DevConfig()

# frames: {ticker: OHLCV DataFrame}, e.g. one clean_timeseries output per ticker.
# to_panel stitches one field into a (dates x tickers) frame on the union of dates.
close = to_panel(frames, "close")
volume = to_panel(frames, "volume")

backtester = WalkForwardBacktester(
    config=config,
    price_data=close,
    volume_data=volume,
    signal_fn=sma_crossover_portfolio,
    weight_fn=equal_active_weights,
    cost_model=linear_cost,
    cost_bps=10,
    window_params={"window_type": "rolling", "window_len": 21},
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

## Plotting a run

`vis/backtest_plots.py` charts a run record directly:

```python
from src.backend.vis import plot_backtest_summary, plot_equity_curve

run = backtester.runs[1]
fig, axes = plot_backtest_summary(config, run, benchmark=spx_returns)
fig.savefig("docs/images/backtest_summary.png", dpi=150)
```

`plot_backtest_summary` draws three stacked panels on a shared date axis: the
out-of-sample equity curve (with an optional benchmark rebased to the starting capital),
the drawdown from the running peak with its deepest point labelled, and the rolling
Sharpe ratio. The title carries the run's settings, Sharpe and maximum drawdown. The
panels are also available on their own — `plot_equity_curve`, `plot_drawdown` and
`plot_rolling_sharpe` — each drawing on an axes you pass in, so they compose into your
own layouts:

```python
fig, ax = plt.subplots()
plot_equity_curve(config, backtester.runs[1], ax, params_dict={"label": "5/21"})
plot_equity_curve(config, backtester.runs[2], ax, params_dict={"label": "10/50"})
```

The rolling Sharpe is annualised by `sqrt(config.annualise)` by default
(`annualised=False` plots the per-period ratio), with the window defaulting to
`config.sharpe_window`. Unlike the other `vis` helpers, which take `(data, ax,
params_dict)`, these take `config` first, because they compute analysis quantities from
it.

## Judging a parameter search

The best of many backtests looks better than it is. `deflated_sharpe` (Bailey & López de
Prado) corrects the winning run's Sharpe ratio for the number of runs tried, their
spread, and the skewness, kurtosis and length of its return series, and returns the
probability that its true Sharpe beats what selection alone would produce:

```python
from src.backend.analysis import deflated_sharpe, probabilistic_sharpe

for short, long in [(5, 21), (5, 63), (10, 50), (20, 100)]:
    backtester.run(signal_args=[short, long], portfolio_args=[])

trials = pd.DataFrame({n: run["net_returns"] for n, run in backtester.runs.items()})

deflated_sharpe(config, trials)               # the best run, deflated for 4 trials
deflated_sharpe(config, trials, selected=2)   # a specific run
probabilistic_sharpe(config, trials[1])       # one run, no deflation (benchmark 0)
```

Only the dates every trial shares are used, and zero-volatility runs are dropped. `n_trials`
should count *independent* trials; close parameter variations are highly correlated, so
pass a smaller `n_trials` rather than letting the raw run count over-deflate.

## Factor regressions

`capm_regression`, `fama_french_three`, `fama_french_five` and `regress` each return
`(summary, fitted)`: a coefficient table (`coef`, `std err`, `t`, `P>|t|`, `0.025`,
`0.975`) and a frame of actual, fitted and residual values.

Standard errors default to Newey-West HAC, because daily factor returns are serially
correlated and heteroskedastic and plain OLS understates the uncertainty. The coefficients
are the same either way; only the inference columns change.

```python
summary, fitted = fama_french_three(config, data)                          # HAC, lags by rule
summary, fitted = fama_french_three(config, data, maxlags=10)              # HAC, fixed lags
summary, fitted = fama_french_three(config, data, cov_type="nonrobust")    # classical OLS
summary.attrs    # {"cov_type": "HAC", "maxlags": 4}
```

The defaults are `config.regression_cov_type` (`"HAC"`) and `config.hac_maxlags`. When no
lag length is given, the Newey-West rule `floor(4 * (T / 100) ** (2 / 9))` is used, which
is 4 lags for a year of daily data. Any other statsmodels `cov_type` (e.g. `"HC0"`) is
passed through.

## Value at Risk

The VaR helpers use two sign conventions, so check which one you have before comparing or
plotting them together:

- `est_var` and `cond_var` are historical estimates over the whole series. They return a
  **return quantile**, so a loss is negative (e.g. `-0.04` for a 4% one-day VaR).
- `norm_parametric_var` and `t_parametric_var` are rolling estimates over `window` bars
  (default `config.sma_window`). They return a **loss threshold** as a positive number,
  indexed like the input.

`t_parametric_var` scales the Student-t so that its variance matches the rolling sample
variance. A t with scale `sigma` has variance `sigma² · dof / (dof − 2)`, so the scale is
divided by `sqrt(dof / (dof − 2))`.

Both functions take the lower-tail quantile of the fitted distribution and negate it, so the
result is `z·sigma − mu`: a positive drift lowers the VaR.

To check a rolling estimate against what actually happened, compare each day's return
with the previous day's estimate. At 95% confidence, about 5% of days should fall below it:

```python
var = t_parametric_var(config, returns, window=63).shift(1)
breach_rate = (returns < -var)[var.notna()].mean()
```

## Case study

[`notebooks/trend_following_case_study.ipynb`](notebooks/trend_following_case_study.ipynb)
applies the library to real IBKR data: eight technology stocks, daily bars from September
2021 to September 2026. It has two parts:

- **Portfolio risk profile.** Returns and their distribution, volatility (rolling, EWMA and
  GARCH), drawdowns, historical, Monte Carlo and parametric VaR/CVaR with a breach test,
  CAPM beta, and Fama-French five-factor exposures for the portfolio and each stock.
- **Strategy backtest.** A walk-forward SMA crossover against SPX and against a long-only
  book run through the same backtester and cost model. It also covers sensitivity to
  transaction costs and a deflated Sharpe ratio across a grid of 14 window pairs.

It runs from the cached data in `src/backend/cache/`, so no IBKR connection is needed
once that is populated. The notebook's summary and conclusions sections have the findings.

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
- **`sqrt_cost` needs volume wherever it trades.** An order with no quantity always costs 0,
  but a non-zero order against zero, negative or missing daily volume raises a `ValueError`
  naming the tickers, because the square-root impact model has no answer there. Fill or
  drop gaps in the volume panel before backtesting with it.

## Layout

| Path | Contents |
|---|---|
| `src/backend/config.py` | `Config`, `DevConfig`, `TestConfig` dataclasses holding every default constant |
| `src/backend/data/` | IBKR and AlphaVantage retrieval, Fama-French factor loading, CSV/JSON caching, timeseries cleaning, `to_panel` for `(dates x tickers)` frames |
| `src/backend/analysis/` | returns, volatility (incl. GARCH), correlation and covariance, Ledoit-Wolf shrinkage, Sharpe/Sortino/Calmar, probabilistic and deflated Sharpe, drawdowns, VaR/CVaR, factor regressions with HAC standard errors, PCA |
| `src/backend/strategy/` | signal generation: SMA/EMA crossover and RSI mean reversion, single-asset and portfolio variants |
| `src/backend/portfolio_construction/` | equal-weight, long/short split, inverse-volatility weighting, benchmarks, turnover and exposure helpers |
| `src/backend/backtest/` | `WalkForwardBacktester`, `Order`/`MktState`, `linear_cost` and `sqrt_cost` |
| `src/backend/simulations/` | Monte Carlo VaR (single asset and portfolio), additive/multiplicative random walks, GBM |
| `src/backend/vis/` | backtest run charts (equity curve, drawdown, rolling Sharpe, three-panel summary), candlestick, timeseries, histogram, QQ and heatmap helpers |
| `src/backend/tests/` | the test suite, mirroring the package layout |
| `notebooks/` | worked examples; currently the trend-following case study |

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

907 tests, all deterministic and offline — external data sources are mocked, and
synthetic data is seeded from `TestConfig.random_seed`.

| Suite | Tests |
|---|---|
| `test_analysis/` | 168 |
| `test_backtest/` | 208 |
| `test_data/` | 169 |
| `test_portfolio_construction/` | 103 |
| `test_simulations/` | 96 |
| `test_strategy/` | 77 |
| `test_vis/` | 53 |
| `test_integration/` | 33 |

`test_vis/` runs on matplotlib's non-interactive Agg backend and asserts on what each
helper drew — the data behind every line, fill and image, plus labels and counts — rather
than comparing rendered images.

`test_integration/` is the one suite that composes the whole library rather than testing
a function in isolation. It runs the workflow above end to end on synthetic data and
checks the properties that only exist between modules: that index and column alignment
survives every stage, that the walk-forward process has no look-ahead, that transaction
costs reach the P&L at the right magnitude, and that the portfolio return on a
non-trading day equals the previous day's weights dotted with that day's asset returns.
It also feeds the backtester's run history into `deflated_sharpe` and its returns into
`capm_regression`, checking that both keep the backtest's dates. Start there when changing
anything that crosses a module boundary.

The root `conftest.py` exists only to anchor pytest's path insertion at the project root
so `from src.backend... import ...` resolves however pytest is invoked.

## Current limitations

- Not packaged — there is no `pyproject.toml`, so the library is used from the project
  root rather than installed. Imports are absolute from `src.backend`.
- IBKR is the only live equity data source. `yfinance` support may be added later.
- The backtester models no margin or financing cost, so levered and short books carry
  negative cash without being charged for it.
- `window_params["window_len"]` is a number of bars, not a calendar period.
- `to_panel` joins tickers on the union of their dates, so a ticker missing a day
  leaves a NaN in that row; the backtester does not fill these for you.
- `clean_timeseries` expects IBKR's column layout for frames without a DatetimeIndex.
- `portfolio_construction/optimised_weights.py` is a placeholder for mean-variance and
  risk-parity construction.
