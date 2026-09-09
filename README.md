# Portfolio risk analyser

This project is meant as a tool quantitatively analysing a given (or constructed) equity portfolio. Data is fetched primarily using the IBKR API, though support for yfinance may be added in a later version. Each module in `portfolio/src/backend/` is designed to be modular, with test suites written in `portfolio/src/backend/tests/`. Data retrieval and manipulation belongs solely to `portfolio/src/backend/data`, while all forms of analysis are performed in `portfolio/src/backend/analysis/`. Visualisations are performed using helpers in `portfolio/src/backend/vis/`, which work with direct analysis data, or data generated from `portfolio/src/backend/simulations/` and `portfolio/src/backend/backtest/`. Constants necessary for configuration reside in `portfolio/src/backend/config.py`, which retrieves a few IBKR-related values from `portfolio/.env`. The primary interface will be CLI-based using `argparse`, accessed via `portfolio/src/backend/main.py`, though a web-based UI may be added in the future.

## Example workflow

> Data from IBKR is retrieved via the `portfolio/src/backend/data` module, and cached locally in `portfolio/src/backend/cache`.
> Various metrics are computed in `portfolio/src/backend/analysis/`, such as the Sharpe ratio, Sortino ratio, etc.
> Several strategies may be simulated by `portfolio/src/backend/strategy/`, such as trend-following and mean-reversion.
> A custom portfolio may be constructed with `portfolio/src/backend/portfolio_construction/`, with an equal-weightage as the default.
> Backtesting is available through `portfolio/src/backend/backtest/`
> Analysis results, strategy performances, and backtesting results may all be visualised using helpers in `portfolio/src/backend/vis/`

Alongside the above workflow, `portfolio/src/backend/simulations/` provides helpers for forward-value projections Monte Carlo VaR or GBM random-walks.

## How to setup

The dependencies for the project can be found in `portfolio/requirements.txt`. Installation of the packages can be done by simply running
``` pip install -r requirements.txt ```
from the project root. Note that using IBKR as a data source requires host and port information in a `.env` file to be present.
Certain constants in `portfolio/src/backend/config.py` will rely on variables in `.env` to register, though they can be easily hardcoded.
