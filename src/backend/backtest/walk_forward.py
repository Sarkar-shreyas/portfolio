import pandas as pd
import numpy as np
from typing import Optional, Callable

from src.backend.config import DevConfig
from src.backend.analysis import *
from src.backend.strategy import *
from src.backend.portfolio_construction import *
from src.backend.simulations import *
from src.backend.backtest.costs import Order, MktState


class WalkForwardBacktester:
    """
    A class that simulates portfolio construction and tests strategies on historical price
    data. Single asset and multi-asset portfolios are testable.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    price_data: pd.Series | pd.DataFrame
        The close price data to train and test strategies with
    volume_data:
        The volume data for the given price data
    signal_fn: Callable
        The function to generate signals with.
        For details, see src/backend/strategy
    weight_fn: Callable
        The function to generate portfolio weights with.
        For details, see src/backend/portfolio_construction
    cost_model: Class
        The model to compute transaction costs with.
        For details, see src/backend/backtest/costs.py
    cost_bps: float, Optional
        The cost basis points for an order. Defaults to 10.0 bps.
    start_capital: float, Optional
        The starting capital for the portfolio. Defaults to 100000.0
    window_params: dict, Optional
        The window types and lengths to use for training and testing data. Defaults to the
        config value. Start date is assumed to be the oldest date in the prices data.
    train_frac: float, Optional
        The fraction of data to be used for training. Defaults to the config value of 0.7
    """

    def __init__(
        self,
        config: DevConfig,
        price_data: pd.Series | pd.DataFrame,
        volume_data: pd.Series | pd.DataFrame,
        signal_fn: Callable,
        weight_fn: Callable,
        cost_model: Callable,
        cost_bps: Optional[float] = None,
        start_capital: Optional[float] = None,
        window_params: Optional[dict] = None,
        train_frac: Optional[float] = None,
    ):
        self.config = config
        if isinstance(price_data, pd.Series):
            self.price_data = price_data.to_frame()  # type: ignore
        else:
            self.price_data = price_data
        if isinstance(volume_data, pd.Series):
            self.volume_data = volume_data.to_frame()  # type: ignore
        else:
            self.volume_data = volume_data
        self.signal_fn = signal_fn
        self.weight_fn = weight_fn
        self.cost_model = cost_model
        if cost_bps is None:
            self.cost_bps = config.cost_bps
        else:
            self.cost_bps = cost_bps
        if start_capital is None:
            self.start_capital = config.start_capital
        else:
            self.start_capital = start_capital
        if window_params is None:
            self.window_params = {
                "window_type": config.window_type,
                "window_freq": config.window_freq,
                "window_len": config.window_len,
            }
        else:
            self.window_params = window_params
        if train_frac is None:
            self.train_frac = config.train_frac
        else:
            self.train_frac = train_frac
        self.test_frac = 1 - self.train_frac
        self.runs = {}
        self.run_count = len(self.runs)

    def run(self, signal_args: list, portfolio_args: list) -> dict:
        """
        Runs the backtester. Carries holdings in shares, weights drift with prices in
        between rebalances. Each fold trades only the difference between drifted weights
        and the new target.

        Parameters
        ----------
        signal_args: list
            A list of arguments to unpack into _generate_signals
        portfolio_args: list
            A list of arguments to unpack into _construct_portfolio

        Returns
        -------
        dict
            A dictionary containing evaluated metrics on the backtest results
        """
        folds = self._generate_folds()
        if not folds:
            raise ValueError(
                f"No folds were generated. Train frac: {self.train_frac}, Test frac: {self.test_frac}"
            )

        current_capital = self.start_capital
        current_shares = pd.Series(0.0, index=self.price_data.columns)
        cash = self.start_capital

        target_share_rows = []
        trade_share_rows = []
        trade_price_rows = []
        fold_capital = []
        oos_weights = []
        oos_equity = []
        trade_dates = []
        fold_costs = []
        fold_turnovers = []
        cash_rebalance = []
        total_turnover = 0.0

        for train_data, test_data in folds:
            signals = self._generate_signals(train_data, signal_args)
            weights = self._construct_portfolio(signals, portfolio_args)
            if isinstance(weights, pd.DataFrame):
                # Final weights for that period are assumed the target weights
                target_weights = weights.loc[train_data.index[-1]]
            else:
                target_weights = weights

            target_weights = target_weights.reindex(self.price_data.columns).fillna(0.0)

            trade_date = test_data.index[0]
            open_prices = test_data.iloc[0]

            target_shares = target_weights * current_capital / open_prices
            trade_shares = target_shares - current_shares

            orders, mkt_states = self._generate_orders(
                trade_date,
                open_prices,
                trade_shares,
            )

            fold_cost = float(np.sum(self._compute_costs(orders, mkt_states)))
            if fold_cost >= current_capital:
                raise ValueError(
                    f"Transaction costs on {trade_date} exceed the available capital: {fold_cost} >= {current_capital}"
                )

            cash -= (trade_shares * open_prices).sum() + fold_cost

            position_vals = test_data.mul(target_shares, axis=1)
            fold_equity = position_vals.sum(axis=1) + cash
            fold_turnover = (trade_shares.abs() * open_prices).sum() / current_capital
            total_turnover += fold_turnover
            target_share_rows.append(target_shares)
            trade_share_rows.append(trade_shares)
            trade_price_rows.append(open_prices)
            fold_capital.append(current_capital)
            cash_rebalance.append(cash)
            trade_dates.append(trade_date)
            fold_costs.append(fold_cost)
            fold_turnovers.append(fold_turnover)
            oos_equity.append(fold_equity)
            oos_weights.append(position_vals.div(fold_equity, axis=0))
            current_shares = target_shares
            current_capital = fold_equity.iloc[-1]

        oos_weights = pd.concat(oos_weights)
        equity = pd.concat(oos_equity)
        target_shares_df = pd.DataFrame(target_share_rows, index=trade_dates)
        trade_shares_df = pd.DataFrame(trade_share_rows, index=trade_dates)
        trade_prices_df = pd.DataFrame(trade_price_rows, index=trade_dates)
        net_returns = equity.pct_change()
        net_returns.iloc[0] = equity.iloc[0] / self.start_capital - 1
        results = self._evaluate(net_returns, oos_weights, total_turnover)
        results["final_equity"] = equity.iloc[-1]

        self.run_count += 1
        self.runs[self.run_count] = {
            "signal_args": list(signal_args),
            "portfolio_args": list(portfolio_args),
            "signal_fn": getattr(self.signal_fn, "__name__", repr(self.signal_fn)),
            "weight_fn": getattr(self.weight_fn, "__name__", repr(self.weight_fn)),
            "cost_model": getattr(self.cost_model, "__name__", repr(self.cost_model)),
            "cost_bps": self.cost_bps,
            "start_capital": self.start_capital,
            "window_params": self.window_params.copy(),
            "train_frac": self.train_frac,
            "target_shares": target_shares_df,
            "trade_shares": trade_shares_df,
            "trade_prices": trade_prices_df,
            "fold_capital": pd.Series(fold_capital, index=trade_dates),
            "cash_rebalance": pd.Series(cash_rebalance, index=trade_dates),
            "fold_costs": pd.Series(fold_costs, index=trade_dates),
            "fold_turnovers": pd.Series(fold_turnovers, index=trade_dates),
            "equity_curve": equity,
            "oos_weights": oos_weights,
            "net_returns": net_returns,
            "results": results.copy(),
        }

        return results

    def clear_runs(self, num_runs: Optional[int] = None):
        """
        Clears older runs to prevent excessive memory usage. If the number of runs is
        unspecified, data from all runs are deleted.
        """
        if num_runs is None:
            self.run_count = 0
            self.runs = {}
        else:
            for key in sorted(self.runs)[:num_runs]:
                del self.runs[key]

    def _generate_folds(self) -> list[tuple]:
        """
        Splits the loaded price data into training and testing datasets, then returns
        a list of training and testing folds based off the initialised window parameters.
        Each argument in the list is a tuple (train_fold, test_fold).
        """
        n_obs = len(self.price_data)
        cutoff_index = int(len(self.price_data) * self.train_frac)

        test_start = cutoff_index
        test_len = self.window_params["window_len"]
        folds = []
        while test_start < n_obs:
            test_end = min(test_start + test_len, n_obs)

            if self.window_params["window_type"] == "expanding":
                train_start = 0
            elif self.window_params["window_type"] == "rolling":
                train_start = max(0, test_start - cutoff_index)
            else:
                raise ValueError(
                    f"Unknown window type: {self.window_params['window_type']}"
                )

            train_fold = self.price_data.iloc[train_start:test_start]
            test_fold = self.price_data.iloc[test_start:test_end]

            folds.append((train_fold, test_fold))
            test_start = test_end

        return folds

    def _generate_signals(
        self, data: pd.Series | pd.DataFrame, signal_args: list
    ) -> pd.Series | pd.DataFrame:
        """
        Unpacks signal_args into self.signal_fn and generates a signals series for the
        input data.
        """
        signals = self.signal_fn(self.config, data, *signal_args)
        return signals

    def _construct_portfolio(
        self, signals: pd.Series | pd.DataFrame, portfolio_args: list
    ) -> pd.Series | pd.DataFrame:
        """
        Unpacks portfolio_args into self.weight_fn and generates a weights series for
        the input data.
        """
        return self.weight_fn(self.config, signals, *portfolio_args)

    def _generate_orders(
        self,
        trade_date: pd.Timestamp,
        prices: pd.Series,
        trade_shares: pd.Series,
    ) -> tuple:
        """
        Builds the necessary orders and corresponding market states for the rebalancing.

        Parameters
        ----------
        trade_date: pd.Timestamp
            The date the trade will execute on
        prices: pd.Series
            The prices of each ticker at the execution date (first bar of OOS fold)
        trade_shares: pd.Series
            The number of shares to be traded per ticker. Negative values indicate shorts

        Returns
        -------
        tuple
            An array of Orders and MktState objects
        """
        orders = []
        mkt_states = []

        # Find the starting time
        volumes = self.volume_data.loc[trade_date]

        for ticker in trade_shares.index:
            price = prices[ticker]
            orders.append(Order(ticker, trade_shares[ticker], price))
            mkt_states.append(MktState(price, volumes[ticker]))

        return np.array(orders), np.array(mkt_states)

    def _compute_costs(
        self, orders: float | np.ndarray, market_states: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Computes the transaction costs for given orders using self.cost_model on the input data.
        """
        return self.cost_model(
            self.config, orders, market_states, cost_bps=self.cost_bps
        )

    def _daily_rebalanced_backtest(
        self,
        train_data: pd.Series | pd.DataFrame,
        test_data: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        """
        ***Currently not in use. Kept for reference***

        Computes the expected returns over the given OOS data for the given weights.
        Uses a daily-rebalanced portfolio.

        Parameters
        ----------
        train_data: pd.Series | pd.DataFrame
            The training data preceding the test fold
        test_data: pd.Series | pd.DataFrame
            The test fold data
        weights: pd.Series | pd.DataFrame
            Portfolio weights generated using the training data

        Returns
        -------
        pd.Series
            Daily portfolio returns over the OOS test period.
        """
        if isinstance(weights, pd.DataFrame):
            # Final weights for that period are assumed the target weights
            target_weights = weights.iloc[-1]
        else:
            target_weights = weights

        # Prepend the final training data so initial test data returns are not NaN
        prices = pd.concat([train_data.tail(1), test_data])

        # Discard the first row which belongs to the previous timeperiod
        returns_data = prices.pct_change().iloc[1:]
        port_returns = returns_data.mul(target_weights, axis=1).sum(axis=1)
        return port_returns

    def _evaluate(
        self, returns_data: pd.Series, weights: pd.DataFrame, total_turnover: float
    ) -> dict:
        """
        Evaluates performance of the strategy. For details, see src/backend/analysis

        Parameters
        ----------
        returns_data: pd.Series
            The OOS net returns data to be evaluated
        weights: pd.DataFrame
            The OOS portfolio weights
        total_turnover: float
            The traded turnover accumulated at each rebalance.
        Returns
        -------
        dict
            A dictionary containing the computed statistics
        """
        cum_return = cumulative_returns(returns_data).iloc[-1]
        annualised_returns = ann_returns(self.config, returns_data)
        annualised_vol = ann_volatility(self.config, returns_data)

        sharpe = ann_sharpe(self.config, returns_data)
        sortino = ann_sortino(self.config, returns_data)
        max_draw = max_drawdown(self.config, returns_data)
        calmar = ann_calmar(self.config, returns_data)

        mean_gross_exposure = avg_gross_exposure(self.config, weights)
        mean_net_exposure = avg_net_exposure(self.config, weights)
        return {
            "cum_return": cum_return,
            "ann_return": annualised_returns,
            "ann_vol": annualised_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_draw,
            "calmar": calmar,
            "total_turnover": total_turnover,
            "avg_gross_exposure": mean_gross_exposure,
            "avg_net_exposure": mean_net_exposure,
        }
