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
        window_params: Optional[dict] = None,
        train_frac: Optional[float] = None,
    ):
        self.config = config
        self.price_data = price_data
        self.volume_data = volume_data
        self.signal_fn = signal_fn
        self.weight_fn = weight_fn
        self.cost_model = cost_model
        if cost_bps is None:
            self.cost_bps = config.cost_bps
        else:
            self.cost_bps = cost_bps
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

    def run(self, signal_args: list, portfolio_args: list):
        """
        Runs the backtester.

        Parameters
        ----------
        signal_args: list
            A list of arguments to unpack into _generate_signals
        portfolio_args: list
            A list of arguments to unpack into _construct_portfolio

        """
        folds = self._generate_folds()

        oos_returns = []
        oos_costs = []
        oos_weights = []

        previous_weights = None

        for train_data, test_data in folds:
            signals = self._generate_signals(train_data, signal_args)
            weights = self._construct_portfolio(signals, portfolio_args)

            test_returns = self._backtest_fold(train_data, test_data, weights)
            orders, mkt_states = self._generate_orders(
                test_data,
                weights,
                previous_weights,  # type: ignore
            )
            costs = self._compute_costs(orders, mkt_states)

            if isinstance(costs, np.ndarray):
                fold_cost = costs.sum()  # type: ignore
            else:
                fold_cost = costs

            fold_cost_series = pd.Series(0.0, index=test_returns.index)
            fold_cost_series.iloc[0] = fold_cost

            if isinstance(weights, pd.DataFrame):
                # Final weights for that period are assumed the target weights
                target_weights = weights.iloc[-1]
            else:
                target_weights = weights

            fold_weights = np.tile(target_weights.values, (len(test_data), 1))

            oos_returns.append(test_returns)
            oos_costs.append(fold_cost_series)
            oos_weights.append(
                pd.DataFrame(
                    fold_weights, index=test_data.index, columns=target_weights.index
                )
            )

            previous_weights = target_weights

        oos_returns = pd.concat(oos_returns)
        oos_costs = pd.concat(oos_costs)
        oos_weights = pd.concat(oos_weights)
        net_returns = oos_returns - oos_costs
        results = self._evaluate(net_returns, oos_weights)

        return results

    def _generate_folds(self) -> list[tuple]:
        """
        Splits the loaded price data into training and testing datasets, then returns
        training and testing folds based off the initialised window parameters.
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
        Generates signals using self.signal_fn on the input data.

        Parameters
        ----------
        data: pd.Series | pd.DataFrame
            The returns data to compute signals from
        signal_args: list
            A list containing the relevant function parameters, dependent on self.signal_fn

        Returns
        -------
        pd.Series | pd.DataFrame
            The computed signals series or dataframe
        """
        signals = self.signal_fn(self.config, data, *signal_args)
        return signals

    def _construct_portfolio(
        self, signals: pd.Series | pd.DataFrame, portfolio_args: list
    ) -> pd.Series | pd.DataFrame:
        """
        Generates portfolio weights using the initialised weights function and input params.

        Parameters
        ----------
        signals: pd.Series | pd.DataFrame
            The signals to compute portfolio weights with
        portfolio_args: list
            A list containing the relevant function parameters, dependent on self.weight_fn

        Returns
        -------
        pd.Series | pd.DataFrame
            The compute portfolio weights series or dataframe
        """
        return self.weight_fn(self.config, signals, *portfolio_args)

    def _generate_orders(
        self,
        data: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
    ) -> tuple:
        """
        Generates a list of the necessary orders to rebalance the portfolio from the previous
        weights to the target weights. Assumes data contains price data for the relevant
        tickers, and portfolio weights are static per OOS period.

        Parameters
        ----------
        data: pd.Series | pd.DataFrame
            The price data to compute orders from.
        weights: pd.Series | pd.DataFrame
            The weights over the OOS timeframe.
        prev_weights: pd.Series, Optional
            Optional series of weights held at the end of the previous OOS timeframe.
            Defaults to an empty portfolio if None.
        """
        if isinstance(weights, pd.DataFrame):
            # Final weights for that period are assumed the target weights
            target_weights = weights.iloc[-1]
        else:
            target_weights = weights

        if prev_weights is None:
            prev_weights = pd.Series(0, index=target_weights.index)
        weights_change = target_weights - prev_weights  # type: ignore
        start_prices = data.iloc[0]
        orders = []
        mkt_states = []

        # Find the starting time
        volumes = self.volume_data.loc[data.index[0]]

        for ticker in target_weights.index:
            d_weight = weights_change[ticker]
            price = start_prices[ticker]
            quantity = d_weight / price
            orders.append(Order(ticker, quantity, price))
            mkt_states.append(MktState(price, volumes[ticker]))

        return np.array(orders), np.array(mkt_states)

    def _compute_costs(
        self, orders: float | np.ndarray, market_states: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Computes the transaction costs for given orders using self.cost_model on the input data.
        """
        return self.cost_model(self.config, orders, market_states)

    def _backtest_fold(
        self,
        train_data: pd.Series | pd.DataFrame,
        test_data: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        """
        Computes the expected returns over the given OOS data for the given weights

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

    def _evaluate(self, returns_data: pd.Series, weights: pd.DataFrame) -> dict:
        """
        Evaluates performance of the strategy by computing various metrics such as the
        Sharpe ratio, Calmar ratio, Max Drawdown, Turnover, etc.
        For details, see src/backend/analysis

        Parameters
        ----------
        returns_data: pd.Series
            The OOS net returns data to be evaluated
        weights: pd.DataFrame
            The OOS portfolio weights

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

        total_turnover = turnover(self.config, weights).sum()
        return {
            "cum_return": cum_return,
            "ann_return": annualised_returns,
            "ann_vol": annualised_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_draw,
            "calmar": calmar,
            "total_turnover": total_turnover,
        }
