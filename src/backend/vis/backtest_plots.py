from typing import Optional
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backend.config import DevConfig
from src.backend.analysis.risk_metrics import daily_drawdowns, rolling_sharpe

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_backtest_summary",
]


def plot_equity_curve(
    config: DevConfig,
    run_data: dict,
    ax,
    params_dict: Optional[dict] = None,
    benchmark: Optional[pd.Series] = None,
    show_rebalances: bool = False,
):
    """
    Plots the out-of-sample (OOS) equity curve for a backtester run.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    run_data: dict
        A dictionary containing the results from WalkForwardBacktester.runs
    ax: matplotlib.axes.Axes
        The axes to draw on
    params_dict: dict, Optional
        An optional dictionary of keyword args to pass into the plotter
    benchmark: pd.Series, Optional
        An optional series of benchmark daily returns to compare with the run results.
        Rebased to starting capital of the run for consistency
    show_rebalances: bool
        A flag to add vertical lines marking rebalance dates

    Returns
    -------
    list
        The matplotlib line artists for the plotted strategy
    """
    params_dict = {"label": "strategy", **(params_dict or {})}
    equity = run_data["equity_curve"]
    artists = ax.plot(equity.index, equity.values, **params_dict)

    if benchmark is not None:
        bench_returns = benchmark.reindex(equity.index).fillna(0.0)
        bench_equity = run_data["start_capital"] * (1 + bench_returns).cumprod()
        artists += ax.plot(
            bench_equity.index, bench_equity.values, label="benchmark", alpha=0.7
        )

    if show_rebalances:
        for date in run_data["fold_costs"].index:
            ax.axvline(date, color="black", alpha=0.2, linewidth=0.8, linestyle="--")

    ax.axhline(run_data["start_capital"], color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylabel("equity")
    ax.legend(loc="upper left")

    return artists


def plot_drawdown(
    config: DevConfig,
    run_data: dict,
    ax,
    params_dict: Optional[dict] = None,
):
    """
    Plots the drawdown of a backtester run from its running peak and labels the max
    drawdown. Params dict is fed into ax.fill_between

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    run_data: dict
        A dictionary containing the results from WalkForwardBacktester.runs
    ax: matplotlib.axes.Axes
        The axes to draw on
    params_dict: dict, Optional
        An optional dictionary of keyword args to pass into the plotter

    Returns
    -------
    PolyCollection
        The filled drawdown area
    """
    params_dict = {"color": "tab:red", "alpha": 0.35, **(params_dict or {})}
    drawdown = daily_drawdowns(config, run_data["net_returns"])
    artist = ax.fill_between(drawdown.index, drawdown.values, 0.0, **params_dict)

    ax.annotate(
        f"{drawdown.min():.1%}",
        xy=(drawdown.idxmin(), drawdown.min()),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
    ax.set_ylabel("drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    return artist


def plot_rolling_sharpe(
    config: DevConfig,
    run_data: dict,
    ax,
    params_dict: Optional[dict] = None,
    window: Optional[int] = None,
    annualised: bool = True,
):
    """
    Plots the rolling Sharpe of a backtester run. The rolling Sharpe is per-period, with
    annualised=True it is scaled by sqrt(config.annualise).

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    run_data: dict
        A dictionary containing the results from WalkForwardBacktester.runs
    ax: matplotlib.axes.Axes
        The axes to draw on
    params_dict: dict, Optional
        An optional dictionary of keyword args to pass into the plotter
    window: int, Optional
        An optional window length for computing the rolling Sharpe ratio. Defaults to
        config.sharpe_window
    annualised: bool
        A flag to annualise the rolling Sharpe ratio

    Returns
    -------
    list
        The matplotlib line artists for the plotted strategy
    """
    if window is None:
        window = config.sharpe_window

    sharpe = rolling_sharpe(config, run_data["net_returns"], window)
    label = f"Rolling Sharpe ({window}d"
    if annualised:
        sharpe = sharpe * np.sqrt(config.annualise)
        label += ", ann."
    label += ")"

    artists = ax.plot(sharpe.index, sharpe.values, **(params_dict or {}))
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_ylabel(label)

    return artists


def plot_backtest_summary(
    config: DevConfig,
    run_data: dict,
    benchmark: Optional[pd.Series] = None,
    window: Optional[int] = None,
    figsize: tuple = (10, 8),
):
    """
    Draw the equity curve, drawdown and rolling Sharpe on 3 stacked axes sharing the same
    date index. This helper creates its own figure.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    run_data: dict
        A dictionary containing the results from WalkForwardBacktester.runs
    benchmark: pd.Series, Optional
        An optional series of benchmark daily returns to compare with the run results.
        Rebased to starting capital of the run for consistency
    window: int, Optional
        An optional window length for computing the rolling Sharpe ratio. Defaults to
        config.sharpe_window
    figsize: tuple
        The figure size to create. Defaults to (10, 8)

    Returns
    -------
    tuple
        A tuple containing the generated (figure, axes)
    """
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]}
    )
    plot_equity_curve(config, run_data, axes[0], benchmark=benchmark)
    plot_drawdown(config, run_data, axes[1])
    plot_rolling_sharpe(config, run_data, axes[2], window=window)

    results = run_data["results"]
    axes[0].set_title(
        f"{run_data['signal_fn']} {run_data['signal_args']} | {run_data['weight_fn']} | "
        f"{run_data['cost_bps']} bps - Sharpe {results['sharpe']:.2f}, "
        f"Max DD {results['max_drawdown']:.1%}",
        fontsize=10,
    )

    fig.tight_layout()
    return fig, axes
