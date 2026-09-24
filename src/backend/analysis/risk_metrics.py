import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
from typing import Optional
from src.backend.config import DevConfig
from src.backend.analysis.returns import (
    ann_returns,
    cumulative_returns,
    rolling_returns,
)
from src.backend.analysis.vol import ann_volatility, rolling_volatility

__all__ = [
    "get_metrics",
    "get_correlation",
    "rolling_correlation",
    "get_covariance",
    "rolling_covariance",
    "ann_sharpe",
    "rolling_sharpe",
    "ann_sortino",
    "rolling_sortino",
    "daily_drawdowns",
    "max_daily_drawdowns",
    "min_drawdown",
    "max_drawdown",
    "rolling_drawdown",
    "ann_calmar",
    "probabilistic_sharpe",
    "expected_max_sharpe",
    "deflated_sharpe",
]


def _daily_rf(config: DevConfig) -> float:
    """Converts the annual risk free rate in config to a daily rate."""
    return ((1 + config.risk_free_rate) ** (1 / config.annualise)) - 1


def _per_period_sharpe(config: DevConfig, data: pd.Series) -> float:
    """
    Computes the per-period (non-annualised) Sharpe ratio of the given returns series.
    """
    data = data.dropna()
    volatility = data.std(ddof=1)
    if np.isclose(volatility, 0.0, atol=1e-12):
        return np.nan
    return (data.mean() - _daily_rf(config)) / volatility


def get_metrics(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes several key statistics for each column of the input data.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    metrics = data.agg(["max", "min", "mean", "median", "std", "skew", "kurtosis"])
    lower = np.quantile(data.dropna(), 0.25, axis=0)
    upper = np.quantile(data.dropna(), 0.75, axis=0)
    iqr = upper - lower
    metrics.loc["0.25"] = pd.Series(lower, index=data.columns)
    metrics.loc["0.75"] = pd.Series(upper, index=data.columns)
    metrics.loc["iqr"] = pd.Series(iqr, index=data.columns)
    metrics = metrics.T
    return metrics


def get_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the correlation matrix for a given dataframe of returns"""
    return data.corr()


def rolling_correlation(
    config: DevConfig, data: pd.DataFrame, cols: list, window: Optional[int] = None
) -> pd.DataFrame:
    """Computes the rolling correlations matrix for the specified columns of a given dataframe"""
    if window is None:
        window = config.sma_window
    if len(cols) < 2:
        raise ValueError("Cannot compute correlations for fewer than two columns.")
    if not cols:
        print(
            "No columns were specified. Computing correlations for first two columns."
        )
        cols = list(data.columns)
    rolling_corr = data[cols[0]].rolling(window).corr(data[cols[1]])
    return rolling_corr


def get_covariance(data: pd.DataFrame) -> pd.DataFrame:
    """Computes the covariance matrix for a given dataframe of returns"""
    return data.cov()


def rolling_covariance(
    config: DevConfig, data: pd.DataFrame, cols: list, window: Optional[int] = None
) -> pd.DataFrame:
    """Computes the rolling covariance matrix for the specified columns of a given dataframe"""
    if window is None:
        window = config.sma_window
    if len(cols) < 2:
        raise ValueError("Cannot compute covariance for fewer than two columns.")
    if not cols:
        print("No columns were specified. Computing covariances for first two columns.")
        cols = list(data.columns)
    rolling_cov = data[cols[0]].rolling(window).cov(data[cols[1]])
    return rolling_cov


def ann_sharpe(
    config: DevConfig,
    data: pd.Series,
) -> float:
    """Computes the Sharpe ratio for annualised data"""

    ann_ret = ann_returns(config, data)
    ann_vol = ann_volatility(config, data)
    if np.isclose(ann_vol, 0.0, atol=1e-12):
        return np.nan
    sharpe = (ann_ret - config.risk_free_rate) / ann_vol
    return sharpe


def rolling_sharpe(
    config: DevConfig, data: pd.Series, window: Optional[int] = None
) -> pd.Series:
    """Computes the rolling Sharpe ratio over a given window"""
    if window is None:
        window = config.sharpe_window
    excess_ret = data - _daily_rf(config)
    rolling_ret = rolling_returns(config, excess_ret, window)
    rolling_vol = rolling_volatility(config, excess_ret, window)
    sharpe = rolling_ret / rolling_vol
    return sharpe


def ann_sortino(config: DevConfig, data: pd.Series) -> float:
    """Computes the Sortino ratio for annualised data"""

    loss = data.copy()
    loss = loss.apply(lambda x: x if x < 0 else 0)
    ann_ret = ann_returns(config, data)
    loss_vol = ann_volatility(config, loss)
    if np.isclose(loss_vol, 0.0, atol=1e-12):
        return np.nan
    sortino = (ann_ret - config.risk_free_rate) / loss_vol
    return sortino


def rolling_sortino(
    config: DevConfig, data: pd.Series, window: Optional[int] = None
) -> pd.Series:
    """Computes the rolling Sortino ratio over a given window"""
    if window is None:
        window = config.sharpe_window
    excess_ret = data - _daily_rf(config)
    loss = data.copy()
    loss = loss.apply(lambda x: x if x < 0 else 0)
    rolling_ret = rolling_returns(config, excess_ret, window)
    loss_vol = rolling_volatility(config, loss, window)
    sortino = rolling_ret / loss_vol
    return sortino


def daily_drawdowns(config: DevConfig, data: pd.Series) -> pd.Series:
    """Computes the drawdown array for a given return series"""
    wealth = cumulative_returns(data) + 1
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1
    return drawdowns


def max_daily_drawdowns(config: DevConfig, data: pd.Series) -> pd.Series:
    """Computes the running max drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    max_daily_drawdowns = drawdowns.rolling(
        min_periods=1, window=config.annualise
    ).min()
    return max_daily_drawdowns


def max_drawdown(config: DevConfig, data: pd.Series) -> float:
    """Computes the maximum drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    max_dd = drawdowns.min()
    return max_dd


def min_drawdown(config: DevConfig, data: pd.Series) -> float:
    """Computes the maximum drawdown for a given return series"""
    drawdowns = daily_drawdowns(config, data)
    min_dd = drawdowns.max()
    return min_dd


def rolling_drawdown(
    config: DevConfig,
    data: pd.Series,
    window: Optional[int] = None,
    metric: str = "mean",
) -> pd.Series:
    """Computes the rolling drawdown for a given return series"""
    if window is None:
        window = config.sma_window
    drawdowns = daily_drawdowns(config, data)
    rolling_draw = drawdowns.rolling(window).agg(metric).dropna()
    return rolling_draw


def ann_calmar(config: DevConfig, data: pd.Series) -> float:
    """Computes the Calmar ratio for a given return series"""
    ann_ret = ann_returns(config, data)
    max_draw = max_drawdown(config, data)
    if max_draw == 0:
        return np.nan
    calmar = (ann_ret - config.risk_free_rate) / np.abs(max_draw)
    return calmar


def probabilistic_sharpe(
    config: DevConfig, data: pd.Series, benchmark_sharpe: Optional[float] = None
) -> float:
    """
    Computes the probabilistic Sharpe ratio following Bailey & Lopez de Prado.
    The probability that the true Sharpe ratio of a given series exceeds a specified
    benchmark ratio, given its length, skewness and kurtosis.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    data: pd.Series
        A daily returns series.
    benchmark_sharpe: float, Optional
        An optional benchmark sharpe, defaulting to 0.0

    Returns
    -------
    float
        The probabilistic sharpe, returning np.nan if the data has 0 volatility or the
        variance of the estimator is negative.
    """
    if benchmark_sharpe is None:
        benchmark_sharpe = 0.0
    data = data.dropna()
    n_obs = len(data)
    if n_obs < 3:
        raise ValueError(
            f"Cannot perform analysis on fewer than 3 observations: {n_obs}"
        )

    sharpe = _per_period_sharpe(config, data)
    if np.isnan(sharpe):
        return np.nan

    skewness = skew(data, bias=True)
    kurt = kurtosis(data, fisher=False, bias=True)
    denom = 1 - skewness * sharpe + (kurt - 1) / 4 * sharpe**2
    if denom <= 0:
        return np.nan

    z = (sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(
    config: DevConfig, trial_sharpes: pd.Series, n_trials: Optional[int] = None
) -> float:
    """
    Computes the expected max Sharpe across the n_trials independent trials with 0.0 true
    Sharpe.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    trial_sharpes: pd.Series
        Per period Sharpe ratios of trial runs.
    n_trials: int, Optional
        An optional number of independent trials, defaults to len(trial_sharpes)

    Returns
    -------
    float
        The per-period expected Sharpe
    """
    # Guard against inconsistent dtypes and NaN
    trial_sharpes = pd.Series(trial_sharpes, dtype=float).dropna()
    if n_trials is None:
        n_trials = len(trial_sharpes)
    if n_trials < 2:
        return 0.0
    if len(trial_sharpes) < 2:
        raise ValueError(
            f"Cannot estimate variance between fewer than 2 Sharpes: {len(trial_sharpes)}"
        )

    sharpe_std = np.sqrt(trial_sharpes.var(ddof=1))
    gamma = np.euler_gamma

    return float(
        sharpe_std
        * (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials)
            + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe(
    config: DevConfig,
    trial_returns: pd.DataFrame,
    selected: Optional[str | int] = None,
    n_trials: Optional[int] = None,
) -> float:
    """
    Computes the deflated Sharpe ratio following Bailey & Lopez de Prado. The probabilistic
    Sharpe ratio of a selected trial compared against the maximum estimated Sharpe from a
    given number of trials of pure noise.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    trial_returns: pd.DataFrame
        A dataframe containing daily returns data. Each column indicates 1 trial.
    selected: str | int, Optional
        The column (trial) to evaluate. Defaults to the trial with the highest Sharpe
    n_trials: int, Optional
        Number of independent trials, defaulting to the number of usable columns.

    Returns
    -------
    float
        The probability that the observed Sharpe exceeds the adjused benchmark. np.nan
        if no trial has non-zero volatility
    """
    aligned = trial_returns.dropna(how="any")
    if len(aligned) < len(trial_returns):
        print(f"Trials do not share all dates, using {len(aligned)} aligned rows.")

    sharpes = aligned.apply(lambda x: _per_period_sharpe(config, x))
    degen = sharpes.index[sharpes.isna()]

    if len(degen) > 0:
        print(f"Dropping 0-volatility trials: {list(degen)}")
        sharpes = sharpes.dropna()

    if sharpes.empty:
        return np.nan

    if selected is None:
        selected = sharpes.idxmax()
    elif selected not in sharpes.index:
        raise KeyError(f"Cannot select missing trial column: {selected}")

    max_sharpe = expected_max_sharpe(config, sharpes, n_trials)
    return probabilistic_sharpe(config, aligned[selected], benchmark_sharpe=max_sharpe)
