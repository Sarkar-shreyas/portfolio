"""Tests for ``src.backend.vis.backtest_plots``.

Each helper must draw exactly what the run record (or the analysis function it
delegates to) says, on the axes it was given, without creating figures of its own.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.ticker import PercentFormatter

from src.backend.analysis.risk_metrics import max_drawdown, rolling_sharpe
from src.backend.vis.backtest_plots import (
    plot_backtest_summary,
    plot_drawdown,
    plot_equity_curve,
    plot_rolling_sharpe,
)


def _dates(line) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(line.get_xdata())


def _vertical_lines(ax) -> list:
    """axvline artists: two points sharing an x value, spanning the axes in y."""
    return [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
        and list(line.get_ydata()) == [0, 1]
    ]


def _fill_y(poly) -> np.ndarray:
    return np.concatenate([path.vertices[:, 1] for path in poly.get_paths()])


# ---------------------------------------------------------------------------
# plot_equity_curve()
# ---------------------------------------------------------------------------


def test_equity_curve_plots_the_run_equity_on_its_dates(config, run, ax):
    (line,) = plot_equity_curve(config, run, ax)
    np.testing.assert_array_equal(line.get_ydata(), run["equity_curve"].values)
    pd.testing.assert_index_equal(_dates(line), run["equity_curve"].index, check_names=False)


def test_equity_curve_labels_the_strategy_by_default(config, run, ax):
    (line,) = plot_equity_curve(config, run, ax)
    assert line.get_label() == "strategy"
    assert ax.get_legend() is not None


def test_equity_curve_forwards_params_dict_over_the_defaults(config, run, ax):
    (line,) = plot_equity_curve(
        config, run, ax, params_dict={"label": "sma 5/21", "color": "purple"}
    )
    assert line.get_label() == "sma 5/21"
    assert line.get_color() == "purple"


def test_equity_curve_marks_the_starting_capital(config, run, ax):
    plot_equity_curve(config, run, ax)
    horizontal = [
        line
        for line in ax.get_lines()
        if list(line.get_ydata()) == [run["start_capital"]] * 2
    ]
    assert len(horizontal) == 1


def test_benchmark_is_rebased_to_the_starting_capital(config, run, ax):
    returns = pd.Series(0.001, index=run["equity_curve"].index)
    _, bench_line = plot_equity_curve(config, run, ax, benchmark=returns)
    n = len(returns)
    expected = run["start_capital"] * 1.001 ** np.arange(1, n + 1)
    np.testing.assert_allclose(bench_line.get_ydata(), expected, rtol=1e-12)
    assert bench_line.get_label() == "benchmark"


def test_benchmark_with_no_returns_is_flat_at_the_starting_capital(config, run, ax):
    returns = pd.Series(0.0, index=run["equity_curve"].index)
    _, bench_line = plot_equity_curve(config, run, ax, benchmark=returns)
    assert (bench_line.get_ydata() == run["start_capital"]).all()


def test_benchmark_is_restricted_to_the_out_of_sample_dates(config, run, close_panel, ax):
    # A benchmark covering the whole sample (training period included) must be
    # cut to the run's dates, and compounding must start on the first OOS date.
    full_sample = close_panel["AAA"].pct_change().fillna(0.0)
    _, bench_line = plot_equity_curve(config, run, ax, benchmark=full_sample)
    oos = full_sample.reindex(run["equity_curve"].index)

    pd.testing.assert_index_equal(
        _dates(bench_line), run["equity_curve"].index, check_names=False
    )
    assert bench_line.get_ydata()[0] == pytest.approx(
        run["start_capital"] * (1 + oos.iloc[0])
    )


def test_benchmark_gaps_are_treated_as_zero_return_days(config, run, ax):
    index = run["equity_curve"].index
    returns = pd.Series(0.01, index=index).drop(index[5])
    _, bench_line = plot_equity_curve(config, run, ax, benchmark=returns)
    y = bench_line.get_ydata()
    assert y[5] == pytest.approx(y[4])
    assert not np.isnan(y).any()


def test_rebalance_markers_are_drawn_only_on_request(config, run, ax):
    plot_equity_curve(config, run, ax)
    assert _vertical_lines(ax) == []

    _, other_ax = plt.subplots()
    plot_equity_curve(config, run, other_ax, show_rebalances=True)
    markers = _vertical_lines(other_ax)
    assert len(markers) == len(run["fold_costs"])
    marked = pd.DatetimeIndex([line.get_xdata()[0] for line in markers])
    pd.testing.assert_index_equal(marked, run["fold_costs"].index, check_names=False)


def test_equity_curve_returns_one_artist_per_series(config, run, ax):
    assert len(plot_equity_curve(config, run, ax)) == 1
    returns = pd.Series(0.0, index=run["equity_curve"].index)
    assert len(plot_equity_curve(config, run, ax, benchmark=returns)) == 2


# ---------------------------------------------------------------------------
# plot_drawdown()
# ---------------------------------------------------------------------------


def test_drawdown_never_rises_above_zero(config, run, ax):
    poly = plot_drawdown(config, run, ax)
    assert (_fill_y(poly) <= 0.0).all()


def test_drawdown_trough_is_the_max_drawdown(config, run, ax):
    poly = plot_drawdown(config, run, ax)
    assert _fill_y(poly).min() == pytest.approx(max_drawdown(config, run["net_returns"]))


def test_drawdown_trough_is_reachable_from_the_equity_curve(config, run, ax):
    # Independent of the analysis layer: the deepest fall of the equity curve
    # below its running peak, counting the starting capital as the first peak.
    equity = pd.concat(
        [pd.Series([run["start_capital"]]), run["equity_curve"].reset_index(drop=True)]
    )
    expected = (equity / equity.cummax() - 1).min()
    poly = plot_drawdown(config, run, ax)
    assert _fill_y(poly).min() == pytest.approx(expected, abs=1e-12)


def test_drawdown_labels_its_deepest_point(config, run, ax):
    plot_drawdown(config, run, ax)
    (label,) = ax.texts
    max_dd = max_drawdown(config, run["net_returns"])
    assert label.get_text() == f"{max_dd:.1%}"
    assert label.xy[1] == pytest.approx(max_dd)


def test_drawdown_axis_is_formatted_as_percent(config, run, ax):
    plot_drawdown(config, run, ax)
    assert isinstance(ax.yaxis.get_major_formatter(), PercentFormatter)
    assert ax.get_ylabel() == "drawdown"


def test_drawdown_forwards_params_dict_over_the_defaults(config, run, ax):
    poly = plot_drawdown(config, run, ax, params_dict={"alpha": 0.9})
    assert poly.get_alpha() == 0.9


def test_drawdown_of_a_flat_book_is_zero(config, flat_run, ax):
    poly = plot_drawdown(config, flat_run, ax)
    assert (_fill_y(poly) == 0.0).all()
    assert ax.texts[0].get_text() == "0.0%"


# ---------------------------------------------------------------------------
# plot_rolling_sharpe()
# ---------------------------------------------------------------------------


def test_rolling_sharpe_is_annualised_by_the_square_root_of_the_year(config, run, ax):
    (annual,) = plot_rolling_sharpe(config, run, ax, window=30)
    _, other_ax = plt.subplots()
    (per_period,) = plot_rolling_sharpe(config, run, other_ax, window=30, annualised=False)
    np.testing.assert_allclose(
        annual.get_ydata(),
        per_period.get_ydata() * np.sqrt(config.annualise),
        rtol=1e-12,
    )


def test_rolling_sharpe_plots_the_analysis_layer_series(config, run, ax):
    (line,) = plot_rolling_sharpe(config, run, ax, window=30, annualised=False)
    expected = rolling_sharpe(config, run["net_returns"], 30)
    np.testing.assert_allclose(line.get_ydata(), expected.values, rtol=1e-12)


def test_rolling_sharpe_starts_once_the_first_window_is_full(config, run, ax):
    (line,) = plot_rolling_sharpe(config, run, ax, window=30)
    assert _dates(line)[0] == run["net_returns"].index[29]
    assert _dates(line)[-1] == run["net_returns"].index[-1]


def test_rolling_sharpe_defaults_to_the_config_window(config, run, ax):
    (line,) = plot_rolling_sharpe(config, run, ax)
    assert len(line.get_ydata()) == len(run["net_returns"]) - config.sharpe_window + 1
    assert ax.get_ylabel() == f"Rolling Sharpe ({config.sharpe_window}d, ann.)"


def test_rolling_sharpe_label_says_when_it_is_not_annualised(config, run, ax):
    plot_rolling_sharpe(config, run, ax, window=30, annualised=False)
    assert ax.get_ylabel() == "Rolling Sharpe (30d)"


def test_rolling_sharpe_draws_a_zero_reference_line(config, run, ax):
    plot_rolling_sharpe(config, run, ax, window=30)
    assert any(list(line.get_ydata()) == [0, 0] for line in ax.get_lines())


def test_rolling_sharpe_forwards_params_dict(config, run, ax):
    (line,) = plot_rolling_sharpe(config, run, ax, params_dict={"color": "green"})
    assert line.get_color() == "green"


# ---------------------------------------------------------------------------
# plot_backtest_summary()
# ---------------------------------------------------------------------------


def test_summary_creates_one_figure_with_three_stacked_axes(config, run):
    before = set(plt.get_fignums())
    fig, axes = plot_backtest_summary(config, run)
    assert len(set(plt.get_fignums()) - before) == 1
    assert len(axes) == 3
    assert all(a.figure is fig for a in axes)
    plt.close(fig)


def test_summary_axes_share_the_date_axis(config, run):
    fig, axes = plot_backtest_summary(config, run)
    assert axes[1].get_shared_x_axes().joined(axes[0], axes[1])
    assert axes[2].get_shared_x_axes().joined(axes[0], axes[2])
    plt.close(fig)


def test_summary_panels_are_equity_drawdown_and_rolling_sharpe(config, run):
    fig, axes = plot_backtest_summary(config, run)
    np.testing.assert_array_equal(
        axes[0].get_lines()[0].get_ydata(), run["equity_curve"].values
    )
    assert axes[1].get_ylabel() == "drawdown"
    assert axes[2].get_ylabel().startswith("Rolling Sharpe")
    plt.close(fig)


def test_summary_title_reports_the_run_settings_and_results(config, run):
    fig, axes = plot_backtest_summary(config, run)
    title = axes[0].get_title()
    assert run["signal_fn"] in title
    assert str(run["signal_args"]) in title
    assert run["weight_fn"] in title
    assert f"{run['cost_bps']} bps" in title
    assert f"Sharpe {run['results']['sharpe']:.2f}" in title
    assert f"Max DD {run['results']['max_drawdown']:.1%}" in title
    plt.close(fig)


def test_summary_passes_the_benchmark_and_window_through(config, run):
    returns = pd.Series(0.0, index=run["equity_curve"].index)
    fig, axes = plot_backtest_summary(config, run, benchmark=returns, window=30)
    assert len(axes[0].get_lines()) >= 2
    assert axes[2].get_ylabel() == "Rolling Sharpe (30d, ann.)"
    plt.close(fig)


def test_summary_handles_a_flat_book(config, flat_run):
    # A book that never trades has NaN Sharpe; the figure must still render.
    fig, axes = plot_backtest_summary(config, flat_run)
    assert "Sharpe nan" in axes[0].get_title()
    fig.canvas.draw()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Caller owns the figure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "helper", [plot_equity_curve, plot_drawdown, plot_rolling_sharpe]
)
def test_axes_helpers_draw_on_the_given_axes_without_new_figures(
    config, run, ax, helper
):
    before = plt.get_fignums()
    helper(config, run, ax)
    assert plt.get_fignums() == before
    assert ax.has_data()
