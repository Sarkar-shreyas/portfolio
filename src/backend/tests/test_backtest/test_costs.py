"""Tests for src.backend.backtest.costs.

Both cost models return a cost in the same units as ``|quantity| * price``
(notional), never a rate, and are always non-negative regardless of trade
direction. The linear model charges a fixed ``cost_bps`` of notional; the
square-root model scales that charge by the square root of the participation
rate ``|quantity| / daily_volume``, so the two agree exactly when an order is
the size of a full trading day of volume.
"""

import numpy as np
import pytest

from src.backend.backtest.costs import Order, MktState, linear_cost, sqrt_cost


# ---------------------------------------------------------------------------
# Closed-form benchmarks
# ---------------------------------------------------------------------------


def expected_linear(quantity: float, price: float, cost_bps: float) -> float:
    """|q| * p * c / 10000."""
    return abs(quantity) * price * cost_bps / 10000


def expected_sqrt(
    quantity: float, price: float, daily_volume: float, cost_bps: float
) -> float:
    """|q| * p * (c * sqrt(|q| / V)) / 10000."""
    impact_bps = cost_bps * np.sqrt(abs(quantity) / daily_volume)
    return abs(quantity) * price * impact_bps / 10000


# ---------------------------------------------------------------------------
# Order / MktState
# ---------------------------------------------------------------------------


def test_order_stores_its_fields():
    order = Order(ticker="AAA", quantity=250.0, price=12.5)
    assert order.ticker == "AAA"
    assert order.quantity == 250.0
    assert order.price == 12.5


def test_order_accepts_negative_quantity_for_shorts():
    order = Order("AAA", -250.0, 12.5)
    assert order.quantity < 0


def test_mkt_state_stores_its_fields():
    state = MktState(price=12.5, daily_volume=1_000.0)
    assert state.price == 12.5
    assert state.daily_volume == 1_000.0


# ---------------------------------------------------------------------------
# linear_cost() - single order
# ---------------------------------------------------------------------------


def test_linear_cost_matches_closed_form(config, order):
    cost = linear_cost(config, order, cost_bps=10.0)
    assert cost == pytest.approx(expected_linear(order.quantity, order.price, 10.0))


def test_linear_cost_returns_a_finite_scalar(config, order):
    cost = linear_cost(config, order)
    assert np.ndim(cost) == 0
    assert np.isfinite(cost)


def test_linear_cost_defaults_to_config_bps(config, order):
    implicit = linear_cost(config, order)
    explicit = linear_cost(config, order, cost_bps=config.cost_bps)
    assert implicit == explicit


def test_linear_cost_explicit_bps_overrides_config(config, order):
    assert linear_cost(config, order, cost_bps=50.0) != linear_cost(config, order)


def test_linear_cost_is_zero_at_zero_bps(config, order):
    assert linear_cost(config, order, cost_bps=0.0) == 0.0


def test_linear_cost_is_zero_for_a_zero_quantity_order(config):
    assert linear_cost(config, Order("AAA", 0.0, 50.0)) == 0.0


def test_linear_cost_ignores_trade_direction(config):
    buy = linear_cost(config, Order("AAA", 1000.0, 50.0))
    sell = linear_cost(config, Order("AAA", -1000.0, 50.0))
    assert buy == pytest.approx(sell)


@pytest.mark.parametrize("factor", [2.0, 5.0, 0.5])
def test_linear_cost_is_linear_in_quantity(config, factor):
    base = linear_cost(config, Order("AAA", 1000.0, 50.0))
    scaled = linear_cost(config, Order("AAA", 1000.0 * factor, 50.0))
    assert scaled == pytest.approx(base * factor)


@pytest.mark.parametrize("factor", [2.0, 5.0, 0.5])
def test_linear_cost_is_linear_in_price(config, factor):
    base = linear_cost(config, Order("AAA", 1000.0, 50.0))
    scaled = linear_cost(config, Order("AAA", 1000.0, 50.0 * factor))
    assert scaled == pytest.approx(base * factor)


def test_linear_cost_is_linear_in_bps(config, order):
    assert linear_cost(config, order, cost_bps=20.0) == pytest.approx(
        2 * linear_cost(config, order, cost_bps=10.0)
    )


# ---------------------------------------------------------------------------
# linear_cost() - vectorised
# ---------------------------------------------------------------------------


def test_linear_cost_array_returns_one_cost_per_order(config, order_array):
    costs = linear_cost(config, order_array)
    assert isinstance(costs, np.ndarray)
    assert costs.shape == order_array.shape


def test_linear_cost_array_matches_elementwise_singles(config, order_array):
    costs = linear_cost(config, order_array, cost_bps=10.0)
    singles = [linear_cost(config, o, cost_bps=10.0) for o in order_array]
    assert costs == pytest.approx(np.array(singles))


def test_linear_cost_array_matches_closed_form(config, order_array):
    costs = linear_cost(config, order_array, cost_bps=7.5)
    expected = [expected_linear(o.quantity, o.price, 7.5) for o in order_array]
    assert costs == pytest.approx(np.array(expected))


def test_linear_cost_array_is_non_negative(config, order_array):
    assert (linear_cost(config, order_array) >= 0).all()


def test_linear_cost_array_defaults_to_config_bps(config, order_array):
    implicit = linear_cost(config, order_array)
    explicit = linear_cost(config, order_array, cost_bps=config.cost_bps)
    assert implicit == pytest.approx(explicit)


# ---------------------------------------------------------------------------
# sqrt_cost() - single order
# ---------------------------------------------------------------------------


def test_sqrt_cost_matches_closed_form(config, order, mkt_state):
    cost = sqrt_cost(config, order, mkt_state, cost_bps=10.0)
    assert cost == pytest.approx(
        expected_sqrt(order.quantity, order.price, mkt_state.daily_volume, 10.0)
    )


def test_sqrt_cost_defaults_to_config_bps(config, order, mkt_state):
    implicit = sqrt_cost(config, order, mkt_state)
    explicit = sqrt_cost(config, order, mkt_state, cost_bps=config.cost_bps)
    assert implicit == explicit


def test_sqrt_cost_explicit_bps_overrides_config(config, order, mkt_state):
    assert sqrt_cost(config, order, mkt_state, cost_bps=50.0) != sqrt_cost(
        config, order, mkt_state
    )


def test_sqrt_cost_is_zero_at_zero_bps(config, order, mkt_state):
    assert sqrt_cost(config, order, mkt_state, cost_bps=0.0) == 0.0


def test_sqrt_cost_is_zero_for_a_zero_quantity_order(config, mkt_state):
    assert sqrt_cost(config, Order("AAA", 0.0, 50.0), mkt_state) == 0.0


def test_sqrt_cost_ignores_trade_direction(config, mkt_state):
    buy = sqrt_cost(config, Order("AAA", 1000.0, 50.0), mkt_state)
    sell = sqrt_cost(config, Order("AAA", -1000.0, 50.0), mkt_state)
    assert buy == pytest.approx(sell)


@pytest.mark.parametrize("quantity", [-5000.0, -1.0, 0.0, 1.0, 5000.0])
def test_sqrt_cost_is_non_negative(config, mkt_state, quantity):
    assert sqrt_cost(config, Order("AAA", quantity, 50.0), mkt_state) >= 0


@pytest.mark.parametrize("factor", [2.0, 4.0, 0.25])
def test_sqrt_cost_scales_with_quantity_to_the_three_halves(config, mkt_state, factor):
    # Notional is linear in q and impact_bps is proportional to sqrt(q),
    # so the total charge grows like q ** 1.5.
    base = sqrt_cost(config, Order("AAA", 1000.0, 50.0), mkt_state)
    scaled = sqrt_cost(config, Order("AAA", 1000.0 * factor, 50.0), mkt_state)
    assert scaled == pytest.approx(base * factor**1.5)


def test_sqrt_cost_is_linear_in_price(config, mkt_state):
    base = sqrt_cost(config, Order("AAA", 1000.0, 50.0), mkt_state)
    scaled = sqrt_cost(config, Order("AAA", 1000.0, 100.0), mkt_state)
    assert scaled == pytest.approx(2 * base)


def test_sqrt_cost_falls_with_the_square_root_of_volume(config, order):
    thin = sqrt_cost(config, order, MktState(order.price, 1_000_000.0))
    deep = sqrt_cost(config, order, MktState(order.price, 4_000_000.0))
    assert deep == pytest.approx(thin / 2)
    assert deep < thin


def test_sqrt_cost_per_unit_notional_rises_with_order_size(config, mkt_state):
    small = Order("AAA", 1_000.0, 50.0)
    large = Order("AAA", 100_000.0, 50.0)
    small_rate = sqrt_cost(config, small, mkt_state) / abs(small.quantity * small.price)
    large_rate = sqrt_cost(config, large, mkt_state) / abs(large.quantity * large.price)
    assert large_rate > small_rate


# ---------------------------------------------------------------------------
# sqrt_cost() vs linear_cost()
# ---------------------------------------------------------------------------


def test_sqrt_cost_equals_linear_cost_at_full_day_participation(config):
    # impact_bps == cost_bps exactly when |q| == daily_volume.
    order = Order("AAA", 1_000_000.0, 50.0)
    state = MktState(50.0, 1_000_000.0)
    assert sqrt_cost(config, order, state) == pytest.approx(linear_cost(config, order))


def test_sqrt_cost_is_cheaper_than_linear_below_full_participation(config, mkt_state):
    order = Order("AAA", 1_000.0, 50.0)
    assert sqrt_cost(config, order, mkt_state) < linear_cost(config, order)


def test_sqrt_cost_is_dearer_than_linear_above_full_participation(config):
    order = Order("AAA", 4_000_000.0, 50.0)
    state = MktState(50.0, 1_000_000.0)
    assert sqrt_cost(config, order, state) > linear_cost(config, order)


# ---------------------------------------------------------------------------
# sqrt_cost() - vectorised
# ---------------------------------------------------------------------------


def test_sqrt_cost_array_returns_one_cost_per_order(
    config, order_array, mkt_state_array
):
    costs = sqrt_cost(config, order_array, mkt_state_array)
    assert isinstance(costs, np.ndarray)
    assert costs.shape == order_array.shape


def test_sqrt_cost_array_matches_elementwise_singles(
    config, order_array, mkt_state_array
):
    costs = sqrt_cost(config, order_array, mkt_state_array, cost_bps=10.0)
    singles = [
        sqrt_cost(config, o, s, cost_bps=10.0)
        for o, s in zip(order_array, mkt_state_array)
    ]
    assert costs == pytest.approx(np.array(singles))


def test_sqrt_cost_array_matches_closed_form(config, order_array, mkt_state_array):
    costs = sqrt_cost(config, order_array, mkt_state_array, cost_bps=7.5)
    expected = [
        expected_sqrt(o.quantity, o.price, s.daily_volume, 7.5)
        for o, s in zip(order_array, mkt_state_array)
    ]
    assert costs == pytest.approx(np.array(expected))


def test_sqrt_cost_array_uses_the_positionally_matching_market_state(config):
    # Per-order volumes must line up positionally: the same order priced
    # against a deeper book has to come out cheaper.
    orders = np.array([Order("AAA", 1000.0, 50.0), Order("BBB", 1000.0, 50.0)])
    states = np.array([MktState(50.0, 1_000_000.0), MktState(50.0, 4_000_000.0)])
    costs = sqrt_cost(config, orders, states)
    assert costs[1] == pytest.approx(costs[0] / 2)


def test_sqrt_cost_array_is_non_negative(config, order_array, mkt_state_array):
    assert (sqrt_cost(config, order_array, mkt_state_array) >= 0).all()


def test_sqrt_cost_array_defaults_to_config_bps(config, order_array, mkt_state_array):
    implicit = sqrt_cost(config, order_array, mkt_state_array)
    explicit = sqrt_cost(config, order_array, mkt_state_array, cost_bps=config.cost_bps)
    assert implicit == pytest.approx(explicit)
