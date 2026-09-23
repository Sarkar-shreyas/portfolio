from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from src.backend.config import DevConfig


@dataclass
class Order:
    """
    An order class containing the ticker name, trade quantity and individual asset
    price. Note: quantity < 0 indicates a short order.
    """

    ticker: str
    quantity: float
    price: float
    # volatility: float


@dataclass
class MktState:
    """
    A class containing the current market state of an asset at the time of a desired order.
    For now, only price and average daily volume are stored. Bid/Ask spread will be included
    in a future version.
    """

    price: float
    daily_volume: float


def linear_cost(
    config: DevConfig,
    orders: Order | np.ndarray,
    mkt_states: Optional[MktState | np.ndarray] = None,
    cost_bps: Optional[float] = None,
) -> float | np.ndarray:
    """
    Computes the cost of orders using a simplified linear cost model: |q_i| * p_i * c/10000,
    where q_i, p_i and c_i are the quantity, price and cost bps for trade i.

    Parameters
    ----------
    config: DevConfig
        A DevConfig instance containing default constants
    orders: Order | np.ndarray
        An individual or numpy array of Order objects. See Order class above for details.
    cost_bps: float, Optional
        The cost basis points (bps) for the input orders. Defaults to 10 bps from config.

    Returns
    -------
    float | np.ndarray
        The transaction cost(s) for the input order(s)
    """
    if cost_bps is None:
        cost_bps = config.cost_bps
    if isinstance(orders, np.ndarray):
        quantities = np.array([x.quantity for x in orders])
        prices = np.array([x.price for x in orders])
        notional = np.abs(quantities) * prices
    else:
        notional = np.abs(orders.quantity) * orders.price

    return notional * cost_bps / 10000


def sqrt_cost(
    config: DevConfig,
    orders: Order | np.ndarray,
    mkt_states: MktState | np.ndarray,
    cost_bps: Optional[float] = None,
) -> float | np.ndarray:
    """
    Computes the transaction costs for the input orders following a square-root market
    impact model. Uses daily volume as the normalisation constant.

    Parameters
    ----------
    config: DevConfig
            A DevConfig instance containing default constants
    orders: Order | np.ndarray
        An individual or numpy array of Order objects. See Order class above for details.
    mkt_states: MktState | np.ndarray
        An individual or numpy array of MktState objects. See MktState class above for details.
    cost_bps: float, Optional
        The cost basis points (bps) for the input orders. Defaults to 10 bps from config.

    """
    if cost_bps is None:
        cost_bps = config.cost_bps
    if isinstance(orders, np.ndarray):
        quantities = np.array([x.quantity for x in orders])
        prices = np.array([x.price for x in orders])
        # volatilities = np.array([x.volatility for x in orders])
    else:
        quantities = orders.quantity
        prices = orders.price
        # volatilities = orders.volatility
    if isinstance(mkt_states, np.ndarray):
        volumes = np.array([x.daily_volume for x in mkt_states])
    else:
        volumes = mkt_states.daily_volume

    impact_bps = cost_bps * np.sqrt(np.abs(quantities) / volumes)
    notional = np.abs(quantities) * prices

    return notional * impact_bps / 10000
