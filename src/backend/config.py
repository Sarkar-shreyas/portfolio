from dataclasses import dataclass
from typing import Any, Literal
import dotenv
import os


@dataclass
class Config:
    # Data params

    exchange: str = "SMART"
    currency: str = "SGD"
    sec_type: str = "STK"
    benchmark: str = "SPX"

    # Conventions
    annualise: int = 252
    risk_free_rate: float = 0.0

    # Technical analysis
    sma_window: int = 21
    ema_window: int = 21

    # Risk Metrics
    vol_window: int = 21
    beta_window: int = 63
    sharpe_window: int = 63
    var_conf: float = 0.95

    # Reproducability
    random_seed: int = 42


@dataclass
class DevConfig(Config):
    IB_HOST: str = os.getenv("host", "127.0.0.1")
    IB_PORT: int = int(os.getenv("port", 4001))
    IB_CLIENT_ID: int = int(os.getenv("clientId", 1))


@dataclass
class TestConfig(Config):
    random_seed: int = 123
