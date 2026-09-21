from dataclasses import dataclass
import dotenv
import os
from pathlib import Path


@dataclass
class Config:
    # Data params

    exchange: str = "SMART"
    currency: str = "USD"
    sec_type: str = "STK"
    benchmark: str = "SPX"

    # Conventions
    annualise: int = 252
    risk_free_rate: float = 0.0363
    n_paths: int = 100000
    n_timesteps: int = 252
    var_conf: float = 0.95
    T: float = 1.0

    # Technical analysis
    sma_window: int = 21
    ema_window: int = 21
    pca_components: int = 2
    garch_p: int = 1
    garch_q: int = 1
    garch_o: int = 0
    garch_vol: str = "GARCH"
    garch_dist: str = "normal"

    # Risk Metrics
    vol_window: int = 21
    beta_window: int = 63
    sharpe_window: int = 63

    # Strategy Metrics
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    short_exposure: float = 0.5
    long_exposure: float = 0.5
    tot_exposure: float = 1.0

    # Reproducability
    random_seed: int = 42

    # filepath
    root_dir: Path = Path(os.getenv("ROOT", ""))
    cache_dir: Path = Path(root_dir / "src/backend/cache")
    trial_dir: Path = Path(root_dir / "trial_data")


@dataclass
class DevConfig(Config):
    IB_HOST: str = os.getenv("host", "127.0.0.1")
    IB_PORT: int = int(os.getenv("port", 4001))
    IB_CLIENT_ID: int = int(os.getenv("clientId", 1))
    ALPHA_VANTAGE_KEY: str = str(os.getenv("ALPHA_VANTAGE_KEY", ""))


@dataclass
class TestConfig(Config):
    random_seed: int = 123
