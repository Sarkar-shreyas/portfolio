import pandas as pd
import numpy as np
from typing import Optional

from src.backend.config import DevConfig


def rsi_mean_reversion(
    config: DevConfig,
    data: pd.Series,
    rsi_overbought: Optional[int] = None,
    rsi_oversold: Optional[int] = None,
) -> pd.Series:
    """Generates a signal series based off the RSI index."""
    if rsi_overbought is None:
        rsi_overbought = config.rsi_overbought
    if rsi_oversold is None:
        rsi_oversold = config.rsi_oversold
    # oversold = data[data < rsi_oversold]
    # overbought = data[data > rsi_overbought]
    signal = pd.Series(0, index=data.index, name="signal")
    signal[data < rsi_oversold] = 1
    signal[data > rsi_overbought] = -1
    return signal
