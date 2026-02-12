import numpy as np
import pandas as pd


def sma(values, period: int):
    return pd.Series(values, dtype=float).rolling(period).mean().to_numpy(copy=True)


def ma_crossover_signals(close_values, fast: int = 50, slow: int = 200, min_separation: float = 0.0):
    if fast >= slow:
        raise ValueError("Parameter 'fast' must be lower than 'slow'.")

    close = pd.Series(close_values, dtype=float)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    separation = (fast_ma - slow_ma) / slow_ma.replace(0, np.nan)

    long_signal = ((fast_ma > slow_ma) & (separation >= min_separation)).fillna(False)
    short_signal = ((fast_ma < slow_ma) & (separation <= -min_separation)).fillna(False)

    return (
        long_signal.to_numpy(dtype=float, copy=True),
        short_signal.to_numpy(dtype=float, copy=True),
        fast_ma.to_numpy(copy=True),
        slow_ma.to_numpy(copy=True),
        separation.to_numpy(copy=True),
    )


def latest_signal(close_values, **kwargs) -> str:
    long_signal, short_signal, *_ = ma_crossover_signals(close_values, **kwargs)
    if len(long_signal) == 0:
        return "neutral"
    if long_signal[-1] == 1:
        return "long"
    if short_signal[-1] == 1:
        return "short"
    return "neutral"
