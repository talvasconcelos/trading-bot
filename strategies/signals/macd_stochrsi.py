import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd_stochrsi_indicators(
    close_values,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    stoch_period: int = 14,
    stoch_smooth_k: int = 3,
    stoch_smooth_d: int = 3,
):
    close = pd.Series(close_values, dtype=float)

    macd_line = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_signal_line = _ema(macd_line, macd_signal)

    rsi = _rsi(close, rsi_period)
    rsi_low = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    rsi_high = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    stoch_rsi_raw = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    stoch_k = stoch_rsi_raw.rolling(stoch_smooth_k, min_periods=1).mean().fillna(50)
    stoch_d = stoch_k.rolling(stoch_smooth_d, min_periods=1).mean().fillna(50)

    return (
        macd_line.to_numpy(copy=True),
        macd_signal_line.to_numpy(copy=True),
        stoch_k.to_numpy(copy=True),
        stoch_d.to_numpy(copy=True),
    )


def macd_stochrsi_entry_signals(
    close_values,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    stoch_period: int = 14,
    stoch_smooth_k: int = 3,
    stoch_smooth_d: int = 3,
    stoch_oversold: float = 40.0,
    stoch_overbought: float = 60.0,
):
    macd_line, macd_signal_line, stoch_k, stoch_d = macd_stochrsi_indicators(
        close_values,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        rsi_period=rsi_period,
        stoch_period=stoch_period,
        stoch_smooth_k=stoch_smooth_k,
        stoch_smooth_d=stoch_smooth_d,
    )

    macd_cross_up = (macd_line[1:] > macd_signal_line[1:]) & (macd_line[:-1] <= macd_signal_line[:-1])
    macd_cross_down = (macd_line[1:] < macd_signal_line[1:]) & (macd_line[:-1] >= macd_signal_line[:-1])
    stoch_cross_up = (stoch_k[1:] > stoch_d[1:]) & (stoch_k[:-1] <= stoch_d[:-1])
    stoch_cross_down = (stoch_k[1:] < stoch_d[1:]) & (stoch_k[:-1] >= stoch_d[:-1])

    long_entry = np.zeros_like(macd_line, dtype=float)
    short_entry = np.zeros_like(macd_line, dtype=float)

    valid_long = (
        macd_cross_up
        & stoch_cross_up
        & (stoch_k[1:] <= stoch_oversold)
    )
    valid_short = (
        macd_cross_down
        & stoch_cross_down
        & (stoch_k[1:] >= stoch_overbought)
    )

    long_entry[1:] = valid_long.astype(float)
    short_entry[1:] = valid_short.astype(float)

    return long_entry, short_entry, macd_line, macd_signal_line, stoch_k, stoch_d


def latest_signal(close_values, **kwargs) -> str:
    long_entry, short_entry, *_ = macd_stochrsi_entry_signals(close_values, **kwargs)
    if len(long_entry) == 0:
        return "neutral"
    if long_entry[-1] == 1:
        return "long"
    if short_entry[-1] == 1:
        return "short"
    return "neutral"
