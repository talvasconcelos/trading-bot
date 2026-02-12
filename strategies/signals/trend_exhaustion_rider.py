import numpy as np
import pandas as pd

TREND_EXHAUSTION_PROFILES = {
    "conservative": {
        "fast_ema": 50,
        "slow_ema": 200,
        "breakout_lookback": 220,
        "atr_period": 14,
        "volume_mult": 1.2,
        "rsi_period": 14,
        "rsi_exit_long": 55,
        "rsi_exit_short": 45,
        "htf_bias_tf": "1D",
        "htf_ema_period": 200,
        "sl_atr_mult": 2.5,
        "min_target_pct": 0.03,
        "trail_pct": 0.02,
    },
    "balanced": {
        "fast_ema": 50,
        "slow_ema": 200,
        "breakout_lookback": 160,
        "atr_period": 14,
        "volume_mult": 1.1,
        "rsi_period": 14,
        "rsi_exit_long": 55,
        "rsi_exit_short": 50,
        "htf_bias_tf": "1D",
        "htf_ema_period": 200,
        "sl_atr_mult": 2.0,
        "min_target_pct": 0.025,
        "trail_pct": 0.015,
    },
    "aggressive": {
        "fast_ema": 34,
        "slow_ema": 150,
        "breakout_lookback": 80,
        "atr_period": 14,
        "volume_mult": 1.05,
        "rsi_period": 14,
        "rsi_exit_long": 58,
        "rsi_exit_short": 45,
        "htf_bias_tf": "4H",
        "htf_ema_period": 150,
        "sl_atr_mult": 1.2,
        "min_target_pct": 0.025,
        "trail_pct": 0.01,
    },
}


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def trend_exhaustion_signals(
    ohlcv: pd.DataFrame,
    fast_ema: int = 50,
    slow_ema: int = 200,
    breakout_lookback: int = 160,
    atr_period: int = 14,
    volume_mult: float = 1.1,
    rsi_period: int = 14,
    rsi_exit_long: float = 55,
    rsi_exit_short: float = 50,
    htf_bias_tf: str = "1D",
    htf_ema_period: int = 200,
):
    df = ohlcv.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    ema_fast = _ema(close, fast_ema)
    ema_slow = _ema(close, slow_ema)
    atr = _atr(df, period=atr_period)
    rsi = _rsi(close, period=rsi_period)

    hh_prev = high.rolling(breakout_lookback).max().shift(1)
    ll_prev = low.rolling(breakout_lookback).min().shift(1)
    vol_ok = volume > volume.rolling(20).mean() * volume_mult

    if isinstance(df.index, pd.DatetimeIndex):
        htf_close = close.resample(htf_bias_tf).last().dropna()
        htf_ema = _ema(htf_close, htf_ema_period)
        htf_bull = (htf_close > htf_ema).reindex(df.index, method="ffill").fillna(False)
        htf_bear = (htf_close < htf_ema).reindex(df.index, method="ffill").fillna(False)
    else:
        htf_bull = close > _ema(close, htf_ema_period)
        htf_bear = close < _ema(close, htf_ema_period)

    long_entry = (
        (close > ema_fast)
        & (ema_fast > ema_slow)
        & (close > hh_prev)
        & vol_ok
        & htf_bull
    )
    short_entry = (
        (close < ema_fast)
        & (ema_fast < ema_slow)
        & (close < ll_prev)
        & vol_ok
        & htf_bear
    )

    long_exhaust = ((close < ema_fast) | (rsi < rsi_exit_long)).fillna(False)
    short_exhaust = ((close > ema_fast) | (rsi > rsi_exit_short)).fillna(False)

    return (
        long_entry.fillna(False).to_numpy(dtype=float, copy=True),
        short_entry.fillna(False).to_numpy(dtype=float, copy=True),
        long_exhaust.to_numpy(dtype=float, copy=True),
        short_exhaust.to_numpy(dtype=float, copy=True),
        atr.bfill().ffill().fillna(0).to_numpy(copy=True),
        ema_fast.to_numpy(copy=True),
        ema_slow.to_numpy(copy=True),
    )


def latest_signal(ohlcv: pd.DataFrame, **kwargs) -> str:
    long_entry, short_entry, *_ = trend_exhaustion_signals(ohlcv, **kwargs)
    if len(long_entry) == 0:
        return "neutral"
    if long_entry[-1] == 1:
        return "long"
    if short_entry[-1] == 1:
        return "short"
    return "neutral"


def resolve_profile(profile: str) -> dict:
    if not profile:
        profile = "balanced"
    key = str(profile).lower()
    if key not in TREND_EXHAUSTION_PROFILES:
        key = "balanced"
    return TREND_EXHAUSTION_PROFILES[key].copy()
