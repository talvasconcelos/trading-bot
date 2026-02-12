import numpy as np
import pandas as pd

MA7_RSI_STOCH_PROFILES = {
    "long_only_safe": {
        "ma_fast": 7,
        "ma_mid": 21,
        "ma_slow": 50,
        "rsi_period": 14,
        "rsi_smooth": 10,
        "stoch_len": 21,
        "stoch_smooth_k": 7,
        "stoch_smooth_d": 7,
        "rsi_entry_floor": 47.0,
        "rsi_confirm": 58.0,
        "allow_shorts": False,
        "use_trend_filter": True,
        "min_target_pct": 0.04,
        "trail_pct": 0.015,
        "hard_stop_pct": 0.02,
    },
    "balanced": {
        "ma_fast": 7,
        "ma_mid": 21,
        "ma_slow": 50,
        "rsi_period": 14,
        "rsi_smooth": 7,
        "stoch_len": 21,
        "stoch_smooth_k": 7,
        "stoch_smooth_d": 7,
        "rsi_entry_floor": 45.0,
        "rsi_confirm": 58.0,
        "allow_shorts": False,
        "use_trend_filter": True,
        "min_target_pct": 0.04,
        "trail_pct": 0.015,
        "hard_stop_pct": 0.02,
    },
    "with_shorts": {
        "ma_fast": 7,
        "ma_mid": 21,
        "ma_slow": 50,
        "rsi_period": 14,
        "rsi_smooth": 7,
        "stoch_len": 14,
        "stoch_smooth_k": 5,
        "stoch_smooth_d": 5,
        "rsi_entry_floor": 43.0,
        "rsi_confirm": 55.0,
        "allow_shorts": True,
        "use_trend_filter": True,
        "min_target_pct": 0.03,
        "trail_pct": 0.012,
        "hard_stop_pct": 0.015,
    },
}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _stoch_rsi(rsi: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    rsi_low = rsi.rolling(length).min()
    rsi_high = rsi.rolling(length).max()
    raw = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    k = raw.rolling(smooth_k, min_periods=1).mean().fillna(50)
    d = k.rolling(smooth_d, min_periods=1).mean().fillna(50)
    return k, d


def ma7_rsi_stoch_signals(
    ohlcv: pd.DataFrame,
    ma_fast: int = 7,
    ma_mid: int = 21,
    ma_slow: int = 50,
    rsi_period: int = 14,
    rsi_smooth: int = 7,
    stoch_len: int = 21,
    stoch_smooth_k: int = 7,
    stoch_smooth_d: int = 7,
    rsi_entry_floor: float = 45.0,
    rsi_confirm: float = 58.0,
    allow_shorts: bool = False,
    use_trend_filter: bool = True,
):
    df = ohlcv.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    close = df["Close"]
    ma7 = close.rolling(ma_fast).mean()
    ma21 = close.rolling(ma_mid).mean()
    ma50 = close.rolling(ma_slow).mean()

    rsi = _rsi(close, period=rsi_period)
    rsi_sm = rsi.rolling(rsi_smooth, min_periods=1).mean()
    stoch_k, stoch_d = _stoch_rsi(rsi, length=stoch_len, smooth_k=stoch_smooth_k, smooth_d=stoch_smooth_d)

    k_above_d = stoch_k > stoch_d
    k_below_d = stoch_k < stoch_d
    k_cross_up = k_above_d & (stoch_k.shift(1) <= stoch_d.shift(1))
    k_cross_down = k_below_d & (stoch_k.shift(1) >= stoch_d.shift(1))

    trend_long = (ma21 > ma50) if use_trend_filter else pd.Series(True, index=df.index)
    trend_short = (ma21 < ma50) if use_trend_filter else pd.Series(True, index=df.index)

    long_entry = (
        k_cross_up
        & (rsi_sm > rsi_entry_floor)
        & (rsi_sm > rsi_confirm)
        & (close > ma7)
        & trend_long
    ).fillna(False)

    short_entry = (
        allow_shorts
        & k_cross_down
        & (rsi_sm < (100 - rsi_entry_floor))
        & (rsi_sm < 50)
        & (close < ma7)
        & trend_short
    )
    short_entry = pd.Series(short_entry, index=df.index).fillna(False)

    long_exit = (k_cross_down | (close < ma7)).fillna(False)
    short_exit = (k_cross_up | (close > ma7)).fillna(False)

    return (
        long_entry.to_numpy(dtype=float, copy=True),
        short_entry.to_numpy(dtype=float, copy=True),
        long_exit.to_numpy(dtype=float, copy=True),
        short_exit.to_numpy(dtype=float, copy=True),
        ma7.to_numpy(copy=True),
        ma21.to_numpy(copy=True),
        ma50.to_numpy(copy=True),
        rsi_sm.to_numpy(copy=True),
        stoch_k.to_numpy(copy=True),
        stoch_d.to_numpy(copy=True),
    )


def latest_signal(ohlcv: pd.DataFrame, **kwargs) -> str:
    long_entry, short_entry, *_ = ma7_rsi_stoch_signals(ohlcv, **kwargs)
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
    if key not in MA7_RSI_STOCH_PROFILES:
        key = "balanced"
    return MA7_RSI_STOCH_PROFILES[key].copy()
