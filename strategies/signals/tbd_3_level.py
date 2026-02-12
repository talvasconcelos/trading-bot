import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    return tr.rolling(period).mean()


def _weekend_no_trade(index: pd.Index) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        return np.zeros(len(index), dtype=bool)
    dow = index.dayofweek
    hour = index.hour
    return (
        ((dow == 4) & (hour >= 17))
        | (dow == 5)
        | ((dow == 6) & (hour < 17))
    )


def _session_filter(index: pd.Index, start_hour: int, end_hour: int) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        return np.ones(len(index), dtype=bool)
    hour = index.hour
    if start_hour <= end_hour:
        return ((hour >= start_hour) & (hour < end_hour)).to_numpy()
    return ((hour >= start_hour) | (hour < end_hour)).to_numpy()


def _htf_bias(df: pd.DataFrame, htf: str = "1D", ema_period: int = 50):
    if not isinstance(df.index, pd.DatetimeIndex):
        close = df["Close"]
        ema = _ema(close, ema_period)
        return (close > ema).to_numpy(), (close < ema).to_numpy()

    htf_close = df["Close"].resample(htf).last().dropna()
    htf_ema = _ema(htf_close, ema_period)
    bullish = (htf_close > htf_ema).reindex(df.index, method="ffill").fillna(False)
    bearish = (htf_close < htf_ema).reindex(df.index, method="ffill").fillna(False)
    return bullish.to_numpy(dtype=bool), bearish.to_numpy(dtype=bool)


def _vector_mid(df: pd.DataFrame, vol_multiplier: float):
    volume = df["Volume"]
    avg_vol = volume.rolling(20).mean()
    bullish_vector = (df["Close"] > df["Open"]) & (volume > avg_vol * vol_multiplier)
    bearish_vector = (df["Close"] < df["Open"]) & (volume > avg_vol * vol_multiplier)

    body_mid = (df["Open"] + df["Close"]) / 2
    bullish_mid = pd.Series(np.where(bullish_vector, body_mid, np.nan), index=df.index).ffill()
    bearish_mid = pd.Series(np.where(bearish_vector, body_mid, np.nan), index=df.index).ffill()
    return bullish_mid.to_numpy(), bearish_mid.to_numpy()


def _w_m_breakout_patterns(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    max_gap: int,
    confirm_bars: int,
    hl_threshold: float,
    lh_threshold: float,
):
    n = len(close)
    w_pattern = np.zeros(n, dtype=bool)
    m_pattern = np.zeros(n, dtype=bool)

    if n < 5:
        return w_pattern, m_pattern

    low_pivots = np.where((low[1:-1] < low[:-2]) & (low[1:-1] < low[2:]))[0] + 1
    high_pivots = np.where((high[1:-1] > high[:-2]) & (high[1:-1] > high[2:]))[0] + 1

    for i in range(n):
        low_pos = np.searchsorted(low_pivots, i, side="right") - 1
        if low_pos >= 1:
            idx2 = low_pivots[low_pos]
            idx1 = low_pivots[low_pos - 1]
            gap = idx2 - idx1
            if gap <= max_gap and (i - idx2) <= confirm_bars:
                higher_low = low[idx2] > low[idx1] * (1 + hl_threshold)
                neckline = np.max(high[idx1:idx2 + 1])
                breakout = close[i] > neckline
                w_pattern[i] = higher_low and breakout

        high_pos = np.searchsorted(high_pivots, i, side="right") - 1
        if high_pos >= 1:
            idx2 = high_pivots[high_pos]
            idx1 = high_pivots[high_pos - 1]
            gap = idx2 - idx1
            if gap <= max_gap and (i - idx2) <= confirm_bars:
                lower_high = high[idx2] < high[idx1] * (1 - lh_threshold)
                neckline = np.min(low[idx1:idx2 + 1])
                breakdown = close[i] < neckline
                m_pattern[i] = lower_high and breakdown

    return w_pattern, m_pattern


def tbd_3_level_signals(
    ohlcv: pd.DataFrame,
    ema_fast: int = 9,
    ema_mid: int = 21,
    ema_slow: int = 50,
    htf_bias_tf: str = "1D",
    htf_bias_ema: int = 50,
    atr_period: int = 14,
    vector_vol_multiplier: float = 1.5,
    breakout_vol_multiplier: float = 1.0,
    wick_sweep_multiplier: float = 1.0,
    level_lookback: int = 42,
    level_tolerance: float = 0.0025,
    formation_max_gap: int = 20,
    formation_confirm_bars: int = 6,
    higher_low_threshold: float = 0.001,
    lower_high_threshold: float = 0.001,
    exhaustion_lookback: int = 20,
    push_window: int = 4,
    require_weekend_consolidation: bool = False,
    weekend_consolidation_lookback: int = 18,
    weekend_consolidation_atr_mult: float = 1.5,
    session_start_hour: int = 8,
    session_end_hour: int = 18,
):
    df = ohlcv.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"].fillna(0)

    ema9 = _ema(close, ema_fast)
    ema21 = _ema(close, ema_mid)
    ema50 = _ema(close, ema_slow)
    atr = _atr(df, period=atr_period).replace(0, np.nan)
    avg_volume = volume.rolling(20).mean()

    bullish_bias, bearish_bias = _htf_bias(df, htf=htf_bias_tf, ema_period=htf_bias_ema)

    bull_vector_mid, bear_vector_mid = _vector_mid(df, vol_multiplier=vector_vol_multiplier)

    body = (close - open_).abs()
    avg_body = body.rolling(20).mean().replace(0, np.nan)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    liquidity_sweep_long = lower_wick > (avg_body * wick_sweep_multiplier)
    liquidity_sweep_short = upper_wick > (avg_body * wick_sweep_multiplier)

    support = low.rolling(level_lookback).min()
    resistance = high.rolling(level_lookback).max()
    support_hit = low <= support * (1 + level_tolerance)
    resistance_hit = high >= resistance * (1 - level_tolerance)
    support_hits3 = support_hit.rolling(level_lookback).sum() >= 3
    resistance_hits3 = resistance_hit.rolling(level_lookback).sum() >= 3

    push_down = low < low.rolling(push_window).min().shift(1)
    push_up = high > high.rolling(push_window).max().shift(1)
    three_push_down = push_down.rolling(exhaustion_lookback).sum() >= 3
    three_push_up = push_up.rolling(exhaustion_lookback).sum() >= 3

    w_pattern, m_pattern = _w_m_breakout_patterns(
        high=high.to_numpy(dtype=float),
        low=low.to_numpy(dtype=float),
        close=close.to_numpy(dtype=float),
        max_gap=formation_max_gap,
        confirm_bars=formation_confirm_bars,
        hl_threshold=higher_low_threshold,
        lh_threshold=lower_high_threshold,
    )

    ema_bull = (ema9 > ema21) & (ema21 > ema50)
    ema_bear = (ema9 < ema21) & (ema21 < ema50)
    volume_break = volume > (avg_volume * breakout_vol_multiplier)

    vector_long_ok = np.isnan(bull_vector_mid) | (close.to_numpy() >= bull_vector_mid)
    vector_short_ok = np.isnan(bear_vector_mid) | (close.to_numpy() <= bear_vector_mid)

    weekend_no_trade = _weekend_no_trade(df.index)
    session_ok = _session_filter(df.index, session_start_hour, session_end_hour)

    consolidation_range = high.rolling(weekend_consolidation_lookback).max() - low.rolling(weekend_consolidation_lookback).min()
    weekend_cons_ok = consolidation_range <= (atr * weekend_consolidation_atr_mult)
    if require_weekend_consolidation:
        weekend_filter = weekend_cons_ok.fillna(False).to_numpy(dtype=bool)
    else:
        weekend_filter = np.ones(len(df), dtype=bool)

    long_entry = (
        bullish_bias
        & three_push_down.fillna(False).to_numpy(dtype=bool)
        & w_pattern
        & ema_bull.fillna(False).to_numpy(dtype=bool)
        & volume_break.fillna(False).to_numpy(dtype=bool)
        & liquidity_sweep_long.fillna(False).to_numpy(dtype=bool)
        & support_hits3.fillna(False).to_numpy(dtype=bool)
        & vector_long_ok
        & weekend_filter
        & session_ok
        & (~weekend_no_trade)
    )

    short_entry = (
        bearish_bias
        & three_push_up.fillna(False).to_numpy(dtype=bool)
        & m_pattern
        & ema_bear.fillna(False).to_numpy(dtype=bool)
        & volume_break.fillna(False).to_numpy(dtype=bool)
        & liquidity_sweep_short.fillna(False).to_numpy(dtype=bool)
        & resistance_hits3.fillna(False).to_numpy(dtype=bool)
        & vector_short_ok
        & weekend_filter
        & session_ok
        & (~weekend_no_trade)
    )

    return (
        long_entry.astype(float),
        short_entry.astype(float),
        ema9.to_numpy(copy=True),
        ema21.to_numpy(copy=True),
        ema50.to_numpy(copy=True),
        atr.bfill().ffill().fillna(0).to_numpy(copy=True),
    )


def latest_signal(ohlcv: pd.DataFrame, **kwargs) -> str:
    long_entry, short_entry, *_ = tbd_3_level_signals(ohlcv, **kwargs)
    if len(long_entry) == 0:
        return "neutral"
    if long_entry[-1] == 1:
        return "long"
    if short_entry[-1] == 1:
        return "short"
    return "neutral"
