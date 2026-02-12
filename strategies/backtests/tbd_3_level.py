from backtesting import Strategy
import pandas as pd

from strategies.signals.tbd_3_level import tbd_3_level_signals


def _frame(open_, high, low, close, volume):
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def long_entries(
    open_,
    high,
    low,
    close,
    volume,
    ema_fast,
    ema_mid,
    ema_slow,
    htf_bias_tf,
    htf_bias_ema,
    atr_period,
    vector_vol_multiplier,
    breakout_vol_multiplier,
    wick_sweep_multiplier,
    level_lookback,
    level_tolerance,
    formation_max_gap,
    formation_confirm_bars,
    higher_low_threshold,
    lower_high_threshold,
    exhaustion_lookback,
    push_window,
    require_weekend_consolidation,
    weekend_consolidation_lookback,
    weekend_consolidation_atr_mult,
    session_start_hour,
    session_end_hour,
):
    return tbd_3_level_signals(
        _frame(open_, high, low, close, volume),
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        htf_bias_tf=htf_bias_tf,
        htf_bias_ema=htf_bias_ema,
        atr_period=atr_period,
        vector_vol_multiplier=vector_vol_multiplier,
        breakout_vol_multiplier=breakout_vol_multiplier,
        wick_sweep_multiplier=wick_sweep_multiplier,
        level_lookback=level_lookback,
        level_tolerance=level_tolerance,
        formation_max_gap=formation_max_gap,
        formation_confirm_bars=formation_confirm_bars,
        higher_low_threshold=higher_low_threshold,
        lower_high_threshold=lower_high_threshold,
        exhaustion_lookback=exhaustion_lookback,
        push_window=push_window,
        require_weekend_consolidation=require_weekend_consolidation,
        weekend_consolidation_lookback=weekend_consolidation_lookback,
        weekend_consolidation_atr_mult=weekend_consolidation_atr_mult,
        session_start_hour=session_start_hour,
        session_end_hour=session_end_hour,
    )[0]


def short_entries(
    open_,
    high,
    low,
    close,
    volume,
    ema_fast,
    ema_mid,
    ema_slow,
    htf_bias_tf,
    htf_bias_ema,
    atr_period,
    vector_vol_multiplier,
    breakout_vol_multiplier,
    wick_sweep_multiplier,
    level_lookback,
    level_tolerance,
    formation_max_gap,
    formation_confirm_bars,
    higher_low_threshold,
    lower_high_threshold,
    exhaustion_lookback,
    push_window,
    require_weekend_consolidation,
    weekend_consolidation_lookback,
    weekend_consolidation_atr_mult,
    session_start_hour,
    session_end_hour,
):
    return tbd_3_level_signals(
        _frame(open_, high, low, close, volume),
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        htf_bias_tf=htf_bias_tf,
        htf_bias_ema=htf_bias_ema,
        atr_period=atr_period,
        vector_vol_multiplier=vector_vol_multiplier,
        breakout_vol_multiplier=breakout_vol_multiplier,
        wick_sweep_multiplier=wick_sweep_multiplier,
        level_lookback=level_lookback,
        level_tolerance=level_tolerance,
        formation_max_gap=formation_max_gap,
        formation_confirm_bars=formation_confirm_bars,
        higher_low_threshold=higher_low_threshold,
        lower_high_threshold=lower_high_threshold,
        exhaustion_lookback=exhaustion_lookback,
        push_window=push_window,
        require_weekend_consolidation=require_weekend_consolidation,
        weekend_consolidation_lookback=weekend_consolidation_lookback,
        weekend_consolidation_atr_mult=weekend_consolidation_atr_mult,
        session_start_hour=session_start_hour,
        session_end_hour=session_end_hour,
    )[1]


def atr_values(
    open_,
    high,
    low,
    close,
    volume,
    ema_fast,
    ema_mid,
    ema_slow,
    htf_bias_tf,
    htf_bias_ema,
    atr_period,
    vector_vol_multiplier,
    breakout_vol_multiplier,
    wick_sweep_multiplier,
    level_lookback,
    level_tolerance,
    formation_max_gap,
    formation_confirm_bars,
    higher_low_threshold,
    lower_high_threshold,
    exhaustion_lookback,
    push_window,
    require_weekend_consolidation,
    weekend_consolidation_lookback,
    weekend_consolidation_atr_mult,
    session_start_hour,
    session_end_hour,
):
    return tbd_3_level_signals(
        _frame(open_, high, low, close, volume),
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        htf_bias_tf=htf_bias_tf,
        htf_bias_ema=htf_bias_ema,
        atr_period=atr_period,
        vector_vol_multiplier=vector_vol_multiplier,
        breakout_vol_multiplier=breakout_vol_multiplier,
        wick_sweep_multiplier=wick_sweep_multiplier,
        level_lookback=level_lookback,
        level_tolerance=level_tolerance,
        formation_max_gap=formation_max_gap,
        formation_confirm_bars=formation_confirm_bars,
        higher_low_threshold=higher_low_threshold,
        lower_high_threshold=lower_high_threshold,
        exhaustion_lookback=exhaustion_lookback,
        push_window=push_window,
        require_weekend_consolidation=require_weekend_consolidation,
        weekend_consolidation_lookback=weekend_consolidation_lookback,
        weekend_consolidation_atr_mult=weekend_consolidation_atr_mult,
        session_start_hour=session_start_hour,
        session_end_hour=session_end_hour,
    )[5]


class TBDThreeLevel(Strategy):
    ema_fast = 9
    ema_mid = 21
    ema_slow = 50
    htf_bias_tf = "1D"
    htf_bias_ema = 50
    atr_period = 14
    vector_vol_multiplier = 1.5
    breakout_vol_multiplier = 1.0
    wick_sweep_multiplier = 1.0
    level_lookback = 42
    level_tolerance = 0.0025
    formation_max_gap = 20
    formation_confirm_bars = 6
    higher_low_threshold = 0.001
    lower_high_threshold = 0.001
    exhaustion_lookback = 20
    push_window = 4
    require_weekend_consolidation = False
    weekend_consolidation_lookback = 18
    weekend_consolidation_atr_mult = 1.5
    session_start_hour = 8
    session_end_hour = 18

    sl_atr_mult = 1.2
    rr_ratio = 2.5

    def init(self):
        self.long_entry = self.I(
            long_entries,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.ema_fast,
            self.ema_mid,
            self.ema_slow,
            self.htf_bias_tf,
            self.htf_bias_ema,
            self.atr_period,
            self.vector_vol_multiplier,
            self.breakout_vol_multiplier,
            self.wick_sweep_multiplier,
            self.level_lookback,
            self.level_tolerance,
            self.formation_max_gap,
            self.formation_confirm_bars,
            self.higher_low_threshold,
            self.lower_high_threshold,
            self.exhaustion_lookback,
            self.push_window,
            self.require_weekend_consolidation,
            self.weekend_consolidation_lookback,
            self.weekend_consolidation_atr_mult,
            self.session_start_hour,
            self.session_end_hour,
        )
        self.short_entry = self.I(
            short_entries,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.ema_fast,
            self.ema_mid,
            self.ema_slow,
            self.htf_bias_tf,
            self.htf_bias_ema,
            self.atr_period,
            self.vector_vol_multiplier,
            self.breakout_vol_multiplier,
            self.wick_sweep_multiplier,
            self.level_lookback,
            self.level_tolerance,
            self.formation_max_gap,
            self.formation_confirm_bars,
            self.higher_low_threshold,
            self.lower_high_threshold,
            self.exhaustion_lookback,
            self.push_window,
            self.require_weekend_consolidation,
            self.weekend_consolidation_lookback,
            self.weekend_consolidation_atr_mult,
            self.session_start_hour,
            self.session_end_hour,
        )
        self.atr = self.I(
            atr_values,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.ema_fast,
            self.ema_mid,
            self.ema_slow,
            self.htf_bias_tf,
            self.htf_bias_ema,
            self.atr_period,
            self.vector_vol_multiplier,
            self.breakout_vol_multiplier,
            self.wick_sweep_multiplier,
            self.level_lookback,
            self.level_tolerance,
            self.formation_max_gap,
            self.formation_confirm_bars,
            self.higher_low_threshold,
            self.lower_high_threshold,
            self.exhaustion_lookback,
            self.push_window,
            self.require_weekend_consolidation,
            self.weekend_consolidation_lookback,
            self.weekend_consolidation_atr_mult,
            self.session_start_hour,
            self.session_end_hour,
        )

    def next(self):
        price = self.data.Close[-1]
        atr_now = max(float(self.atr[-1]), 1e-9)
        long_sl = price - atr_now * self.sl_atr_mult
        short_sl = price + atr_now * self.sl_atr_mult
        long_tp = price + (price - long_sl) * self.rr_ratio
        short_tp = price - (short_sl - price) * self.rr_ratio

        if self.long_entry[-1]:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy(sl=long_sl, tp=long_tp)
        elif self.short_entry[-1]:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell(sl=short_sl, tp=short_tp)
