import pandas as pd
from backtesting import Strategy

from strategies.signals.trend_exhaustion_rider import resolve_profile, trend_exhaustion_signals


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


def _signals(open_, high, low, close, volume, **kwargs):
    return trend_exhaustion_signals(_frame(open_, high, low, close, volume), **kwargs)


def long_entry(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[0]


def short_entry(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[1]


def long_exhaust(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[2]


def short_exhaust(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[3]


def atr_values(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[4]


class TrendExhaustionRider(Strategy):
    profile = "balanced"

    fast_ema = 50
    slow_ema = 200
    breakout_lookback = 160
    atr_period = 14
    volume_mult = 1.1
    rsi_period = 14
    rsi_exit_long = 55
    rsi_exit_short = 50
    htf_bias_tf = "1D"
    htf_ema_period = 200

    sl_atr_mult = 2.0
    min_target_pct = 0.025
    trail_pct = 0.015

    def init(self):
        profile_cfg = resolve_profile(self.profile)
        kwargs = dict(
            fast_ema=profile_cfg["fast_ema"],
            slow_ema=profile_cfg["slow_ema"],
            breakout_lookback=profile_cfg["breakout_lookback"],
            atr_period=profile_cfg["atr_period"],
            volume_mult=profile_cfg["volume_mult"],
            rsi_period=profile_cfg["rsi_period"],
            rsi_exit_long=profile_cfg["rsi_exit_long"],
            rsi_exit_short=profile_cfg["rsi_exit_short"],
            htf_bias_tf=profile_cfg["htf_bias_tf"],
            htf_ema_period=profile_cfg["htf_ema_period"],
        )
        self.sl_atr_mult = profile_cfg["sl_atr_mult"]
        self.min_target_pct = profile_cfg["min_target_pct"]
        self.trail_pct = profile_cfg["trail_pct"]
        self.long_entry = self.I(long_entry, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.short_entry = self.I(short_entry, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.long_exhaust = self.I(long_exhaust, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.short_exhaust = self.I(short_exhaust, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.atr = self.I(atr_values, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)

        self.entry_price = None
        self.peak_price = None
        self.trough_price = None
        self.trailing_active = False

    def _reset_state(self):
        self.entry_price = None
        self.peak_price = None
        self.trough_price = None
        self.trailing_active = False

    def _update_trailing(self, price):
        if not self.trades:
            return
        active_trade = self.trades[-1]

        if self.position.is_long:
            self.peak_price = max(self.peak_price or price, price)
            if self.entry_price and self.peak_price >= self.entry_price * (1 + self.min_target_pct):
                self.trailing_active = True
            if self.trailing_active:
                trail_sl = self.peak_price * (1 - self.trail_pct)
                current_sl = active_trade.sl if active_trade.sl is not None else 0
                active_trade.sl = max(current_sl, trail_sl)
        elif self.position.is_short:
            self.trough_price = min(self.trough_price or price, price)
            if self.entry_price and self.trough_price <= self.entry_price * (1 - self.min_target_pct):
                self.trailing_active = True
            if self.trailing_active:
                trail_sl = self.trough_price * (1 + self.trail_pct)
                current_sl = active_trade.sl if active_trade.sl is not None else trail_sl
                active_trade.sl = min(current_sl, trail_sl)

    def next(self):
        price = float(self.data.Close[-1])
        atr_now = max(float(self.atr[-1]), 1e-9)

        if self.position:
            self._update_trailing(price)
            if self.position.is_long and self.long_exhaust[-1]:
                self.position.close()
                self._reset_state()
                return
            if self.position.is_short and self.short_exhaust[-1]:
                self.position.close()
                self._reset_state()
                return

        if not self.position and self.long_entry[-1]:
            sl = price - atr_now * self.sl_atr_mult
            self.buy(sl=sl)
            self.entry_price = price
            self.peak_price = price
            self.trough_price = price
            self.trailing_active = False
        elif not self.position and self.short_entry[-1]:
            sl = price + atr_now * self.sl_atr_mult
            self.sell(sl=sl)
            self.entry_price = price
            self.peak_price = price
            self.trough_price = price
            self.trailing_active = False
