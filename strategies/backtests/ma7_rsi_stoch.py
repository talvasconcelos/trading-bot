import pandas as pd
from backtesting import Strategy

from strategies.signals.ma7_rsi_stoch import ma7_rsi_stoch_signals, resolve_profile


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
    return ma7_rsi_stoch_signals(_frame(open_, high, low, close, volume), **kwargs)


def long_entry(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[0]


def short_entry(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[1]


def long_exit(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[2]


def short_exit(open_, high, low, close, volume, **kwargs):
    return _signals(open_, high, low, close, volume, **kwargs)[3]


class MA7RSIStoch(Strategy):
    profile = "balanced"

    ma_fast = 7
    ma_mid = 21
    ma_slow = 50
    rsi_period = 14
    rsi_smooth = 7
    stoch_len = 21
    stoch_smooth_k = 7
    stoch_smooth_d = 7
    rsi_entry_floor = 45.0
    rsi_confirm = 58.0
    allow_shorts = False
    use_trend_filter = True

    min_target_pct = 0.04
    trail_pct = 0.015
    hard_stop_pct = 0.02

    def init(self):
        profile_cfg = resolve_profile(self.profile)
        kwargs = dict(
            ma_fast=profile_cfg["ma_fast"],
            ma_mid=profile_cfg["ma_mid"],
            ma_slow=profile_cfg["ma_slow"],
            rsi_period=profile_cfg["rsi_period"],
            rsi_smooth=profile_cfg["rsi_smooth"],
            stoch_len=profile_cfg["stoch_len"],
            stoch_smooth_k=profile_cfg["stoch_smooth_k"],
            stoch_smooth_d=profile_cfg["stoch_smooth_d"],
            rsi_entry_floor=profile_cfg["rsi_entry_floor"],
            rsi_confirm=profile_cfg["rsi_confirm"],
            allow_shorts=profile_cfg["allow_shorts"],
            use_trend_filter=profile_cfg["use_trend_filter"],
        )
        self.min_target_pct = profile_cfg["min_target_pct"]
        self.trail_pct = profile_cfg["trail_pct"]
        self.hard_stop_pct = profile_cfg["hard_stop_pct"]
        self.long_entry = self.I(long_entry, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.short_entry = self.I(short_entry, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.long_exit = self.I(long_exit, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
        self.short_exit = self.I(short_exit, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume, **kwargs)
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

        if self.position:
            self._update_trailing(price)
            if self.position.is_long and self.long_exit[-1]:
                self.position.close()
                self._reset_state()
                return
            if self.position.is_short and self.short_exit[-1]:
                self.position.close()
                self._reset_state()
                return

        if not self.position and self.long_entry[-1]:
            sl = price * (1 - self.hard_stop_pct)
            self.buy(sl=sl)
            self.entry_price = price
            self.peak_price = price
            self.trough_price = price
            self.trailing_active = False
        elif not self.position and self.short_entry[-1]:
            sl = price * (1 + self.hard_stop_pct)
            self.sell(sl=sl)
            self.entry_price = price
            self.peak_price = price
            self.trough_price = price
            self.trailing_active = False
