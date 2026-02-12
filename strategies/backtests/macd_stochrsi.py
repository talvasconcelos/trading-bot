from backtesting import Strategy

from strategies.signals.macd_stochrsi import macd_stochrsi_entry_signals


def long_entries(close, macd_fast, macd_slow, macd_signal, rsi_period, stoch_period, stoch_smooth_k, stoch_smooth_d, stoch_oversold, stoch_overbought):
    return macd_stochrsi_entry_signals(
        close,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        rsi_period=rsi_period,
        stoch_period=stoch_period,
        stoch_smooth_k=stoch_smooth_k,
        stoch_smooth_d=stoch_smooth_d,
        stoch_oversold=stoch_oversold,
        stoch_overbought=stoch_overbought,
    )[0]


def short_entries(close, macd_fast, macd_slow, macd_signal, rsi_period, stoch_period, stoch_smooth_k, stoch_smooth_d, stoch_oversold, stoch_overbought):
    return macd_stochrsi_entry_signals(
        close,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        rsi_period=rsi_period,
        stoch_period=stoch_period,
        stoch_smooth_k=stoch_smooth_k,
        stoch_smooth_d=stoch_smooth_d,
        stoch_oversold=stoch_oversold,
        stoch_overbought=stoch_overbought,
    )[1]


class MACDStochRSI(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    stoch_period = 14
    stoch_smooth_k = 3
    stoch_smooth_d = 3
    stoch_oversold = 40.0
    stoch_overbought = 60.0

    def init(self):
        self.long_entry = self.I(
            long_entries,
            self.data.Close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            self.rsi_period,
            self.stoch_period,
            self.stoch_smooth_k,
            self.stoch_smooth_d,
            self.stoch_oversold,
            self.stoch_overbought,
        )
        self.short_entry = self.I(
            short_entries,
            self.data.Close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            self.rsi_period,
            self.stoch_period,
            self.stoch_smooth_k,
            self.stoch_smooth_d,
            self.stoch_oversold,
            self.stoch_overbought,
        )

    def next(self):
        if self.long_entry[-1]:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy()
        elif self.short_entry[-1]:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell()
