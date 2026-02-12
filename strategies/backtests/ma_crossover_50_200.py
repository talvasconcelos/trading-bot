from backtesting import Strategy

from strategies.signals.ma_crossover_50_200 import ma_crossover_signals


def long_signals(close, fast, slow, min_separation):
    return ma_crossover_signals(
        close,
        fast=fast,
        slow=slow,
        min_separation=min_separation,
    )[0]


def short_signals(close, fast, slow, min_separation):
    return ma_crossover_signals(
        close,
        fast=fast,
        slow=slow,
        min_separation=min_separation,
    )[1]


class MACrossover50200(Strategy):
    fast = 50
    slow = 200
    min_separation = 0.0

    def init(self):
        self.long_entry = self.I(
            long_signals,
            self.data.Close,
            self.fast,
            self.slow,
            self.min_separation,
        )
        self.short_entry = self.I(
            short_signals,
            self.data.Close,
            self.fast,
            self.slow,
            self.min_separation,
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
