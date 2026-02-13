import logging
import os
from time import sleep, time

import ccxt
import pandas as pd

from lnm_client import lnm_client
from strategies.signals.ma_crossover_50_200 import latest_signal


INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def process_long(self, quantity, leverage, takeprofit, stoploss, id_list):
    last = self.lnm.get_last_price()
    tp = round(last * (1 + takeprofit))
    sl = round(last * (1 - stoploss))
    operation_id = self.lnm.market_long(
        quantity=quantity,
        leverage=leverage,
        takeprofit=tp,
        stoploss=sl,
    )["id"]
    id_list.append(operation_id)


def process_short(self, quantity, leverage, takeprofit, stoploss, id_list):
    last = self.lnm.get_last_price()
    tp = round(last * (1 - takeprofit))
    sl = round(last * (1 + stoploss))
    operation_id = self.lnm.market_short(
        quantity=quantity,
        leverage=leverage,
        takeprofit=tp,
        stoploss=sl,
    )["id"]
    id_list.append(operation_id)


class MACrossover:
    def __init__(self, options):
        self.options = options
        self.lnm = lnm_client(options)
        self.exchange = ccxt.binance({"enableRateLimit": True})

    def process_close(self, operation_id, id_list):
        self.lnm.close_position(operation_id)
        if operation_id in id_list:
            id_list.remove(operation_id)

    def _fetch_close_prices(self, symbol: str, interval: str, lookback: int) -> pd.Series:
        candles = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=interval, limit=lookback)
        if not candles:
            return pd.Series(dtype=float)
        df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        return pd.to_numeric(df["Close"], errors="coerce").dropna()

    def get_signal(self, symbol: str, interval: str, lookback: int, fast: int, slow: int, min_separation: float) -> str:
        try:
            close_prices = self._fetch_close_prices(symbol=symbol, interval=interval, lookback=lookback)
            return latest_signal(
                close_prices,
                fast=fast,
                slow=slow,
                min_separation=min_separation,
            )
        except Exception as exc:
            logging.error(f"MA crossover signal failed: {exc}")
            return "neutral"

    def run(
        self,
        quantity,
        leverage,
        takeprofit,
        stoploss,
        interval,
        timeout,
        symbol="BTC/USDT",
        lookback=300,
        fast=50,
        slow=200,
        min_separation=0.0,
    ):
        if interval not in INTERVAL_TO_SECONDS:
            raise ValueError(f"Unsupported interval '{interval}'.")

        timeout_at = time() + 60 * timeout
        wait_seconds = INTERVAL_TO_SECONDS[interval]
        id_list = []

        side = self.get_signal(
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            fast=fast,
            slow=slow,
            min_separation=min_separation,
        )
        logging.info(f"Initial MA crossover signal: {side}")

        if side == "long":
            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
        elif side == "short":
            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
        else:
            side = "neutral"

        sleep(wait_seconds)

        while True:
            signal = self.get_signal(
                symbol=symbol,
                interval=interval,
                lookback=lookback,
                fast=fast,
                slow=slow,
                min_separation=min_separation,
            )
            logging.info(f"Current MA crossover signal: {signal}")

            running_positions = self.lnm.get_trades(type_trade="running")
            id_running = [p["id"] for p in running_positions]

            if len(id_running) > 0 and len(id_list) > 0:
                for operation_id in id_list[:]:
                    if operation_id not in id_running:
                        id_list.remove(operation_id)
                        continue

                    if side == "long":
                        if signal == "short":
                            self.process_close(operation_id, id_list)
                            side = "short"
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                    elif side == "short":
                        if signal == "long":
                            self.process_close(operation_id, id_list)
                            side = "long"
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                    else:
                        if signal == "long":
                            side = "long"
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        elif signal == "short":
                            side = "short"
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)

            else:
                if signal == "long":
                    side = "long"
                    process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                elif signal == "short":
                    side = "short"
                    process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                else:
                    side = "neutral"

            if time() > timeout_at:
                break

            logging.info(f"Active position IDs: {id_list}")
            sleep(wait_seconds)

        closed_ids = id_list[:]
        for operation_id in id_list[:]:
            self.process_close(operation_id, id_list)

        closed_positions = self.lnm.get_trades(type_trade="closed")
        df_closed_positions = pd.DataFrame.from_dict(closed_positions)
        df_closed_pos = df_closed_positions[df_closed_positions["id"].isin(closed_ids)].copy()

        if not df_closed_pos.empty:
            pl = df_closed_pos["pl"].sum()
            logging.info(f"Total PL (sats) = {pl}")
            path = os.path.join(os.path.dirname(__file__), "df_closed_pos.csv")
            df_closed_pos.to_csv(path, index=False)
            logging.info("df_closed_pos.csv saved in strategies folder")
        else:
            logging.info("No positions to report")
