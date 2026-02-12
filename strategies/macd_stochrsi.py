import json
import logging
import os
from time import sleep, time

import ccxt
import pandas as pd

from lnm_client import lnm_client
from strategies.signals.macd_stochrsi import latest_signal


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
    last = json.loads(self.lnm.get_last())["lastPrice"]
    tp = round(last * (1 + takeprofit))
    sl = round(last * (1 - stoploss))
    operation_id = json.loads(
        self.lnm.market_long(
            quantity=quantity,
            leverage=leverage,
            takeprofit=tp,
            stoploss=sl,
        )
    )["id"]
    id_list.append(operation_id)


def process_short(self, quantity, leverage, takeprofit, stoploss, id_list):
    last = json.loads(self.lnm.get_last())["lastPrice"]
    tp = round(last * (1 - takeprofit))
    sl = round(last * (1 + stoploss))
    operation_id = json.loads(
        self.lnm.market_short(
            quantity=quantity,
            leverage=leverage,
            takeprofit=tp,
            stoploss=sl,
        )
    )["id"]
    id_list.append(operation_id)


class MACDStochRSILive:
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

    def get_signal(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        rsi_period: int,
        stoch_period: int,
        stoch_smooth_k: int,
        stoch_smooth_d: int,
        stoch_oversold: float,
        stoch_overbought: float,
    ) -> str:
        try:
            close_prices = self._fetch_close_prices(symbol=symbol, interval=interval, lookback=lookback)
            return latest_signal(
                close_prices,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                rsi_period=rsi_period,
                stoch_period=stoch_period,
                stoch_smooth_k=stoch_smooth_k,
                stoch_smooth_d=stoch_smooth_d,
                stoch_oversold=stoch_oversold,
                stoch_overbought=stoch_overbought,
            )
        except Exception as exc:
            logging.error(f"Signal calculation failed: {exc}")
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
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        stoch_period=14,
        stoch_smooth_k=3,
        stoch_smooth_d=3,
        stoch_oversold=40.0,
        stoch_overbought=60.0,
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
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            stoch_smooth_k=stoch_smooth_k,
            stoch_smooth_d=stoch_smooth_d,
            stoch_oversold=stoch_oversold,
            stoch_overbought=stoch_overbought,
        )
        logging.info(f"Initial MACD+StochRSI signal: {side}")

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
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                rsi_period=rsi_period,
                stoch_period=stoch_period,
                stoch_smooth_k=stoch_smooth_k,
                stoch_smooth_d=stoch_smooth_d,
                stoch_oversold=stoch_oversold,
                stoch_overbought=stoch_overbought,
            )
            logging.info(f"Current MACD+StochRSI signal: {signal}")

            num_pos_running = len(json.loads(self.lnm.get_trades(type_trade="running")))
            id_running = [
                json.loads(self.lnm.get_trades(type_trade="running"))[i]["id"]
                for i in range(num_pos_running)
            ]

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

        closed_positions = json.loads(self.lnm.get_trades(type_trade="closed"))
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
