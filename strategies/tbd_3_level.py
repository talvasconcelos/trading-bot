import json
import logging
import os
from time import sleep, time

import ccxt
import pandas as pd

from lnm_client import lnm_client
from strategies.signals.tbd_3_level import latest_signal


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


class TBDThreeLevelLive:
    def __init__(self, options):
        self.options = options
        self.lnm = lnm_client(options)
        self.exchange = ccxt.binance({"enableRateLimit": True})

    def process_close(self, operation_id, id_list):
        self.lnm.close_position(operation_id)
        if operation_id in id_list:
            id_list.remove(operation_id)

    def _fetch_ohlcv(self, symbol: str, interval: str, lookback: int) -> pd.DataFrame:
        candles = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=interval, limit=lookback)
        if not candles:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    def get_signal(self, ohlcv_df: pd.DataFrame, **signal_kwargs) -> str:
        try:
            return latest_signal(ohlcv_df, **signal_kwargs)
        except Exception as exc:
            logging.error(f"TBD 3-level signal failed: {exc}")
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
        lookback=500,
        ema_fast=9,
        ema_mid=21,
        ema_slow=50,
        htf_bias_tf="1D",
        htf_bias_ema=50,
        atr_period=14,
        vector_vol_multiplier=1.5,
        breakout_vol_multiplier=1.0,
        wick_sweep_multiplier=1.0,
        level_lookback=42,
        level_tolerance=0.0025,
        formation_max_gap=20,
        formation_confirm_bars=6,
        higher_low_threshold=0.001,
        lower_high_threshold=0.001,
        exhaustion_lookback=20,
        push_window=4,
        require_weekend_consolidation=False,
        weekend_consolidation_lookback=18,
        weekend_consolidation_atr_mult=1.5,
        session_start_hour=8,
        session_end_hour=18,
    ):
        if interval not in INTERVAL_TO_SECONDS:
            raise ValueError(f"Unsupported interval '{interval}'.")

        timeout_at = time() + 60 * timeout
        wait_seconds = INTERVAL_TO_SECONDS[interval]
        id_list = []

        signal_kwargs = {
            "ema_fast": ema_fast,
            "ema_mid": ema_mid,
            "ema_slow": ema_slow,
            "htf_bias_tf": htf_bias_tf,
            "htf_bias_ema": htf_bias_ema,
            "atr_period": atr_period,
            "vector_vol_multiplier": vector_vol_multiplier,
            "breakout_vol_multiplier": breakout_vol_multiplier,
            "wick_sweep_multiplier": wick_sweep_multiplier,
            "level_lookback": level_lookback,
            "level_tolerance": level_tolerance,
            "formation_max_gap": formation_max_gap,
            "formation_confirm_bars": formation_confirm_bars,
            "higher_low_threshold": higher_low_threshold,
            "lower_high_threshold": lower_high_threshold,
            "exhaustion_lookback": exhaustion_lookback,
            "push_window": push_window,
            "require_weekend_consolidation": require_weekend_consolidation,
            "weekend_consolidation_lookback": weekend_consolidation_lookback,
            "weekend_consolidation_atr_mult": weekend_consolidation_atr_mult,
            "session_start_hour": session_start_hour,
            "session_end_hour": session_end_hour,
        }

        candles = self._fetch_ohlcv(symbol=symbol, interval=interval, lookback=lookback)
        side = self.get_signal(candles, **signal_kwargs)
        logging.info(f"Initial TBD 3-level signal: {side}")

        if side == "long":
            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
        elif side == "short":
            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
        else:
            side = "neutral"

        sleep(wait_seconds)

        while True:
            candles = self._fetch_ohlcv(symbol=symbol, interval=interval, lookback=lookback)
            signal = self.get_signal(candles, **signal_kwargs)
            logging.info(f"Current TBD 3-level signal: {signal}")

            running_positions = json.loads(self.lnm.get_trades(type_trade="running"))
            id_running = [p["id"] for p in running_positions]

            if len(id_running) > 0 and len(id_list) > 0:
                for operation_id in id_list[:]:
                    if operation_id not in id_running:
                        id_list.remove(operation_id)
                        continue

                    if side == "long" and signal == "short":
                        self.process_close(operation_id, id_list)
                        side = "short"
                        process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                    elif side == "short" and signal == "long":
                        self.process_close(operation_id, id_list)
                        side = "long"
                        process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                    elif side == "neutral":
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
