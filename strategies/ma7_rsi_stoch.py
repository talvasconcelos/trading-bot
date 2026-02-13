import logging
import os
from time import sleep, time

import ccxt
import pandas as pd

from lnm_client import lnm_client
from strategies.signals.ma7_rsi_stoch import ma7_rsi_stoch_signals, resolve_profile


INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
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


class MA7RSIStochLive:
    def __init__(self, options):
        self.options = options
        self.lnm = lnm_client(options)
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.entry_price = None
        self.peak_price = None
        self.trough_price = None
        self.trailing_active = False

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
        return df.set_index("timestamp")

    def _reset_state(self):
        self.entry_price = None
        self.peak_price = None
        self.trough_price = None
        self.trailing_active = False

    def _get_signal_pack(self, ohlcv_df: pd.DataFrame, **kwargs):
        long_entry, short_entry, long_exit, short_exit, *_ = ma7_rsi_stoch_signals(ohlcv_df, **kwargs)
        if len(long_entry) == 0:
            return "neutral", False
        if long_entry[-1] == 1:
            return "long", bool(long_exit[-1])
        if short_entry[-1] == 1:
            return "short", bool(short_exit[-1])
        if long_exit[-1] == 1 or short_exit[-1] == 1:
            return "neutral", True
        return "neutral", False

    def run(
        self,
        quantity,
        leverage,
        takeprofit,
        stoploss,
        interval,
        timeout,
        profile="balanced",
        symbol="BTC/USDT",
        lookback=500,
        ma_fast=None,
        ma_mid=None,
        ma_slow=None,
        rsi_period=None,
        rsi_smooth=None,
        stoch_len=None,
        stoch_smooth_k=None,
        stoch_smooth_d=None,
        rsi_entry_floor=None,
        rsi_confirm=None,
        allow_shorts=None,
        use_trend_filter=None,
        min_target_pct=None,
        trail_pct=None,
    ):
        if interval not in INTERVAL_TO_SECONDS:
            raise ValueError(f"Unsupported interval '{interval}'.")

        profile_cfg = resolve_profile(profile)
        ma_fast = profile_cfg["ma_fast"] if ma_fast is None else ma_fast
        ma_mid = profile_cfg["ma_mid"] if ma_mid is None else ma_mid
        ma_slow = profile_cfg["ma_slow"] if ma_slow is None else ma_slow
        rsi_period = profile_cfg["rsi_period"] if rsi_period is None else rsi_period
        rsi_smooth = profile_cfg["rsi_smooth"] if rsi_smooth is None else rsi_smooth
        stoch_len = profile_cfg["stoch_len"] if stoch_len is None else stoch_len
        stoch_smooth_k = profile_cfg["stoch_smooth_k"] if stoch_smooth_k is None else stoch_smooth_k
        stoch_smooth_d = profile_cfg["stoch_smooth_d"] if stoch_smooth_d is None else stoch_smooth_d
        rsi_entry_floor = profile_cfg["rsi_entry_floor"] if rsi_entry_floor is None else rsi_entry_floor
        rsi_confirm = profile_cfg["rsi_confirm"] if rsi_confirm is None else rsi_confirm
        allow_shorts = profile_cfg["allow_shorts"] if allow_shorts is None else allow_shorts
        use_trend_filter = profile_cfg["use_trend_filter"] if use_trend_filter is None else use_trend_filter
        min_target_pct = profile_cfg["min_target_pct"] if min_target_pct is None else min_target_pct
        trail_pct = profile_cfg["trail_pct"] if trail_pct is None else trail_pct

        timeout_at = time() + 60 * timeout
        wait_seconds = INTERVAL_TO_SECONDS[interval]
        id_list = []
        side = "neutral"

        signal_kwargs = dict(
            ma_fast=ma_fast,
            ma_mid=ma_mid,
            ma_slow=ma_slow,
            rsi_period=rsi_period,
            rsi_smooth=rsi_smooth,
            stoch_len=stoch_len,
            stoch_smooth_k=stoch_smooth_k,
            stoch_smooth_d=stoch_smooth_d,
            rsi_entry_floor=rsi_entry_floor,
            rsi_confirm=rsi_confirm,
            allow_shorts=allow_shorts,
            use_trend_filter=use_trend_filter,
        )

        while True:
            candles = self._fetch_ohlcv(symbol=symbol, interval=interval, lookback=lookback)
            signal, exhausted = self._get_signal_pack(candles, **signal_kwargs)
            last_price = self.lnm.get_last_price()

            running_positions = self.lnm.get_trades(type_trade="running")
            id_running = [p["id"] for p in running_positions]

            if side == "long" and self.entry_price:
                self.peak_price = max(self.peak_price or last_price, last_price)
                if self.peak_price >= self.entry_price * (1 + min_target_pct):
                    self.trailing_active = True
                if self.trailing_active and last_price <= self.peak_price * (1 - trail_pct):
                    exhausted = True

            if side == "short" and self.entry_price:
                self.trough_price = min(self.trough_price or last_price, last_price)
                if self.trough_price <= self.entry_price * (1 - min_target_pct):
                    self.trailing_active = True
                if self.trailing_active and last_price >= self.trough_price * (1 + trail_pct):
                    exhausted = True

            if len(id_running) > 0 and len(id_list) > 0:
                for operation_id in id_list[:]:
                    if operation_id not in id_running:
                        id_list.remove(operation_id)
                        side = "neutral"
                        self._reset_state()
                        continue

                    if exhausted:
                        self.process_close(operation_id, id_list)
                        side = "neutral"
                        self._reset_state()
                        continue

                    if side == "long" and signal == "short":
                        self.process_close(operation_id, id_list)
                        side = "short"
                        process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                        self.entry_price = last_price
                        self.peak_price = last_price
                        self.trough_price = last_price
                        self.trailing_active = False
                    elif side == "short" and signal == "long":
                        self.process_close(operation_id, id_list)
                        side = "long"
                        process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        self.entry_price = last_price
                        self.peak_price = last_price
                        self.trough_price = last_price
                        self.trailing_active = False
            else:
                if signal == "long":
                    side = "long"
                    process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                    self.entry_price = last_price
                    self.peak_price = last_price
                    self.trough_price = last_price
                    self.trailing_active = False
                elif signal == "short":
                    side = "short"
                    process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                    self.entry_price = last_price
                    self.peak_price = last_price
                    self.trough_price = last_price
                    self.trailing_active = False
                else:
                    side = "neutral"

            if time() > timeout_at:
                break
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
