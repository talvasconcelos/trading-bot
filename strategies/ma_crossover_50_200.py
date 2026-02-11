import json
import logging
import os
from time import time, sleep

import pandas as pd
import ccxt

from lnm_client import lnm_client


def process_long(self, quantity, leverage, takeprofit, stoploss, id_list):
    last = json.loads(self.lnm.get_last())['lastPrice']
    tp = round(last * (1 + takeprofit))
    sl = round(last * (1 - stoploss))
    operation_id = json.loads(self.lnm.market_long(quantity=quantity, leverage=leverage, takeprofit=tp, stoploss=sl))['id']
    id_list.append(operation_id)


def process_short(self, quantity, leverage, takeprofit, stoploss, id_list):
    last = json.loads(self.lnm.get_last())['lastPrice']
    tp = round(last * (1 - takeprofit))
    sl = round(last * (1 + stoploss))
    operation_id = json.loads(self.lnm.market_short(quantity=quantity, leverage=leverage, takeprofit=tp, stoploss=sl))['id']
    id_list.append(operation_id)


class MACrossover:
    def __init__(self, options):
        self.options = options
        self.lnm = lnm_client(options)
        # Initialize CCXT exchange for data
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.symbol = 'BTC/USDT'
        self.fast_period = 50
        self.slow_period = 200
        self.min_separation = 0.005  # 0.5%

    def fetch_ohlcv(self, interval='1h', limit=1000):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_indicators(self, df):
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        return df

    def get_signal(self):
        # Fetch enough data for slow MA + some buffer
        limit = self.slow_period + 10
        df = self.fetch_ohlcv(interval='1h', limit=limit)
        df = self.calculate_indicators(df)
        if len(df) < self.slow_period:
            logging.warning("Not enough data to compute MAs")
            return None
        latest = df.iloc[-1]
        fast = latest['fast_ma']
        slow = latest['slow_ma']
        # separation threshold
        sep = (fast - slow) / slow
        if sep >= self.min_separation:
            return 'long'
        elif sep <= -self.min_separation:
            return 'short'
        else:
            return 'neutral'

    def run(self, quantity, leverage, takeprofit, stoploss, interval='1h', timeout=60):
        """
        Main loop: check signal every 'interval' minutes (convert to seconds).
        Runs for 'timeout' minutes.
        """
        # Map interval to seconds
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400,
            '1d': 86400, '1W': 604800, '1M': 2592000
        }
        sleep_sec = interval_map.get(interval, 3600)
        timeout_sec = timeout * 60
        end_time = time() + timeout_sec
        id_list = []

        # Initial signal
        side = self.get_signal()
        if side == 'long':
            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
        elif side == 'short':
            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
        else:
            side = 'neutral'
        logging.info(f"Initial side: {side}")
        print(f"Initial side: {side}")

        sleep(sleep_sec)

        while time() < end_time:
            signal = self.get_signal()
            logging.info(f"Current signal: {signal}")
            print(f"Current signal: {signal}")
            num_pos_running = len(json.loads(self.lnm.get_trades(type_trade='running')))
            id_running = [json.loads(self.lnm.get_trades(type_trade='running'))[i]['id'] for i in range(num_pos_running)]

            if len(id_running) > 0 and len(id_list) > 0:
                for pos_id in id_list[:]:  # iterate over copy
                    if pos_id not in id_running:
                        # already closed manually? remove
                        if pos_id in id_list:
                            id_list.remove(pos_id)
                        continue
                    if side == 'long':
                        if signal in ('long', 'neutral'):
                            logging.info('Keep long open')
                        elif signal == 'short':
                            self.lnm.close_position(pos_id)
                            id_list.remove(pos_id)
                            side = 'short'
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:
                            pass
                    elif side == 'short':
                        if signal in ('short', 'neutral'):
                            logging.info('Keep short open')
                        elif signal == 'long':
                            self.lnm.close_position(pos_id)
                            id_list.remove(pos_id)
                            side = 'long'
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:
                            pass
                    elif side == 'neutral':
                        if signal == 'long':
                            side = 'long'
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        elif signal == 'short':
                            side = 'short'
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:
                            logging.info('Stay neutral')
            else:
                if signal == 'long':
                    side = 'long'
                    process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                elif signal == 'short':
                    side = 'short'
                    process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                else:
                    side = 'neutral'
            sleep(sleep_sec)

        # Close all positions at end
        for pos_id in id_list[:]:
            self.lnm.close_position(pos_id)
            id_list.remove(pos_id)

        # Save closed positions to CSV
        try:
            closed_positions = json.loads(self.lnm.get_trades(type_trade='closed'))
            df_closed = pd.DataFrame(closed_positions)
            # Save only those that were opened by this session (by id)
            df_closed_pos = df_closed[df_closed['id'].isin(id_list)].copy()
            if not df_closed_pos.empty:
                path = os.path.join(os.path.dirname(__file__), "df_closed_pos.csv")
                df_closed_pos.to_csv(path)
                logging.info('df_closed_pos.csv saved')
        except Exception as e:
            logging.error(f"Error saving closed positions: {e}")
        return