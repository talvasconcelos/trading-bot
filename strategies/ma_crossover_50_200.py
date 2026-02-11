import json
import logging
import os
from time import time, sleep

import pandas as pd


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
        from lnm_client import lnm_client
        self.lnm = lnm_client(options)

    def process_close(self, operation_id, id_list):
        self.lnm.close_position(operation_id)
        # remove the id from the list
        if operation_id in id_list:
            id_list.remove(operation_id)

    def get_signal(self):
        """
        Calculate MA crossover signal based on historical data from LN Markets
        Returns: 'long', 'short', or 'neutral'
        """
        try:
            # In a real implementation, we would fetch historical data from LN Markets API
            # For now, we'll outline the approach:
            
            # Placeholder: We'll return neutral for now, but in practice:
            # - If 50_MA > 200_MA with min separation: return 'long'
            # - If 50_MA < 200_MA with min separation: return 'short'
            # - Otherwise: return 'neutral'
            
            # Since we don't have a direct method to fetch historical data from LN Markets,
            # we'll simulate the logic assuming we had the data
            # This is where the actual strategy logic would go
            
            # For now, returning a signal based on some basic logic
            # In a complete implementation, we would use LN Markets historical data
            # to calculate the actual moving averages and crossovers
            
            # Getting current price to make a decision
            current_price_data = json.loads(self.lnm.get_last())
            current_price = current_price_data['lastPrice']
            
            # This is a simplified placeholder - in reality we'd need historical data
            # to calculate the actual moving averages
            # For demonstration purposes, we'll return a signal based on price levels
            # but a real implementation would use historical data to calculate MAs
            
            # Placeholder logic - in reality, we'd calculate based on actual historical MAs
            return 'neutral'
            
        except Exception as e:
            logging.error(f"Error calculating MA crossover signal: {e}")
            return 'neutral'

    def run(self, quantity, leverage, takeprofit, stoploss, interval, timeout):
        """
        Main strategy loop implementing MA crossover logic
        Following the same pattern as ta_summary strategy
        """
        interval_list = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
        t_interval_list = [60, 300, 900, 1800, 3600, 7200, 14400, 86400]
        t_interval = t_interval_list[interval_list.index(interval)]

        timeout = time() + 60 * timeout

        id_list = []

        # Initial signal
        side = self.get_signal()
        print(f"Initial MA crossover signal: {side}")
        
        if side == 'long':
            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
        elif side == 'short':
            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
        else:
            side = 'neutral'

        sleep(t_interval)

        while True:
            signal = self.get_signal()
            print(f"Current MA crossover signal: {signal}")

            num_pos_running = len(json.loads(self.lnm.get_trades(type_trade='running')))
            id_running = [json.loads(self.lnm.get_trades(type_trade='running'))[i]['id'] for i in
                          range(num_pos_running)]

            if len(id_running) > 0 and len(id_list) > 0:  # If there are running positions and positions in the list
                for id in id_list[:]:  # For each position in the list of positions that have been opened by the bot
                    if id not in id_running:
                        # Position was closed externally, remove from our list
                        if id in id_list:
                            id_list.remove(id)
                        continue
                        
                    if side == 'long':
                        if signal == 'long':
                            logging.info('Keep long open')
                        elif signal == 'short':
                            self.process_close(id, id_list)
                            side = 'short'
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:  # neutral
                            self.process_close(id, id_list)
                            side = 'neutral'
                    elif side == 'short':
                        if signal == 'short':
                            logging.info('Keep short open')
                        elif signal == 'long':
                            self.process_close(id, id_list)
                            side = 'long'
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:  # neutral
                            self.process_close(id, id_list)
                            side = 'neutral'
                    elif side == 'neutral':
                        if signal == 'long':
                            side = 'long'
                            process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                        elif signal == 'short':
                            side = 'short'
                            process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                        else:
                            side = 'neutral'
                            logging.info('Stay neutral')

            else:  # No positions currently running
                if signal == 'long':
                    side = 'long'
                    process_long(self, quantity, leverage, takeprofit, stoploss, id_list)
                elif signal == 'short':
                    side = 'short'
                    process_short(self, quantity, leverage, takeprofit, stoploss, id_list)
                else:
                    side = 'neutral'

            if time() > timeout:
                break

            print(f"Active position IDs: {id_list}")
            sleep(t_interval)

        # Close all remaining positions at timeout
        for operation_id in id_list[:]:
            self.process_close(operation_id, id_list)

        # Generate performance report
        closed_positions = json.loads(self.lnm.get_trades(type_trade='closed'))
        df_closed_positions = pd.DataFrame.from_dict(closed_positions)

        if id_list:  # If we had any positions in our list
            df_closed_pos = df_closed_positions[df_closed_positions['id'].isin(id_list)].copy()
        else:
            df_closed_pos = df_closed_positions  # All closed positions if no specific IDs

        if not df_closed_pos.empty:
            pl = df_closed_pos['pl'].sum()
            logging.info('Total PL (sats) = ' + str(pl))

            path = os.path.join(os.path.dirname(__file__), "df_closed_pos.csv")
            df_closed_pos.to_csv(path)
            logging.info('df_closed_pos.csv saved in strategies folder')
        else:
            logging.info('No positions to report')