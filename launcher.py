from strategies.ta_summary import TAS
from strategies.ma_crossover_50_200 import MACrossover
from strategies.macd_stochrsi import MACDStochRSILive
import os
import yaml


# To load Configuration.yml file
def load_yaml(file):
    with open(file) as file:
        load = yaml.load(file, Loader=yaml.FullLoader)
    return load

yaml_file = load_yaml(os.path.join(os.path.dirname(__file__), "configuration.yml"))

lnm_options = yaml_file["lnm_credentials"]

# call the bot you have choosed in configuration file
def bot():
    if yaml_file['strategies'].get('macd_stochrsi'):
        config = yaml_file['macd_stochrsi']
        return MACDStochRSILive.run(
            MACDStochRSILive(lnm_options),
            quantity=config['quantity'],
            leverage=config['leverage'],
            takeprofit=config['takeprofit'],
            stoploss=config['stoploss'],
            interval=config['interval'],
            timeout=config['timeout'],
            symbol=config.get('symbol', 'BTC/USDT'),
            lookback=config.get('lookback', 300),
            macd_fast=config.get('macd_fast', 12),
            macd_slow=config.get('macd_slow', 26),
            macd_signal=config.get('macd_signal', 9),
            rsi_period=config.get('rsi_period', 14),
            stoch_period=config.get('stoch_period', 14),
            stoch_smooth_k=config.get('stoch_smooth_k', 3),
            stoch_smooth_d=config.get('stoch_smooth_d', 3),
            stoch_oversold=config.get('stoch_oversold', 40.0),
            stoch_overbought=config.get('stoch_overbought', 60.0),
        )
    elif yaml_file['strategies'].get('ma_crossover_50_200'):
        config = yaml_file['ma_crossover_50_200']
        return MACrossover.run(
            MACrossover(lnm_options),
            quantity=config['quantity'],
            leverage=config['leverage'],
            takeprofit=config['takeprofit'],
            stoploss=config['stoploss'],
            interval=config['interval'],
            timeout=config['timeout'],
            symbol=config.get('symbol', 'BTC/USDT'),
            lookback=config.get('lookback', 300),
            fast=config.get('fast', 50),
            slow=config.get('slow', 200),
            min_separation=config.get('min_separation', 0.0),
        )
    elif yaml_file['strategies']['ta_summary']:
        config = yaml_file['ta_summary']
        return TAS.ta_summary(TAS(lnm_options),
                                    quantity = config['quantity'],
                                    leverage = config['leverage'],
                                    takeprofit = config['takeprofit'],
                                    stoploss = config['stoploss'],
                                    interval = config['interval'],
                                    timeout = config['timeout'])
    else:
        return "No strategy selected in configuration"
