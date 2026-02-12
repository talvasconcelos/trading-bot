from strategies.ta_summary import TAS
from strategies.ma_crossover_50_200 import MACrossover
from strategies.macd_stochrsi import MACDStochRSILive
from strategies.tbd_3_level import TBDThreeLevelLive
from strategies.trend_exhaustion_rider import TrendExhaustionRiderLive
from strategies.ma7_rsi_stoch import MA7RSIStochLive
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
    if yaml_file['strategies'].get('trend_exhaustion_rider'):
        config = yaml_file['trend_exhaustion_rider']
        return TrendExhaustionRiderLive.run(
            TrendExhaustionRiderLive(lnm_options),
            quantity=config['quantity'],
            leverage=config['leverage'],
            takeprofit=config['takeprofit'],
            stoploss=config['stoploss'],
            interval=config['interval'],
            timeout=config['timeout'],
            profile=config.get('profile', 'balanced'),
            symbol=config.get('symbol', 'BTC/USDT'),
            lookback=config.get('lookback', 500),
            fast_ema=config.get('fast_ema'),
            slow_ema=config.get('slow_ema'),
            breakout_lookback=config.get('breakout_lookback'),
            atr_period=config.get('atr_period'),
            volume_mult=config.get('volume_mult'),
            rsi_period=config.get('rsi_period'),
            rsi_exit_long=config.get('rsi_exit_long'),
            rsi_exit_short=config.get('rsi_exit_short'),
            htf_bias_tf=config.get('htf_bias_tf'),
            htf_ema_period=config.get('htf_ema_period'),
            min_target_pct=config.get('min_target_pct'),
            trail_pct=config.get('trail_pct'),
        )
    elif yaml_file['strategies'].get('ma7_rsi_stoch'):
        config = yaml_file['ma7_rsi_stoch']
        return MA7RSIStochLive.run(
            MA7RSIStochLive(lnm_options),
            quantity=config['quantity'],
            leverage=config['leverage'],
            takeprofit=config['takeprofit'],
            stoploss=config['stoploss'],
            interval=config['interval'],
            timeout=config['timeout'],
            profile=config.get('profile', 'balanced'),
            symbol=config.get('symbol', 'BTC/USDT'),
            lookback=config.get('lookback', 500),
            ma_fast=config.get('ma_fast'),
            ma_mid=config.get('ma_mid'),
            ma_slow=config.get('ma_slow'),
            rsi_period=config.get('rsi_period'),
            rsi_smooth=config.get('rsi_smooth'),
            stoch_len=config.get('stoch_len'),
            stoch_smooth_k=config.get('stoch_smooth_k'),
            stoch_smooth_d=config.get('stoch_smooth_d'),
            rsi_entry_floor=config.get('rsi_entry_floor'),
            rsi_confirm=config.get('rsi_confirm'),
            allow_shorts=config.get('allow_shorts'),
            use_trend_filter=config.get('use_trend_filter'),
            min_target_pct=config.get('min_target_pct'),
            trail_pct=config.get('trail_pct'),
        )
    elif yaml_file['strategies'].get('tbd_3_level'):
        config = yaml_file['tbd_3_level']
        return TBDThreeLevelLive.run(
            TBDThreeLevelLive(lnm_options),
            quantity=config['quantity'],
            leverage=config['leverage'],
            takeprofit=config['takeprofit'],
            stoploss=config['stoploss'],
            interval=config['interval'],
            timeout=config['timeout'],
            symbol=config.get('symbol', 'BTC/USDT'),
            lookback=config.get('lookback', 500),
            ema_fast=config.get('ema_fast', 9),
            ema_mid=config.get('ema_mid', 21),
            ema_slow=config.get('ema_slow', 50),
            htf_bias_tf=config.get('htf_bias_tf', '1D'),
            htf_bias_ema=config.get('htf_bias_ema', 50),
            atr_period=config.get('atr_period', 14),
            vector_vol_multiplier=config.get('vector_vol_multiplier', 1.5),
            breakout_vol_multiplier=config.get('breakout_vol_multiplier', 1.0),
            wick_sweep_multiplier=config.get('wick_sweep_multiplier', 1.0),
            level_lookback=config.get('level_lookback', 42),
            level_tolerance=config.get('level_tolerance', 0.0025),
            formation_max_gap=config.get('formation_max_gap', 20),
            formation_confirm_bars=config.get('formation_confirm_bars', 6),
            higher_low_threshold=config.get('higher_low_threshold', 0.001),
            lower_high_threshold=config.get('lower_high_threshold', 0.001),
            exhaustion_lookback=config.get('exhaustion_lookback', 20),
            push_window=config.get('push_window', 4),
            require_weekend_consolidation=config.get('require_weekend_consolidation', False),
            weekend_consolidation_lookback=config.get('weekend_consolidation_lookback', 18),
            weekend_consolidation_atr_mult=config.get('weekend_consolidation_atr_mult', 1.5),
            session_start_hour=config.get('session_start_hour', 8),
            session_end_hour=config.get('session_end_hour', 18),
        )
    elif yaml_file['strategies'].get('macd_stochrsi'):
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
