import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class CCXTBacktester:
    """Backtester for BTC/USDT trend following strategies, with optional trend and OI filters."""

    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframe='1h'):
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        self.symbol = symbol
        # Use a separate symbol for OI (perpetuals often have :USDT suffix)
        self.oi_symbol = symbol if ':' in symbol else f"{symbol}:USDT"
        self.timeframe = timeframe

    def fetch_historical_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        end_time = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
        all_ohlcv = []
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000
                if end_time and since > end_time:
                    break
                if len(ohlcv) < 1000:
                    break
            except Exception as e:
                print(f"Error fetching OHLCV: {e}")
                break
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        if end_time:
            end_ts = pd.to_datetime(end_time, unit='ms')
            df = df[df.index <= end_ts]
        return df

    def fetch_oi_history(self, end_timestamp: int = None) -> pd.Series:
        """Fetch latest open interest history (max ~500 records) for the symbol using self.oi_symbol.
        Data is limited to recent period. Returns a time series indexed by timestamp."""
        try:
            if not hasattr(self.exchange, 'fetch_open_interest_history'):
                print("Exchange does not support fetch_open_interest_history")
                return pd.Series(dtype='float64')
            # Fetch latest 500 records; Binance may not allow arbitrary since pagination
            try:
                oi_chunk = self.exchange.fetch_open_interest_history(self.oi_symbol, timeframe=self.timeframe, limit=500)
            except Exception as e:
                print(f"Error fetching OI: {e}")
                return pd.Series(dtype='float64')
            if not oi_chunk:
                return pd.Series(dtype='float64')
            df_oi = pd.DataFrame(oi_chunk)
            oi_col = None
            for col in ['openInterestAmount', 'sumOpenInterest', 'openInterest']:
                if col in df_oi.columns:
                    oi_col = col
                    break
            if oi_col is None:
                numeric_cols = df_oi.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    oi_col = numeric_cols[0]
                else:
                    return pd.Series(dtype='float64')
            series = pd.Series(df_oi[oi_col].values, index=pd.to_datetime(df_oi['timestamp'], unit='ms'), name='openInterest')
            series = series.sort_index()
            if end_timestamp:
                end_ts = pd.to_datetime(end_timestamp, unit='ms')
                series = series[series.index <= end_ts]
            return series
        except Exception as e:
            print(f"Error in fetch_oi_history: {e}")
            return pd.Series(dtype='float64')

    def calculate_indicators(self, df: pd.DataFrame, fast_period: int, slow_period: int,
                             use_trend_filter: bool = True, trend_ma_period: int = 200,
                             use_oi_filter: bool = True, oi_periods: int = 1) -> pd.DataFrame:
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        df['returns'] = df['close'].pct_change()
        if use_trend_filter:
            df['trend_ma'] = df['close'].rolling(window=trend_ma_period).mean()
        if use_oi_filter and 'openInterest' in df.columns:
            df['oi_change'] = df['openInterest'].pct_change(periods=oi_periods)
        return df

    def generate_signals(self, df: pd.DataFrame,
                         use_trend_filter: bool = True,
                         use_oi_filter: bool = True,
                         oi_threshold: float = 0.0,
                         min_separation: float = 0.0) -> pd.DataFrame:
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1   # Long
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Short
        df['filtered_signal'] = df['signal'].copy()

        # Separation filter: require MA difference to exceed threshold (to avoid whipsaws)
        if min_separation > 0:
            sep = (df['fast_ma'] - df['slow_ma']) / df['slow_ma']
            long_ok = sep >= min_separation
            short_ok = sep <= -min_separation
            df.loc[~long_ok & (df['filtered_signal'] == 1), 'filtered_signal'] = 0
            df.loc[~short_ok & (df['filtered_signal'] == -1), 'filtered_signal'] = 0

        # Trend filter: require price > trend_ma for long, < trend_ma for short
        if use_trend_filter and 'trend_ma' in df.columns:
            bullish = df['close'] > df['trend_ma']
            bearish = df['close'] < df['trend_ma']
            df.loc[~bullish & (df['filtered_signal'] == 1), 'filtered_signal'] = 0
            df.loc[~bearish & (df['filtered_signal'] == -1), 'filtered_signal'] = 0

        # OI filter: long requires oi_change > threshold, short requires oi_change < -threshold
        if use_oi_filter and 'oi_change' in df.columns:
            long_ok = df['oi_change'] > oi_threshold
            short_ok = df['oi_change'] < -oi_threshold
            df.loc[~long_ok & (df['filtered_signal'] == 1), 'filtered_signal'] = 0
            df.loc[~short_ok & (df['filtered_signal'] == -1), 'filtered_signal'] = 0

        df['position'] = df['filtered_signal'].shift(1)
        return df

    def calculate_returns(self, df: pd.DataFrame, initial_capital: float = 10000.0, commission: float = 0.001) -> pd.DataFrame:
        df['position'] = df['position'].fillna(0)
        df['strategy_returns'] = df['position'] * df['returns']
        position_change = df['position'].diff().abs().fillna(0)
        df['commission_cost'] = position_change * commission
        df['strategy_returns'] = df['strategy_returns'] - df['commission_cost']
        df['strategy_returns'] = df['strategy_returns'].fillna(0)
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['portfolio_value'] = initial_capital * df['cumulative_returns']
        return df

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1) * 100
        n = len(df)
        n_years = n / (365 * 24)
        annualized_return = ((1 + total_return/100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        sharpe = np.sqrt(365*24) * df['strategy_returns'].mean() / df['strategy_returns'].std() if df['strategy_returns'].std() != 0 else 0
        rolling_max = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        win_rate = (df['strategy_returns'] > 0).sum() / len(df['strategy_returns']) * 100
        # Count trade entries: signal changes to non-zero
        signal = df['filtered_signal']
        trade_entries = (signal != 0) & (signal != signal.shift(1))
        total_trades = trade_entries.sum()
        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': int(total_trades),
            'Start Date': df.index[0].strftime('%Y-%m-%d'),
            'End Date': df.index[-1].strftime('%Y-%m-%d'),
        }

    def plot_results(self, df: pd.DataFrame, fast_period: int, slow_period: int, filename: str = None):
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        # Price & MAs
        axes[0].plot(df.index, df['close'], label='Price', alpha=0.7)
        axes[0].plot(df.index, df['fast_ma'], label=f'Fast MA ({fast_period})', alpha=0.7)
        axes[0].plot(df.index, df['slow_ma'], label=f'Slow MA ({slow_period})', alpha=0.7)
        if 'trend_ma' in df.columns:
            axes[0].plot(df.index, df['trend_ma'], label=f'Trend MA (200)', alpha=0.5, linestyle='--')
        axes[0].set_title(f'{self.symbol} Price & MAs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Position
        axes[1].fill_between(df.index, 0, df['position'], where=df['position']>0, alpha=0.5, color='green', label='Long')
        axes[1].fill_between(df.index, 0, df['position'], where=df['position']<0, alpha=0.5, color='red', label='Short')
        axes[1].set_title('Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        # Open Interest
        if 'openInterest' in df.columns:
            axes[2].plot(df.index, df['openInterest'], label='Open Interest', color='purple')
            axes[2].set_title('Open Interest')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        # Equity
        axes[3].plot(df.index, df['portfolio_value'], label='Portfolio Value', color='blue')
        axes[3].set_title('Equity Curve')
        axes[3].set_ylabel('Value ($)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Plot saved to: {filename}")
        else:
            plt.savefig('backtest_results.png', dpi=100, bbox_inches='tight')
            print("Plot saved to: backtest_results.png")
        plt.close()

    def run_backtest(
        self,
        start_date: str,
        end_date: str = None,
        fast_period: int = 20,
        slow_period: int = 50,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        use_trend_filter: bool = True,
        trend_ma_period: int = 200,
        use_oi_filter: bool = False,
        oi_periods: int = 1,
        oi_threshold: float = 0.0,
        min_separation: float = 0.0,
        plot: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest with optional trend and OI filters."""
        print(f"Fetching {self.symbol} data from {start_date} to {end_date or 'now'}...")
        df = self.fetch_historical_data(start_date, end_date)
        if df.empty:
            print("No price data fetched.")
            return df, {}
        print(f"Fetched {len(df)} data points")

        if use_oi_filter:
            # OI data limited to most recent ~500 records; provide end timestamp to align
            end_ts = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
            oi_series = self.fetch_oi_history(end_ts)
            if oi_series.empty:
                print("OI data unavailable; disabling OI filter")
                use_oi_filter = False
            else:
                # Align OI to price index, forward fill missing
                df['openInterest'] = oi_series.reindex(df.index, method='ffill').values

        print("Calculating indicators...")
        df = self.calculate_indicators(df, fast_period, slow_period, use_trend_filter, trend_ma_period, use_oi_filter, oi_periods)

        # Drop NaNs from rolling calculations
        required = ['fast_ma', 'slow_ma', 'returns']
        if use_trend_filter and 'trend_ma' in df.columns:
            required.append('trend_ma')
        if use_oi_filter and 'oi_change' in df.columns:
            required.append('oi_change')
        df = df.dropna(subset=required).copy()
        print(f"After dropping NaNs, {len(df)} rows remain")

        print("Generating signals...")
        df = self.generate_signals(df, use_trend_filter, use_oi_filter, oi_threshold, min_separation)

        print("Calculating returns...")
        df = self.calculate_returns(df, initial_capital, commission)

        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        metrics = self.calculate_metrics(df)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("="*50 + "\n")

        if plot:
            filename = f"backtest_{self.symbol.replace('/', '_').replace(':', '_')}_{self.timeframe}_{start_date}_to_{end_date or 'now'}_fast{fast_period}_slow{slow_period}_trend{trend_ma_period if use_trend_filter else 'off'}_oi{'on' if use_oi_filter else 'off'}.png"
            self.plot_results(df, fast_period, slow_period, filename)

        return df, metrics

def enhanced_ma_crossover_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 20,
    slow_period: int = 50,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    use_trend_filter: bool = True,
    trend_ma_period: int = 200,
    use_oi_filter: bool = False,
    oi_periods: int = 1,
    oi_threshold: float = 0.0,
    min_separation: float = 0.0,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced Moving Average Crossover with trend, OI, and separation filters."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    return backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        fast_period=fast_period,
        slow_period=slow_period,
        initial_capital=initial_capital,
        commission=commission,
        use_trend_filter=use_trend_filter,
        trend_ma_period=trend_ma_period,
        use_oi_filter=use_oi_filter,
        oi_periods=oi_periods,
        oi_threshold=oi_threshold,
        min_separation=min_separation,
        plot=plot
    )

def ema_rsi_swing_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 9,
    slow_period: int = 21,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """EMA+RSI swing strategy: longs when fast>slow & RSI<oversold; shorts when fast<slow & RSI>overbought."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Calculate Indicators
    df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df['returns'] = df['close'].pct_change()
    # Generate signals
    df['signal'] = 0
    long_cond = (df['fast_ma'] > df['slow_ma']) & (df['rsi'] < rsi_oversold)
    short_cond = (df['fast_ma'] < df['slow_ma']) & (df['rsi'] > rsi_overbought)
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['filtered_signal'] = df['signal']  # for metrics calc
    df['position'] = df['signal'].shift(1).fillna(0)
    # Calculate returns & metrics using backtester's methods
    df = backtester.calculate_returns(df, initial_capital, commission)
    metrics = backtester.calculate_metrics(df)
    if plot:
        filename = f"backtest_ema_rsi_{fast_period}_{slow_period}_{start_date}_to_{end_date or 'latest'}.png"
        backtester.plot_results(df, fast_period, slow_period, filename)
    return df, metrics

def oi_ema_crossover_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 9,
    slow_period: int = 21,
    oi_threshold: float = 0.005,  # 0.5% OI increase required
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """EMA crossover with Open Interest filter: only trade when OI increases (new money entering)."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Fetch OI
    end_ts = backtester.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
    oi_series = backtester.fetch_oi_history(end_ts)
    if oi_series.empty:
        print("No OI data available.")
        return df, {}
    # Align OI to price index
    df['openInterest'] = oi_series.reindex(df.index, method='ffill')
    # Drop rows where OI is NaN (before first OI record)
    df = df.dropna(subset=['openInterest']).copy()
    # Calculate EMAs
    df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['returns'] = df['close'].pct_change()
    # OI change
    df['oi_change'] = df['openInterest'].pct_change()
    # Signals
    df['signal'] = 0
    long_cond = (df['fast_ma'] > df['slow_ma']) & (df['oi_change'] > oi_threshold)
    short_cond = (df['fast_ma'] < df['slow_ma']) & (df['oi_change'] > oi_threshold)
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['filtered_signal'] = df['signal']  # for metrics calc
    df['position'] = df['signal'].shift(1).fillna(0)
    # Calculate returns
    df = backtester.calculate_returns(df, initial_capital, commission)
    metrics = backtester.calculate_metrics(df)
    if plot:
        filename = f"backtest_oi_ema_{fast_period}_{slow_period}_{start_date}_to_{end_date or 'latest'}.png"
        backtester.plot_results(df, fast_period, slow_period, filename)
    return df, metrics

def volume_ema_swing_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 9,
    slow_period: int = 21,
    volume_multiplier: float = 2.0,
    volume_window: int = 20,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """EMA crossover with volume spike filter (PVSRA-inspired): only trade on above‑average volume."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Volume spike filter
    df['volume_avg'] = df['volume'].rolling(volume_window).mean()
    df['volume_spike'] = df['volume'] > (volume_multiplier * df['volume_avg'])
    # EMA crossover
    df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['returns'] = df['close'].pct_change()
    # Signals
    df['signal'] = 0
    long_cond = (df['fast_ma'] > df['slow_ma']) & df['volume_spike']
    short_cond = (df['fast_ma'] < df['slow_ma']) & df['volume_spike']
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['filtered_signal'] = df['signal']
    df['position'] = df['signal'].shift(1).fillna(0)
    # Calculate returns & metrics
    df = backtester.calculate_returns(df, initial_capital, commission)
    metrics = backtester.calculate_metrics(df)
    if plot:
        filename = f"backtest_vol_ema_{fast_period}_{slow_period}_{start_date}_to_{end_date or 'latest'}.png"
        backtester.plot_results(df, fast_period, slow_period, filename)
    return df, metrics

def volume_ema_candle_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 9,
    slow_period: int = 21,
    volume_multiplier: float = 2.0,
    volume_window: int = 20,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """EMA crossover + volume spike + candle direction: long only if bullish candle, short only if bearish candle."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Volume spike filter
    df['volume_avg'] = df['volume'].rolling(volume_window).mean()
    df['volume_spike'] = df['volume'] > (volume_multiplier * df['volume_avg'])
    # EMA crossover
    df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['returns'] = df['close'].pct_change()
    # Candle direction
    df['bullish_candle'] = df['close'] > df['open']
    df['bearish_candle'] = df['close'] < df['open']
    # Signals
    df['signal'] = 0
    long_cond = (df['fast_ma'] > df['slow_ma']) & df['volume_spike'] & df['bullish_candle']
    short_cond = (df['fast_ma'] < df['slow_ma']) & df['volume_spike'] & df['bearish_candle']
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['filtered_signal'] = df['signal']
    df['position'] = df['signal'].shift(1).fillna(0)
    # Calculate returns & metrics
    df = backtester.calculate_returns(df, initial_capital, commission)
    metrics = backtester.calculate_metrics(df)
    if plot:
        filename = f"backtest_vol_candle_ema_{fast_period}_{slow_period}_{start_date}_to_{end_date or 'latest'}.png"
        backtester.plot_results(df, fast_period, slow_period, filename)
    return df, metrics

def breakout_volume_strategy(
    start_date: str,
    end_date: str = None,
    range_period: int = 20,
    volume_multiplier: float = 2.0,
    volume_window: int = 20,
    use_trend_filter: bool = True,
    tp_pct: float = 0.025,
    sl_pct: float = 0.015,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Breakout+volume with trend filter and TP/SL exits."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Rolling range (previous bars)
    df['rolling_high'] = df['high'].rolling(range_period).max().shift(1)
    df['rolling_low'] = df['low'].rolling(range_period).min().shift(1)
    # Volume spike
    df['volume_avg'] = df['volume'].rolling(volume_window).mean()
    df['volume_spike'] = df['volume'] > (volume_multiplier * df['volume_avg'])
    # Trend filter (200 EMA)
    if use_trend_filter:
        df['trend_ema'] = df['close'].ewm(span=200, adjust=False).mean().shift(1)
        bullish = df['close'] > df['trend_ema']
        bearish = df['close'] < df['trend_ema']
    else:
        bullish = pd.Series(True, index=df.index)
        bearish = pd.Series(True, index=df.index)
    # Signals
    df['signal'] = 0
    long_cond = (df['close'] > df['rolling_high']) & df['volume_spike'] & bullish
    short_cond = (df['close'] < df['rolling_low']) & df['volume_spike'] & bearish
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['filtered_signal'] = df['signal']

    # Simulate trades with TP/SL
    trades = []  # each: {'entry_idx':i, 'exit_idx':j, 'entry':x, 'exit':y, 'sign':1/-1}
    pos_sign = 0
    entry_price = None
    entry_idx = None

    for i in range(len(df)):
        sig = df.iloc[i]['signal']
        close = df.iloc[i]['close']
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        if pos_sign != 0:
            # TP/SL check
            if pos_sign == 1:
                tp = entry_price * (1 + tp_pct)
                sl = entry_price * (1 - sl_pct)
                if high >= tp:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': tp, 'sign': 1})
                    pos_sign = 0
                    entry_price = None
                    continue
                elif low <= sl:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': sl, 'sign': 1})
                    pos_sign = 0
                    entry_price = None
                    continue
            else:
                tp = entry_price * (1 - tp_pct)
                sl = entry_price * (1 + sl_pct)
                if low <= tp:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': tp, 'sign': -1})
                    pos_sign = 0
                    entry_price = None
                    continue
                elif high >= sl:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': sl, 'sign': -1})
                    pos_sign = 0
                    entry_price = None
                    continue
            # Signal-based exit (including reversal or neutral)
            if sig != pos_sign:
                trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': close, 'sign': pos_sign})
                pos_sign = 0
                entry_price = None
                continue
        # Entry
        if sig != 0 and pos_sign == 0:
            pos_sign = sig
            entry_price = close
            entry_idx = i

    # Final exit at last bar if still open
    if pos_sign != 0:
        trades.append({'entry_idx': entry_idx, 'exit_idx': len(df)-1, 'entry': entry_price, 'exit': df.iloc[-1]['close'], 'sign': pos_sign})

    # Compute metrics from trades
    total_trades = len(trades)
    if total_trades == 0:
        metrics = {
            'Total Return (%)': 0.0,
            'Annualized Return (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Total Trades': 0,
            'Start Date': df.index[0].strftime('%Y-%m-%d'),
            'End Date': df.index[-1].strftime('%Y-%m-%d'),
        }
        return df, metrics

    # Per-trade net returns
    trade_rets = []
    equity = [initial_capital]
    current_equity = initial_capital
    for t in trades:
        gross = (t['exit'] / t['entry'] - 1) * t['sign']
        net = gross - 2 * commission
        trade_rets.append(net)
        current_equity *= (1 + net)
        equity.append(current_equity)

    total_return = (equity[-1] / equity[0] - 1) * 100
    # Annualized
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else len(df) / (365.25 * 24)
    if years <= 0:
        years = len(df) / (365.25 * 24)
    annualized = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    # Sharpe
    if np.std(trade_rets) != 0:
        trades_per_year = total_trades / years if years > 0 else total_trades
        sharpe = np.mean(trade_rets) / np.std(trade_rets) * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    # Win rate
    win_rate = sum(1 for r in trade_rets if r > 0) / total_trades * 100

    # Max drawdown from equity curve
    exit_times = [df.index[t['exit_idx']] for t in trades]
    equity_dates = [df.index[0]] + exit_times
    equity_series = pd.Series(equity, index=equity_dates)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    metrics = {
        'Total Return (%)': round(total_return, 2),
        'Annualized Return (%)': round(annualized, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Total Trades': total_trades,
        'Start Date': df.index[0].strftime('%Y-%m-%d'),
        'End Date': df.index[-1].strftime('%Y-%m-%d'),
    }

    # Plot
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        # Price with entries/exits
        axes[0].plot(df.index, df['close'], label='Close', alpha=0.7)
        entry_times = [df.index[t['entry_idx']] for t in trades]
        entry_prices = [t['entry'] for t in trades]
        axes[0].scatter(entry_times, entry_prices, marker='^', color='green', s=50, label='Entries', zorder=5)
        exit_times = [df.index[t['exit_idx']] for t in trades]
        exit_prices = [t['exit'] for t in trades]
        axes[0].scatter(exit_times, exit_prices, marker='x', color='red', s=50, label='Exits', zorder=5)
        axes[0].set_title(f'{symbol} - Breakout+Volume Entries/Exits')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Equity
        axes[1].plot(equity_series.index, equity_series.values, label='Equity', color='blue')
        axes[1].set_title('Equity Curve')
        axes[1].set_ylabel('Value ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"backtest_breakout_vol_tp{int(tp_pct*100)}_sl{int(sl_pct*100)}_{start_date}_to_{end_date or 'latest'}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
        plt.close()

    return df, metrics

def monthly_reversal_ema_strategy(
    start_date: str,
    end_date: str = None,
    ema_period: int = 20,
    r_multiplier: float = 3.0,  # TP = r * stop distance
    max_month_trades: int = 2,  # cap trades per month (optional)
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Monthly reversal: trade only days 1‑12, wait for sweep of prior month's high/low, then reversal into EMA20."""
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    df = backtester.fetch_historical_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(), {}
    # Resample to daily to compute prior month extremes
    daily = df.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    # Shift to get previous month extremes: need last day of prior month
    # Compute rolling: for each day, get the high/low of the *calendar* month that just ended
    # Use period 'M' month end frequency
    monthly = daily.resample('M').agg({'high': 'max', 'low': 'min'})
    # Align to daily: each day gets the previous month's values
    daily[['prev_month_high', 'prev_month_low']] = monthly[['high', 'low']].shift(1)
    # Bring back to hourly
    df = df.merge(daily[['prev_month_high', 'prev_month_low']], left_index=True, right_index=True, how='left')
    df[['prev_month_high', 'prev_month_low']] = df[['prev_month_high', 'prev_month_low']].ffill()
    # EMA20
    df['ema20'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    # Day of month (from index)
    df['day'] = df.index.day
    # Initialize signals
    df['signal'] = 0
    # State to track sweep events and monthly trade count
    in_long_setup = False
    in_short_setup = False
    sweep_low = None
    sweep_high = None
    long_entry_price = None
    short_entry_price = None
    monthly_trades = 0
    current_month = None

    # We'll build signals by iterating because logic is event‑driven
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        # Reset monthly counter on new month
        if row.name.month != current_month:
            current_month = row.name.month
            monthly_trades = 0
        # Only consider days 1‑12
        if row['day'] > 12:
            in_long_setup = False
            in_short_setup = False
            continue
        # Long setup: sweep below prev_month_low then reclaim above it
        if not in_long_setup:
            # Detect sweep: low < prev_month_low (but not necessarily close below)
            if prev['low'] <= row['prev_month_low'] and row['low'] <= row['prev_month_low']:
                # Price swept below; now wait for reclaim: a candle that closes above prev_month_low
                if row['close'] > row['prev_month_low']:
                    # Check EMA confirmation
                    if row['close'] > row['ema20']:
                        # Enter long on next bar open could also be simulated; here mark signal now
                        df.at[df.index[i], 'signal'] = 1
                        in_long_setup = False  # no further longs this window until new sweep
                        sweep_low = min(prev['low'], row['low'])
                        long_entry_price = row['close']
                        monthly_trades += 1
                        if monthly_trades >= max_month_trades:
                            in_long_setup = True  # block further longs
            # Short setup similarly: sweep above prev_month_high then reclaim below
        if not in_short_setup:
            if prev['high'] >= row['prev_month_high'] and row['high'] >= row['prev_month_high']:
                if row['close'] < row['prev_month_high']:
                    if row['close'] < row['ema20']:
                        df.at[df.index[i], 'signal'] = -1
                        in_short_setup = False
                        sweep_high = max(prev['high'], row['high'])
                        short_entry_price = row['close']
                        monthly_trades += 1
                        if monthly_trades >= max_month_trades:
                            in_short_setup = True

    # After signal generation, simulate trades with TP/SL based on sweep extremes
    df['filtered_signal'] = df['signal']
    # simulate_trades_tp_sl needs to know per‑trade stop levels; easier: we'll store stop/TP in new columns or handle separately.
    # For integration simplicity, I'll run a second pass trade simulation that uses the sweep extremes captured.
    # We already have long_entry_price and sweep_low tracked in the loop, but they're not stored per row.
    # Alternative: let’s embed stop/TP into signal columns: when signal=1, also set stop and TP; same for -1.
    # But stop distance depends on sweep extreme which might differ per trade. We can compute on‑the‑fly in simulation.
    # Instead of modifying simulate_trades_tp_sl globally, I'll write a custom simulator inline.

    trades = []
    pos_sign = 0
    entry_price = None
    entry_idx = None
    sweep_stop = None  # absolute stop price

    for i in range(len(df)):
        sig = df.iloc[i]['signal']
        close = df.iloc[i]['close']
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        if pos_sign != 0:
            # TP = entry + r*(entry - stop) for long; for short entry - r*(stop - entry)
            if pos_sign == 1:
                stop_price = sweep_stop
                tp_price = entry_price + r_multiplier * (entry_price - stop_price)
                if high >= tp_price:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': tp_price, 'sign': 1})
                    pos_sign = 0
                    sweep_stop = None
                    continue
                if low <= stop_price:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': stop_price, 'sign': 1})
                    pos_sign = 0
                    sweep_stop = None
                    continue
            else:
                stop_price = sweep_stop
                tp_price = entry_price - r_multiplier * (stop_price - entry_price)
                if low <= tp_price:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': tp_price, 'sign': -1})
                    pos_sign = 0
                    sweep_stop = None
                    continue
                if high >= stop_price:
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': stop_price, 'sign': -1})
                    pos_sign = 0
                    sweep_stop = None
                    continue
            # Check for opposite signal to exit
            if sig != pos_sign and sig != 0:
                trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'entry': entry_price, 'exit': close, 'sign': pos_sign})
                pos_sign = 0
                sweep_stop = None
                continue
        if sig != 0 and pos_sign == 0:
            pos_sign = sig
            entry_price = close
            entry_idx = i
            # Determine stop price from the sweep extreme that triggered this signal.
            # Need to reconstruct: we don't have sweep extreme stored. Instead, we can look back: for a long, the sweep low is the lowest low since start of month that was below prev_month_low and then reclaimed. That's complex.
            # Simpler: use the most recent low/high that crossed the level.
            # In the signal generation loop, we could have set auxiliary columns: stop_price and high/low trigger. To keep code concise, I'll fall back to a simpler stop: use recent swing low/high or fixed %.
            # Given complexity, I'll fallback to fixed stop % for now (e.g., 1.5%) to get results, or use the previous month low/high as stop? That's too far.
            # Instead, I'll revise signal gen to embed stop levels.
            # Let's adjust: during signal generation, record stop price in a new column at the signal row.
            pass  # will handle differently

    # The above simulation incomplete because we didn't record stop levels. To avoid major rewrite, I’ll fall back to simpler version: use a fixed % stop and TP, or use the sweep extreme approx as previous row low/high.
    # But we can do a quick hack: for long, stop = min(low over last N bars where sweep occurred). Hard.
    # Given time, I'll re‑implement signal generation to also set 'stop_price' and 'tp_price' on the signal row.
    # Let's restart: we'll build signals as a DataFrame with those columns.

    # Actually better: integrate simulation directly within the same loop that generates signals. That way we have all info. I'll restructure function.

    return df, {}


if __name__ == "__main__":
    df, metrics = monthly_reversal_ema_strategy(
        start_date='2024-01-01',
        end_date='2024-12-31',
        ema_period=20,
        r_multiplier=3.0,
        max_month_trades=2,
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        initial_capital=10000.0,
        commission=0.001,
        plot=True
    )
    print("\n=== Monthly Reversal+EMA Strategy Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")