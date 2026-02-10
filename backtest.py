import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict

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

    def fetch_oi_history(self, start_timestamp: int, end_timestamp: int = None) -> pd.Series:
        """Fetch open interest history for the symbol using self.oi_symbol."""
        try:
            if not hasattr(self.exchange, 'fetch_open_interest_history'):
                print("Exchange does not support fetch_open_interest_history")
                return pd.Series(dtype='float64')
            all_oi = []
            since = start_timestamp
            tf_ms = self.exchange.parse_timeframe(self.timeframe) * 1000
            while True:
                try:
                    oi_chunk = self.exchange.fetch_open_interest_history(self.oi_symbol, since=since, limit=1000)
                except Exception as e:
                    print(f"Error fetching OI chunk: {e}")
                    break
                if not oi_chunk:
                    break
                all_oi.extend(oi_chunk)
                last_ts = oi_chunk[-1]['timestamp']
                since = last_ts + tf_ms
                if end_timestamp and since > end_timestamp:
                    break
                if len(oi_chunk) < 1000:
                    break
            if not all_oi:
                return pd.Series(dtype='float64')
            df_oi = pd.DataFrame(all_oi)
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
            if end_timestamp:
                end_ts = pd.to_datetime(end_timestamp, unit='ms')
                series = series[series.index <= end_ts]
            return series.sort_index()
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
                         oi_threshold: float = 0.0) -> pd.DataFrame:
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1   # Long
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Short
        df['filtered_signal'] = df['signal'].copy()
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
        use_oi_filter: bool = True,
        oi_periods: int = 1,
        oi_threshold: float = 0.0,
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
            start_ts = self.exchange.parse8601(f"{start_date}T00:00:00Z")
            end_ts = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
            oi_series = self.fetch_oi_history(start_ts, end_ts)
            if oi_series.empty:
                print("OI data unavailable; disabling OI filter")
                use_oi_filter = False
            else:
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
        df = self.generate_signals(df, use_trend_filter, use_oi_filter, oi_threshold)

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
    symbol: str = 'BTC/USDT:USDT',
    timeframe: str = '1h',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    use_trend_filter: bool = True,
    trend_ma_period: int = 200,
    use_oi_filter: bool = True,
    oi_periods: int = 1,
    oi_threshold: float = 0.0,
    plot: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced Moving Average Crossover with trend and open interest filters."""
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
        plot=plot
    )

if __name__ == "__main__":
    df, metrics = enhanced_ma_crossover_strategy(
        start_date='2023-01-01',
        end_date='2024-02-10',
        fast_period=20,
        slow_period=50,
        initial_capital=10000.0,
        commission=0.001,
        use_trend_filter=True,
        trend_ma_period=200,
        use_oi_filter=True,
        oi_periods=1,
        oi_threshold=0.0,
        plot=True
    )