import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class CCXTBacktester:
    """Backtester for BTC/USD trend following strategies using CCXT"""

    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframe='1h'):
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
        })
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_historical_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: now)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        end_time = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None

        all_ohlcv = []
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    since,
                    limit=1000
                )
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000

                if end_time and since > end_time:
                    break

                if len(ohlcv) < 1000:  # Less than limit means we reached the end
                    break

            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        if end_time:
            df = df[df.index <= pd.to_datetime(end_time)]

        return df

    def calculate_indicators(self, df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.DataFrame:
        """Calculate moving averages for trend following"""
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        df['returns'] = df['close'].pct_change()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover"""
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Long
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Short
        df['position'] = df['signal'].shift(1)  # Enter next bar
        return df

    def calculate_returns(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
        """Calculate strategy returns"""
        df['strategy_returns'] = df['position'] * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['portfolio_value'] = initial_capital * df['cumulative_returns']
        return df

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1) * 100
        annualized_return = ((1 + total_return/100) ** (365*24 / len(df)) - 1) * 100

        sharpe_ratio = np.sqrt(365*24) * df['strategy_returns'].mean() / df['strategy_returns'].std() if df['strategy_returns'].std() != 0 else 0

        rolling_max = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        win_rate = (df['strategy_returns'] > 0).sum() / len(df['strategy_returns'].dropna()) * 100

        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': len(df[df['signal'] != 0]),
            'Start Date': df.index[0].strftime('%Y-%m-%d'),
            'End Date': df.index[-1].strftime('%Y-%m-%d'),
        }

    def run_backtest(
        self,
        start_date: str,
        end_date: str = None,
        fast_period: int = 20,
        slow_period: int = 50,
        initial_capital: float = 10000.0,
        plot: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete backtest

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date (default: now)
            fast_period: Fast MA period
            slow_period: Slow MA period
            initial_capital: Starting capital
            plot: Whether to generate equity curve plot

        Returns:
            Tuple of (results DataFrame, metrics dictionary)
        """
        print(f"Fetching {self.symbol} data from {start_date} to {end_date or 'now'}...")
        df = self.fetch_historical_data(start_date, end_date)
        print(f"Fetched {len(df)} data points")

        print("Calculating indicators...")
        df = self.calculate_indicators(df, fast_period, slow_period)

        print("Generating signals...")
        df = self.generate_signals(df)

        print("Calculating returns...")
        df = self.calculate_returns(df, initial_capital)

        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)

        metrics = self.calculate_metrics(df)
        for key, value in metrics.items():
            print(f"{key}: {value}")

        print("="*50 + "\n")

        if plot:
            self.plot_results(df, fast_period, slow_period)

        return df, metrics

    def plot_results(self, df: pd.DataFrame, fast_period: int, slow_period: int):
        """Plot equity curve and signals"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Price and MAs
        axes[0].plot(df.index, df['close'], label='Price', alpha=0.7)
        axes[0].plot(df.index, df['fast_ma'], label=f'Fast MA ({fast_period})', alpha=0.7)
        axes[0].plot(df.index, df['slow_ma'], label=f'Slow MA ({slow_period})', alpha=0.7)
        axes[0].set_title(f'{self.symbol} Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Position
        axes[1].fill_between(df.index, 0, df['position'], where=df['position']>0, alpha=0.5, color='green', label='Long')
        axes[1].fill_between(df.index, 0, df['position'], where=df['position']<0, alpha=0.5, color='red', label='Short')
        axes[1].set_title('Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Portfolio value
        axes[2].plot(df.index, df['portfolio_value'], label='Portfolio Value', color='blue')
        axes[2].set_title('Equity Curve')
        axes[2].set_ylabel('Value ($)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def simple_ma_crossover_strategy(
    start_date: str,
    end_date: str = None,
    fast_period: int = 20,
    slow_period: int = 50,
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    initial_capital: float = 10000.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simple Moving Average Crossover Trend Following Strategy

    - When fast MA crosses above slow MA: Buy (Long)
    - When fast MA crosses below slow MA: Sell (Short)
    - Always in the market (long or short)
    """
    backtester = CCXTBacktester(exchange, symbol, timeframe)
    return backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        fast_period=fast_period,
        slow_period=slow_period,
        initial_capital=initial_capital,
        plot=True
    )


if __name__ == "__main__":
    # Example: Backtest from 2023-01-01 to today
    df, metrics = simple_ma_crossover_strategy(
        start_date='2023-01-01',
        end_date='2024-02-10',
        fast_period=20,
        slow_period=50,
        initial_capital=10000.0
    )