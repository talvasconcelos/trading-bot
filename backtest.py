#!/usr/bin/env python
"""
Generic backtesting runner for trading strategies.

Usage:
    python backtest.py --start 2024-01-01 --end 2024-12-31 --strategy macrossover [--fast 50 --slow 200 --min-sep 0.005] [--data data.csv] [--capital 10000] [--commission 0.001] [--plot]

The backtest reads OHLCV data (from CSV if provided, else fetches from Binance via CCXT),
runs the selected strategy to generate signals, simulates trades, and prints performance metrics.
Outputs:
    - Console metrics
    - Equity curve CSV (backtest_results/equity_<timestamp>.csv)
    - Plot PNG (backtest_results/plot_<timestamp>.png) if --plot is set

All output files are placed under 'backtest_results/' which should be gitignored.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ccxt

# --- Strategy Interface ---
class Strategy:
    """Base strategy class. Subclasses must implement compute_signal()."""
    def compute_signal(self, df_window: pd.DataFrame) -> int:
        """
        Given a DataFrame window of historical data up to the current bar (including current),
        return trading signal:
            1  -> long
           -1  -> short
            0  -> neutral / hold
        """
        raise NotImplementedError

# --- MACrossover Strategy ---
class MACrossover(Strategy):
    def __init__(self, fast_period=50, slow_period=200, min_separation=0.005):
        self.fast = fast_period
        self.slow = slow_period
        self.min_sep = min_separation

    def compute_signal(self, df_window):
        if len(df_window) < self.slow:
            return 0
        fast_ma = df_window['close'].iloc[-self.fast:].mean()
        slow_ma = df_window['close'].iloc[-self.slow:].mean()
        sep = (fast_ma - slow_ma) / slow_ma
        if sep >= self.min_sep:
            return 1
        elif sep <= -self.min_sep:
            return -1
        else:
            return 0

# --- Backtester ---
class Backtester:
    def __init__(self, initial_capital=10000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame, strategy: Strategy):
        """
        Run backtest on the given DataFrame (must have a 'close' column).
        Returns a DataFrame with signals, positions, portfolio values, and metrics dict.
        """
        df = df.sort_index().copy()
        n = len(df)

        # Generate signals for each bar using all prior data (including current)
        signals = np.zeros(n, dtype=int)
        for i in range(n):
            window = df.iloc[:i+1]
            sig = strategy.compute_signal(window)
            signals[i] = sig
        df['signal'] = signals

        # Position is previous signal (to avoid lookahead)
        df['position'] = df['signal'].shift(1).fillna(0).astype(int)

        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        df['strategy_ret'] = df['position'] * df['returns']

        # Deduct commissions on position changes (entry and exit)
        position_change = df['position'].diff().abs().fillna(0)
        df['strategy_ret'] -= position_change * self.commission

        # Cumulative returns and portfolio
        df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
        df['portfolio'] = self.initial_capital * df['cum_ret']

        # --- Metrics ---
        total_ret = (df['portfolio'].iloc[-1] / self.initial_capital - 1) * 100
        # Annualized return (assuming daily data: 252, hourly: 365*24, but we infer from index)
        # Approximate using number of periods per year
        period_seconds = (df.index[1] - df.index[0]).total_seconds() if len(df) > 1 else 86400
        periods_per_year = 365 * 24 * 3600 / period_seconds if period_seconds > 0 else 252
        n_years = n / periods_per_year
        if n_years > 0:
            ann_ret = ((1 + total_ret/100) ** (1 / n_years) - 1) * 100
        else:
            ann_ret = 0.0
        # Sharpe
        if df['strategy_ret'].std() != 0:
            sharpe = np.sqrt(periods_per_year) * df['strategy_ret'].mean() / df['strategy_ret'].std()
        else:
            sharpe = 0.0
        # Max drawdown
        rolling_max = df['portfolio'].cummax()
        drawdown = (df['portfolio'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        # Win rate (trades)
        # Identify trade periods: change in position != 0 indicates entry/exit
        # Count only completed trades (entry followed by opposite or exit)
        # Simplify: compute per-period returns where position != 0; count positive returns among those
        active = df['position'] != 0
        if active.any():
            win_rate = (df.loc[active, 'strategy_ret'] > 0).sum() / active.sum() * 100
            total_trades = active.diff().abs().clip(upper=1).sum() / 2  # Each entry+exit pair counts as one trade; rough
            # Actually better: count entries where position changes from 0 to non-zero, and count corresponding exit when position returns to 0 or flips
            entries = (df['position'] != 0) & (df['position'].shift(1) == 0)
            exits = (df['position'] == 0) & (df['position'].shift(1) != 0)
            total_trades = entries.sum()
            if exits.sum() > entries.sum():
                total_trades = exits.sum()
        else:
            win_rate = 0.0
            total_trades = 0

        metrics = {
            'Total Return (%)': round(total_ret, 2),
            'Annualized Return (%)': round(ann_ret, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown (%)': round(max_dd, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': int(total_trades),
            'Start Date': df.index[0].strftime('%Y-%m-%d'),
            'End Date': df.index[-1].strftime('%Y-%m-%d'),
            'Periods': n,
        }
        return df, metrics

def fetch_data_from_ccxt(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    """
    Fetch OHLCV data from Binance using CCXT.
    start_date and end_date are strings like '2024-01-01'.
    Returns a DataFrame with datetime index and columns: open, high, low, close, volume.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(f"{start_date}T00:00:00Z") if start_date else None
    until = exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            since = last_ts + exchange.parse_timeframe(timeframe) * 1000
            if until and since > until:
                break
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    if not all_ohlcv:
        raise ValueError("No data fetched")
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    if end_date:
        end_ts = pd.to_datetime(end_date)
        df = df[df.index <= end_ts]
    return df[['open','high','low','close','volume']]

def ensure_output_dir():
    out_dir = 'backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main():
    parser = argparse.ArgumentParser(description='Generic backtest runner.')
    parser.add_argument('--strategy', required=True, choices=['macrossover'], help='Strategy to test')
    parser.add_argument('--start', required=True, help='Start date, e.g., 2024-01-01')
    parser.add_argument('--end', default=None, help='End date, e.g., 2024-12-31 (default: today)')
    parser.add_argument('--timeframe', default='1h', help='Data timeframe (1h, 4h, 1d, etc.)')
    parser.add_argument('--fast', type=int, default=50, help='Fast MA period for MACrossover')
    parser.add_argument('--slow', type=int, default=200, help='Slow MA period for MACrossover')
    parser.add_argument('--min-sep', type=float, default=0.005, help='Minimum separation ratio (0.005=0.5%)')
    parser.add_argument('--data', default=None, help='Path to a CSV file with OHLCV data (optional)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (e.g., 0.001 for 0.1%)')
    parser.add_argument('--plot', action='store_true', help='Generate equity curve plot')
    args = parser.parse_args()

    # Load data
    if args.data:
        df = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
    else:
        print(f"Fetching {args.timeframe} data from Binance...")
        df = fetch_data_from_ccxt(symbol='BTC/USDT', timeframe=args.timeframe, start_date=args.start, end_date=args.end)

    # Instantiate strategy
    if args.strategy == 'macrossover':
        strategy = MACrossover(fast_period=args.fast, slow_period=args.slow, min_separation=args.min_sep)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Run backtest
    bt = Backtester(initial_capital=args.capital, commission=args.commission)
    print("Running backtest...")
    results_df, metrics = bt.run(df, strategy)

    # Print metrics
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*50 + "\n")

    # Save results
    out_dir = ensure_output_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    equity_path = os.path.join(out_dir, f"equity_{timestamp}.csv")
    results_df.to_csv(equity_path)
    print(f"Equity curve saved to: {equity_path}")

    if args.plot:
        plt.figure(figsize=(12,6))
        plt.plot(results_df.index, results_df['portfolio'], label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = os.path.join(out_dir, f"plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()