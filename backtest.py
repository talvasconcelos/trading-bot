import argparse
import importlib
from pathlib import Path

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest


DATE_COLUMNS = ("timestamp", "date", "datetime", "time", "open_time")
OHLCV_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def parse_cli_value(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        return raw


def parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        params[key] = parse_cli_value(value)
    return params


def load_strategy(strategy_path: str) -> type[Strategy]:
    if ":" in strategy_path:
        module_name, class_name = strategy_path.split(":", 1)
    else:
        module_name, class_name = strategy_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    if not isinstance(strategy_cls, type) or not issubclass(strategy_cls, Strategy):
        raise TypeError(f"{strategy_path} is not a backtesting.Strategy class.")
    return strategy_cls


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}.")

    lower_to_original = {col.lower(): col for col in df.columns}

    date_col = next((lower_to_original[c] for c in DATE_COLUMNS if c in lower_to_original), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.dropna(subset=[date_col]).set_index(date_col)
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    rename_map = {}
    for alias, canonical in OHLCV_ALIASES.items():
        if alias in lower_to_original:
            rename_map[lower_to_original[alias]] = canonical
    df = df.rename(columns=rename_map)

    for required in ("Open", "High", "Low", "Close"):
        if required not in df.columns:
            raise ValueError(
                f"Missing '{required}' column in {csv_path}. "
                "Expected columns like Open/High/Low/Close."
            )

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    parser = argparse.ArgumentParser(description="Run backtests for Strategy subclasses.")
    parser.add_argument("--data", required=True, help="Path to CSV file containing OHLCV candles.")
    parser.add_argument(
        "--strategy",
        required=True,
        help=(
            "Strategy class path. Example: "
            "strategies.backtests.ma_crossover_50_200:MACrossover50200"
        ),
    )
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash (quote currency).")
    parser.add_argument("--commission", type=float, default=0.0005, help="Per-trade commission.")
    parser.add_argument("--leverage", type=float, default=1.0, help="Account leverage. 1.0 disables margin.")
    parser.add_argument("--hedging", action="store_true", help="Allow long and short positions at once.")
    parser.add_argument("--trade-on-close", action="store_true", help="Fill market orders on candle close.")
    parser.add_argument(
        "--no-fractional",
        action="store_true",
        help="Use standard Backtest instead of FractionalBacktest.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Strategy parameter in key=value format. Can be passed multiple times.",
    )
    parser.add_argument("--plot", action="store_true", help="Render interactive backtest chart.")
    parser.add_argument("--export-trades", help="Optional CSV path to write executed trades.")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    strategy_cls = load_strategy(args.strategy)
    df = load_ohlcv(data_path)
    strategy_params = parse_params(args.param)

    margin = 1 / args.leverage if args.leverage > 0 else 1.0
    backtest_cls = Backtest if args.no_fractional else FractionalBacktest
    bt = backtest_cls(
        df,
        strategy_cls,
        cash=args.cash,
        commission=args.commission,
        margin=margin,
        hedging=args.hedging,
        trade_on_close=args.trade_on_close,
        exclusive_orders=not args.hedging,
        finalize_trades=True,
    )

    try:
        stats = bt.run(**strategy_params)
    except ValueError as exc:
        should_fallback = (
            not args.no_fractional
            and backtest_cls is FractionalBacktest
            and "read-only" in str(exc).lower()
        )
        if not should_fallback:
            raise

        print(
            "Warning: FractionalBacktest failed due to read-only indicator arrays. "
            "Falling back to standard Backtest."
        )
        bt = Backtest(
            df,
            strategy_cls,
            cash=args.cash,
            commission=args.commission,
            margin=margin,
            hedging=args.hedging,
            trade_on_close=args.trade_on_close,
            exclusive_orders=not args.hedging,
            finalize_trades=True,
        )
        stats = bt.run(**strategy_params)

    print(f"Strategy: {args.strategy}")
    print(f"Rows: {len(df)}")
    print(f"Range: {df.index[0]} -> {df.index[-1]}")
    print()
    for key in (
        "Return [%]",
        "Buy & Hold Return [%]",
        "# Trades",
        "Win Rate [%]",
        "Profit Factor",
        "Max. Drawdown [%]",
        "Sharpe Ratio",
    ):
        if key in stats:
            print(f"{key}: {stats[key]}")

    if args.export_trades:
        trades_path = Path(args.export_trades)
        stats["_trades"].to_csv(trades_path, index=False)
        print(f"\nTrades exported to: {trades_path}")

    if args.plot:
        bt.plot()


if __name__ == "__main__":
    main()
