# LN Markets Technical Analysis Bot

A simple bot for algorithmic trading with [Trading View Technical Analysis](https://www.tradingview.com/symbols/XBTUSD/technicals/) signals on [LN Markets](https://lnmarkets.com/).

> :warning: CAUTION: Use at your own risk. This repo is meant to be a reference and has been created for educational purposes only. 

Please use carefully, preferably on Testnet or with small amounts.

## Install

Download this Github repository and install dependencies.

### Using uv (recommended)
```
uv sync
```
This will create a virtual environment and install all dependencies.

To run the bot:
```
uv run python main.py
```

### Using pip (legacy)
```
pip install -r requirements.txt
```
This code uses the LN Markets Python SDK v3 (`lnmarkets-sdk`), compatible with LN Markets API v3.
Install/update it with `pip install -U lnmarkets-sdk`.

## Authentication

> For authentication, you need your [LN Markets API](https://docs.lnmarkets.com/api/v1/) **Key**, **Secret**, and **Passphrase**.

Without them, you will not be able to authenticate.

> :warning: **Never share your API Key, Secret or Passphrase**

Make a copy of `example.configuration.yml` and rename it to `configuration.yml`, then fill in your LN Markets API credentials.
```
cp example.configuration.yml configuration.yml
```

You can also add the parameter network with 'testnet' for [LN Markets Testnet](https://testnet.lnmarkets.com/) and 'mainnet' for [LN Markets mainnet](https://lnmarkets.com/).

**Important:** The `configuration.yml` file is included in `.gitignore` and will not be synced to prevent accidental exposure of your credentials.

## Strategies

Current available strategies are:
- ta_summary: use the [Trading View Technical Analysis](https://www.tradingview.com/symbols/XBTUSD/technicals/) summary indicator based on 27 signals (oscillators and moving averages) to open a long or short future position.
- macd_stochrsi: MACD crossover confirmed by StochRSI crossover in oversold/overbought zones.
- tbd_3_level: approximation of the Trade By Design 3-level reversal concept (W/M structure + 3-push exhaustion + MTF EMA bias + liquidity filters).
- trend_exhaustion_rider: trend breakout rider with long/short entries, exhaustion exits, and trailing-style profit protection after a minimum target move.
- ma7_rsi_stoch: MA 7/21/50 with RSI + StochRSI pre-signal logic, optional long-only mode, and trailing exit after minimum move.
- More to come

## Backtesting

Use `backtest.py` to run historical tests from a CSV file with OHLCV columns.

### Required CSV format
- Required columns: `Open`, `High`, `Low`, `Close`
- Optional columns: `Volume`, one date column among `timestamp`, `date`, `datetime`, `time`, `open_time`
- Column names are case-insensitive.

### Example command
```bash
uv run python backtest.py \
  --data data/2023-2025-ohlcv_1h.csv \
  --strategy strategies.backtests.ma_crossover_50_200:MACrossover50200 \
  --cash 10000 \
  --commission 0.0005 \
  --leverage 2 \
  --param fast=50 \
  --param slow=200 \
  --export-trades trades.csv
```

Add `--plot` to open the backtest chart.
By default the runner uses fractional sizing (`FractionalBacktest`), which is usually better for BTC/USD. Use `--no-fractional` to disable it.

### MACD + StochRSI backtest example
```bash
uv run python backtest.py \
  --data data/2023-2025-ohlcv_1h.csv \
  --strategy strategies.backtests.macd_stochrsi:MACDStochRSI \
  --cash 10000 \
  --commission 0.0005 \
  --leverage 2 \
  --param macd_fast=12 \
  --param macd_slow=26 \
  --param macd_signal=9 \
  --param rsi_period=14 \
  --param stoch_period=14 \
  --param stoch_smooth_k=3 \
  --param stoch_smooth_d=3 \
  --param stoch_oversold=40 \
  --param stoch_overbought=60 \
  --export-trades trades_macd_stochrsi.csv
```

### TBD 3-level backtest example
```bash
uv run python backtest.py \
  --data data/2023-2025-ohlcv_1h.csv \
  --strategy strategies.backtests.tbd_3_level:TBDThreeLevel \
  --cash 10000 \
  --commission 0.0005 \
  --leverage 2 \
  --plot \
  --param htf_bias_tf=1D \
  --param require_weekend_consolidation=false \
  --param rr_ratio=2.5 \
  --param sl_atr_mult=1.2 \
  --export-trades trades_tbd_3_level.csv
```

Note: this is a public-information approximation of the TBD concept, not the proprietary paid indicator suite.

### Trend Exhaustion Rider backtest example
```bash
uv run python backtest.py \
  --data data/2023-2025-ohlcv_1h.csv \
  --strategy strategies.backtests.trend_exhaustion_rider:TrendExhaustionRider \
  --cash 10000 \
  --commission 0.0005 \
  --leverage 2 \
  --plot \
  --param profile=balanced \
  --param fast_ema=50 \
  --param slow_ema=200 \
  --param breakout_lookback=160 \
  --param htf_bias_tf=1D \
  --param htf_ema_period=200 \
  --param sl_atr_mult=2.0 \
  --param min_target_pct=0.025 \
  --param trail_pct=0.015 \
  --export-trades trades_trend_exhaustion_rider.csv
```

`trend_exhaustion_rider` supports strategy profiles via `--param profile=conservative|balanced|aggressive`.

### MA7 RSI Stoch backtest example
```bash
uv run python backtest.py \
  --data data/2023-2025-ohlcv_1h.csv \
  --strategy strategies.backtests.ma7_rsi_stoch:MA7RSIStoch \
  --cash 10000 \
  --commission 0.0005 \
  --leverage 2 \
  --plot \
  --param rsi_smooth=7 \
  --param stoch_len=21 \
  --param stoch_smooth_k=7 \
  --param stoch_smooth_d=7 \
  --param rsi_entry_floor=45 \
  --param rsi_confirm=58 \
  --param allow_shorts=false \
  --param min_target_pct=0.04 \
  --param trail_pct=0.015 \
  --param hard_stop_pct=0.02 \
  --export-trades trades_ma7_rsi_stoch.csv
```

`ma7_rsi_stoch` supports strategy profiles via `--param profile=long_only_safe|balanced|with_shorts`.

### Where to put strategies (no logic duplication)
- Shared indicator/signal logic: `strategies/signals/`
- Backtest wrapper classes: `strategies/backtests/`
- Live LN Markets execution classes: `strategies/`

Keep the indicator logic in `strategies/signals/*` and call it from both wrappers, so you only maintain one trading rule implementation.

### Adding new backtest strategies
Create a class that inherits from `backtesting.Strategy`, then pass it with `--strategy module.path:ClassName`.

```python
from backtesting import Strategy

class MyStrategy(Strategy):
    def init(self):
        ...

    def next(self):
        ...
```

More details below.

Open the 'configuration.yml' file.
Then choose the strategy you want for your trading bot setting True to the corresponding variable (and False to others).
```
# Example
Strategies: 
  ta_summary: True
```

## Parameters

In 'configuration.yml', you can define some parameters to customize your strategy.

```
# Example
ta_summary:
  quantity: 10 #type=int, min=1, quantity of each open position
  leverage: 10 #type=int, min=1, max=100
  interval: '5m' #type=string, available interval between 2 TA summary signals: "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1W", "1M"
  timeout: 60 #type=int, min=1, number of minutes the trading strat will be running  
```

## Run program

Open 'main.py' file and simply run it. If there is no error, you should see this message in the terminal:
```
Connection to LN Markets ok!
```
If not, double check the API Key, Secret, and Passphrase you  entered.

[LN Markets API](https://docs.lnmarkets.com/api/v1/) has requests limits (30 requests per minute).

## ta_summary

How does it work?  

Once the bot is launched, it follows [Trading View Technical Analysis](https://www.tradingview.com/symbols/XBTUSD/technicals/) summary indicator based on 27 signals (oscillators and moving averages) to run a directional strategy. The bot opens and keeps running a long Future position while the signal is "STRONG_BUY" and remains at least "BUY", and a short Future position while the signal is "STRONG_SELL" and remains at least "SELL", and close the position otherwise..
While the bot is running, you can have either 0 or 1 position running maximum. 

### Parameters to customize the bot
- quantity: the quantity (in USD) for the position running
- leverage: the leverage for the position running
- take profit: the coefficient to apply to the entry price to compute the take profit level
- stop loss: the coefficient to apply to the entry price to compute the stop loss level
- interval: interval between 2 TA summary signals: "1m" for 1 minute, "5m" for 5 minutes, "15m" for 15 minutes, "30m" for 30 minutes, "1h" for 1 hour, "2h" for 2 hours, "4h" for 4 hours, "1d" for 1 day, "1W" for 1 week, "1M" for 1 month
- interval: number of minutes the trading strat will be running

```
# Example
ta_summary:
  quantity: 10 #type=int, min=1, quantity of each open position
  leverage: 10 #type=int, min=1, max=100
  takeprofit: 0.02 #type=float, price to reach above (for long) or below (for short) entry price to take profit, 0.2 means take profit 20% above or below entry price 
  stoploss: 0.02 #type=float, price to reach below (for long) or above (for short) entry price to stop loss, 0.1 means stop loss 10% below or above entry price
  interval: '5m' #type=string, available interval between 2 TA summary signals: "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1W", "1M"
  timeout: 60 #type=int, min=1, number of minutes the trading strat will be running
```

## History of trades

After timeout, you will find all the bot's trades during its execution in the CSV 'df_closed_pos.csv' in the folder Strategies.

## To go further

Feel free to customize the bot and add your own strategies.

If you want to use more features from [LN Markets API](https://docs.lnmarkets.com/api/v1/), check out the full documentation.
