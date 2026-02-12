# Strategies Overview

This folder contains live strategy runners (`strategies/*.py`), backtest wrappers (`strategies/backtests/*.py`), and shared signal logic (`strategies/signals/*.py`). The intended pattern is: keep indicator/signal rules in `signals/`, then reuse them from both live and backtest wrappers so behavior stays aligned.

For each strategy below, focus on: trade frequency, max drawdown, profit factor, and whether entries happen in the market regime the strategy is designed for.

## `ta_summary`

`ta_summary.py` is a sentiment-driven strategy based on TradingView's technical summary (`STRONG_BUY`, `BUY`, `SELL`, `STRONG_SELL`). It opens long on strong bullish signals, short on strong bearish signals, and closes when conviction weakens or flips. This strategy delegates market interpretation to TradingView, so it behaves more like a directional signal follower than a price-structure model.

When evaluating it, look at signal lag and regime sensitivity: it can perform well in persistent trends and struggle in choppy periods. Also monitor API/data dependency risk, since entries rely on external summary output rather than purely local OHLCV calculations.

## `ma_crossover_50_200`

`ma_crossover_50_200.py` (live) and `backtests/ma_crossover_50_200.py` (backtest) use shared logic from `signals/ma_crossover_50_200.py`. The idea is simple trend-following: when fast MA is above slow MA (with optional minimum separation), bias is long; when below, bias is short. It is easy to reason about and usually robust, but slower to react around turning points.

What to look for is the balance between responsiveness and noise. Lower separation increases trade count but raises whipsaws; higher separation reduces false positives but may enter late. Key metrics to compare while tuning are win rate vs. drawdown and whether profits come from a few big trends or many small moves.

## `macd_stochrsi`

`macd_stochrsi.py` combines momentum shift (MACD crossover) with oscillator confirmation (StochRSI cross in oversold/overbought zones). Shared logic lives in `signals/macd_stochrsi.py`; wrappers in live/backtest just execute/position-manage around that signal. The design aims to reduce low-quality MACD crosses by requiring extra confirmation.

While tuning, watch for over-filtering vs. over-trading. Tight thresholds can produce very few trades; loose thresholds can trigger too often in ranges. Check not only total return, but also trade count stability and whether performance degrades sharply when you slightly change threshold values.

## `tbd_3_level`

`tbd_3_level.py` is an approximation of the public "TBD 3-level" concept: multi-timeframe EMA bias, W/M-style reversal structure, exhaustion via repeated pushes, liquidity sweep/wick filters, vector-candle context, and session/weekend filters. Shared signal construction is in `signals/tbd_3_level.py`, with ATR/RR position logic handled by the backtest/live wrappers.

This is the most parameterized strategy, so focus on robustness over headline return. Check if performance persists across different date slices (not just one backtest window), and watch for low-sample overfitting. A good sign is reasonable trade count, controlled drawdown, and similar behavior when small parameter perturbations are applied.

## `trend_exhaustion_rider`

`trend_exhaustion_rider.py` is a momentum trend strategy designed for BTC futures behavior: it enters on breakout continuation with EMA trend and higher-timeframe bias, supports both long and short, and exits when momentum is exhausted. In backtests it activates trailing protection only after a minimum favorable move (default +2.5%), then tightens risk with a dynamic stop while allowing trends to continue.

What to look for is the return/drawdown profile under leverage and whether the trailing activation is too early or too late. Tune `min_target_pct`, `trail_pct`, and breakout filters together: smaller targets lock gains earlier but can cap big winners; wider trailing can catch longer runs but gives back more on reversals.

This strategy now supports a profile switch (`conservative`, `balanced`, `aggressive`) so you can quickly change behavior without manually editing every parameter. Start with `balanced` in paper trading, then move to `conservative` if drawdown is too high or `aggressive` only if you accept materially higher volatility.

## `ma7_rsi_stoch`

`ma7_rsi_stoch.py` implements your MA 7/21/50 + RSI + StochRSI idea as a dedicated strategy. Long entries require StochRSI momentum turning positive (`%K` crossing above `%D`), RSI above a floor and confirmation threshold, and price closing above MA7. Exit logic is intentionally simple: if StochRSI inverts or price closes below MA7, the long is closed. Shorts are optional and can be disabled (`allow_shorts=false`) for a cleaner long-only version.

What to watch is signal density versus quality. On 1h data this setup can fire too often unless smoothing and RSI thresholds are strict; on higher timeframes it usually gives fewer, cleaner entries. Tune `rsi_confirm`, `stoch_smooth_k/d`, and `allow_shorts` first, then tune trailing/stop parameters (`min_target_pct`, `trail_pct`, `hard_stop_pct`) to shape drawdown.

This strategy also supports profiles: `long_only_safe`, `balanced`, and `with_shorts`. Use `balanced` as default, `long_only_safe` to reduce churn and drawdown, and `with_shorts` only when you explicitly want symmetric long/short exposure.
