# BTCUSDT 1H Margin Trading Bot

Experimental directional trading system for Bitcoin (BTCUSDT) on Binance Margin.

This project implements a configurable execution pipeline for evaluating breakout-based trading strategies using:

-   volatility expansion (ATR)
-   ADX trend regime detection
-   optional MA200 directional bias
-   optional funding filter
-   optional market structure filter (HH/HL/LH/LL)
-   optional MSB (Market Structure Break)
-   optional ML trade filter (XGBoost)

------------------------------------------------------------------------

Functional MVP of a BTCUSDT 1H Binance bot with:
- OHLCV data collection
- deterministic feature engineering
- 2-layer regime detector (macro + micro) with anti-churn hysteresis
- breakout + ATR baseline strategy
- optional signal modes: `ema`, `ema_macd`, `ml_gate`
- backtest with realistic frictions
- simulated paper trading
- connectors for live Binance margin execution
- experiment loop infrastructure to persist run artifacts and compare results

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## Configuration
Edit `config/settings.yaml` to configure symbol, date range, risk, frictions, and execution.

Baseline breakout+ATR structural defaults:
- `strategy_breakout.breakout_lookback_N: 72`
- `strategy_breakout.atr_k: 2.5`
- `regime.adx_trend_threshold: 28`
- `strategy_breakout.use_ma200_filter: true`
- `strategy_breakout.ma200_period: 200`

MA200 directional filter (no lookahead):
- LONG only when `close[t] > ma_200[t]`
- SHORT only when `close[t] < ma_200[t]`
- Decision is made on closed candle `t`; execution remains at `open[t+1]`

## Download data
```bash
python -m bot fetch-data --start 2023-01-01 --end 2026-02-16
```

## Download historical funding (USDT-M perpetual)
```bash
python -m bot fetch-funding --start 2023-01-01 --end 2026-02-16 --symbol BTCUSDT
```

Generated artifacts:
- `data/raw/funding/BTCUSDT_funding_2023-01-01_2026-02-16.parquet`
- `data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet`

Processed funding is aligned to 1H with safe forward-fill and missing-data flag.

## Run backtest with run artifacts
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml
```

Backtest with funding and filter enabled (`funding_filter.enabled: true`):
```bash
python -m bot backtest \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml
```

Also available:
```bash
python -m bot backtest \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml \
  --outdir runs \
  --run-name my_run \
  --seed 42 \
  --tag baseline
```

Strategy mode examples:
```bash
python -m bot backtest --data-path ... --mode ema
python -m bot backtest --data-path ... --mode ema_macd --atr-k 2.5 --adx-threshold 28
python -m bot backtest --data-path ... --mode ml_gate --ml-threshold 0.58 --use-ma200-filter
python -m bot backtest --data-path ... --short-only
python -m bot backtest --data-path ... --long-only
```

Validation without execution:
```bash
python -m bot backtest --data-path ... --dry-run
```

Quick before/after comparison (smoke, 2025-01-01 to 2025-04-01):
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --run-name baseline_after
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --run-name baseline_before --breakout-N 48 --adx-threshold 25 --no-use-ma200-filter
python -m bot compare --runs runs/<baseline_before> runs/<baseline_after>
```

Each run creates a unique folder in `runs/` with:
- `config_used.yaml`
- `summary.json`
- `trades.csv`
- `equity.csv`
- `regime_stats.json`
- `direction_stats.json`
- `params_hash.txt`
- `run_meta.json`
- `metrics.md`

## Compare runs
```bash
python -m bot compare --runs runs/<id1> runs/<id2>
```

Optional save:
```bash
python -m bot compare --runs runs/<id1> runs/<id2> --save-path runs/compare.json
```

## Run simple grid search
```bash
python -m bot grid \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml \
  --param strategy_breakout.breakout_lookback_N=48,72,96 \
  --param strategy_breakout.atr_k=2.0,2.5
```

Alias support:
```bash
python -m bot grid --data-path ... --param strategy.atr_k=2.0,2.5,3.0 --param breakout.breakout_lookback_N=48,72,96
```

## Run paper trading
```bash
python -m bot paper --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet
python -m bot paper --loop --sleep 60 --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet
```

## Run live (dry-run and real)
```bash
python -m bot live --dry-run
python -m bot live --no-dry-run
```

### Binance API key requirements
- Enable Spot & Margin Trading
- Enable Margin Loan, Repay & Transfer
- IP whitelist recommended

## Modeling notes
- Signal is computed at close of `t`; execution always at open of `t+1`
- `ml_gate` uses XGBoost classification with feature-importance selection and walk-forward validation
- Frictions: fee, slippage, and hourly borrow interest
- Structured JSON logs

## Tests
```bash
pytest -q
```

## Regime metrics and switches
`summary.json` includes:
- `trades_by_regime_final` and `pnl_by_regime_final`
- `regime_switch_count_macro`, `regime_switch_count_micro`, `regime_switch_count_total`
- `blocked_funding`, `blocked_macro`, `blocked_micro`, `blocked_chaos`

`regime_stats.json` includes final regime distribution and switch counts.

## Fear & Greed (optional)
```bash
python -m bot fetch-fng --start 2025-01-01 --end 2025-04-01
```

Output:
- `data/raw/fng/fng_1d_2025-01-01_2025-04-01.parquet`
- `data/processed/fng_BTC_1h_2025-01-01_2025-04-01.parquet`

## Compare multiple runs
```bash
python -m bot compare --runs runs/A --runs runs/B --save-path runs/compare.json
```

## Recommended experiments
Baseline vs funding gate smoke:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_short_baseline --tag smoke
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --funding-path data/processed/funding_BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_short_funding --tag smoke
python -m bot compare --runs runs/smoke_short_baseline --runs runs/smoke_short_funding --save-path runs/compare_smoke_short_funding.json
```

MA200 ablation:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --use-ma200-filter --run-name smoke_ma200_on --tag ablation
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --no-use-ma200-filter --run-name smoke_ma200_off --tag ablation
python -m bot compare --runs runs/smoke_ma200_off --runs runs/smoke_ma200_on --save-path runs/compare_ma200.json
```

FNG ablation:
```bash
python -m bot fetch-fng --start 2025-01-01 --end 2025-04-01
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --fng-path data/processed/fng_BTC_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_fng_on --tag fng
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_fng_off --tag fng
python -m bot compare --runs runs/smoke_fng_off --runs runs/smoke_fng_on --save-path runs/compare_fng.json
```

Adaptive router:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml --run-name full_router --tag router
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml --run-name full_no_router --tag router --disable-router
python -m bot compare --runs runs/full_no_router --runs runs/full_router --save-path runs/compare_router.json
```

## Market Structure (HH/HL/LH/LL + MSB)
- Swing High/Low uses window `[i-left_bars, i+right_bars]`
- No lookahead: pivot at `i` is confirmed only at `i+right_bars`, then becomes available
- HH/LH: each new confirmed Swing High vs previous confirmed Swing High
- HL/LL: each new confirmed Swing Low vs previous confirmed Swing Low
- `ms_structure_state`:
  - `BULLISH` when relevant latest types are `HH + HL`
  - `BEARISH` when relevant latest types are `LH + LL`
  - `NEUTRAL` otherwise
- MSB:
  - `msb_bull`: `close[t]` breaks above latest confirmed `LH`
  - `msb_bear`: `close[t]` breaks below latest confirmed `HL`
  - `msb.min_break_atr` adds ATR anti-fakeout buffer
  - `msb.persist_bars` creates `msb_bull_active/msb_bear_active` for a few candles

New CLI flags:
```bash
--ms-enable / --no-ms-enable
--ms-left 3
--ms-right 3
--msb-enable / --no-msb-enable
--ms-gate-mode msb_only|structure_trend|hybrid
```

When `market_structure.enabled=true`, run outputs include:
- `market_structure_stats.json`
- `pivots.csv`

PowerShell commands (recent slice):
```powershell
python -m bot backtest `
  --data-path data/processed/BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --funding-path data/processed/funding_BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --config config/settings.yaml `
  --short-only `
  --run-name recent_short_baseline `
  --tag ms

python -m bot backtest `
  --data-path data/processed/BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --funding-path data/processed/funding_BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --config config/settings.yaml `
  --short-only `
  --ms-enable `
  --ms-gate-mode structure_trend `
  --run-name recent_short_ms_structure `
  --tag ms

python -m bot backtest `
  --data-path data/processed/BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --funding-path data/processed/funding_BTCUSDT_1h_2025-11-01_2026-02-16.parquet `
  --config config/settings.yaml `
  --short-only `
  --ms-enable `
  --msb-enable `
  --ms-gate-mode msb_only `
  --run-name recent_short_msb_only `
  --tag ms

python -m bot compare `
  --runs runs/recent_short_baseline `
  --runs runs/recent_short_ms_structure `
  --save-path runs/compare_recent_baseline_vs_ms_structure.json

python -m bot compare `
  --runs runs/recent_short_baseline `
  --runs runs/recent_short_msb_only `
  --save-path runs/compare_recent_baseline_vs_msb_only.json
```

Repeat on full dataset:
- `full_short_baseline`
- `full_short_ms_structure`
- `full_short_msb_only`