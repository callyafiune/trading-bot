# Validation Pipeline (Offline)

This document explains how to run the offline validation package that reuses `src/bot/pattern_mining`.

## Goal

Validate if pattern-mining features, events, labels, and edge score carry real and stable signal under:

- no feature lookahead
- temporal validation
- realistic execution assumptions (fees, slippage, TP/SL)
- robustness sweeps

## Execution order

The CLI enforces this exact order:

1. `sanity`
2. `baseline`
3. `event_study`
4. `regimes`
5. `walkforward`
6. `sim_execution`
7. `robustness`

## Usage

```bash
python scripts/run_validation.py --ohlcv data.csv
```

With optional OI/CVD/Liquidations:

```bash
python scripts/run_validation.py \
  --ohlcv data/ohlcv.csv \
  --oi data/oi.csv \
  --cvd data/cvd.csv \
  --liq data/liquidations.csv \
  --timeframe 1h \
  --output_dir reports
```

Override defaults:

```bash
python scripts/run_validation.py \
  --ohlcv data/ohlcv.csv \
  --horizon 4 \
  --target 0.009 \
  --wick 0.7 \
  --min_support 300 \
  --train_window 1440 \
  --test_window 240 \
  --slippage_bps 3 \
  --fee_bps 5 \
  --cooldown 2 \
  --max_trades_per_day 20 \
  --seed 7
```

Compare only stages (6) and (7) before/after regime adjustments:

```bash
python scripts/run_validation.py \
  --ohlcv data/ohlcv.csv \
  --compare_before_after \
  --before_config regime_adjustments=0 \
  --after_config regime_adjustments=1
```

## Inputs

Required in OHLCV CSV:

- `open_time` (or `timestamp`/`ts`/`datetime`/`date`)
- `open`, `high`, `low`, `close`, `volume`

Optional CSVs are merged by timestamp:

- OI (`oi` or `open_interest`)
- CVD (`cvd`)
- Liquidations (`liquidations`)

If OI/CVD/Liquidations are missing, dependent pieces are skipped gracefully.

## Outputs

Each run creates:

- `reports/<run_id>/...`

Where `run_id` is UTC short ISO (`YYYYMMDDTHHMMSS`).

Generated files:

- `sanity_report.json`
- `sanity_preview_head.csv`
- `sanity_preview_tail.csv`
- `baseline_metrics.json`
- `baseline_trades.csv`
- `baseline_label_metrics.json`
- `event_study.csv`
- `event_study_top20.txt`
- `regime_summary.json`
- `regime_event_study.csv`
- `walkforward_splits.csv`
- `walkforward_metrics.csv`
- `walkforward_feature_importance.csv`
- `walkforward_last_confusion.csv`
- `execution_metrics.json`
- `execution_trades.csv`
- `equity_curve.csv`
- `robustness_grid.csv`
- `robustness_top_configs.csv`

When `--compare_before_after` is enabled:

- `reports/<run_id>/before/` contains stage (6) and (7) outputs
- `reports/<run_id>/after/` contains stage (6) and (7) outputs
- `reports/<run_id>/comparison_summary.json`
- `reports/<run_id>/comparison_table.csv`

## Notes

- Features are computed with no future leakage.
- Labels and execution simulation use future candles by design.
- Walk-forward uses temporal windows only (no shuffle).
- Random seed is fixed for reproducibility.
- Robustness top configs apply an execution max drawdown filter before ranking by expectancy.
