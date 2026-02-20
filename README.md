<p align="center">
  <img src="docs/banner.png" width="900" alt="BTCUSDT 1H Margin Bot"/>
</p>
# BTCUSDT 1H Margin Trading Bot

Experimental directional trading system for Bitcoin (BTCUSDT) on Binance Margin.

This project implements a configurable execution pipeline for evaluating
breakout-based trading strategies using:

-   volatility expansion (ATR)
-   ADX trend regime detection
-   optional MA200 directional bias
-   optional funding filter
-   optional market structure filter (HH/HL/LH/LL)
-   optional MSB (Market Structure Break)
-   optional ML trade filter (XGBoost)

Supports LONG and SHORT.

------------------------------------------------------------------------

## Project Goal

This repository is intended for:

-   testing directional breakout strategies
-   evaluating regime filters
-   studying structural gating (HH/HL/LH/LL)
-   measuring impact of MSB confirmation
-   comparing baseline vs filtered execution
-   running realistic backtests including margin frictions

It does **not** implement:

-   prediction models
-   market making
-   grid trading
-   arbitrage

All trades are signal-driven from completed 1H candles.

------------------------------------------------------------------------

## Signal and Execution Model

Signal generation occurs at:

    close[t]

Trade execution occurs at:

    open[t+1]

No intra-candle decisions are made.

No future candle data is used.

------------------------------------------------------------------------

## Baseline Strategy

LONG when:

-   price breaks recent N-bar high
-   ATR expansion exceeds configured threshold
-   ADX is above regime threshold

SHORT when:

-   price breaks recent N-bar low
-   ATR expansion exceeds configured threshold
-   ADX is above regime threshold

Optional:

    LONG only if close[t] > MA200[t]
    SHORT only if close[t] < MA200[t]

------------------------------------------------------------------------

## Market Structure Filter (optional)

Swing pivots detected using window:

    [i-left_bars, i+right_bars]

Pivot at `i` becomes available only at:

    i + right_bars

Structure state:

| State | Condition |
| --- | --- |
| BULLISH | HH + HL |
| BEARISH | LH + LL |
| NEUTRAL | otherwise |

------------------------------------------------------------------------

## MSB Filter (optional)

Bullish MSB:

    close[t] breaks last confirmed LH

Bearish MSB:

    close[t] breaks last confirmed HL

Break must exceed ATR buffer:

    min_break_atr

MSB state may persist for a configurable number of candles.

------------------------------------------------------------------------

## Backtest Assumptions

Simulations include:

-   taker trading fee
-   slippage
-   hourly borrow interest
-   funding rate
-   delayed execution (t+1)

Backtests are not frictionless.

------------------------------------------------------------------------

## ML Trade Filter (optional)

Trade signals may be filtered using:

-   XGBoost classification
-   feature importance selection
-   walk-forward validation

ML layer is applied after signal generation.

------------------------------------------------------------------------

## Running Modes

-   Backtest\
-   Paper trading\
-   Live execution (margin)

------------------------------------------------------------------------

## Setup

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

------------------------------------------------------------------------

## Fetch BTCUSDT 1H data

``` bash
python -m bot fetch-data --start 2023-01-01 --end 2026-02-16
```

Fetch historical funding:

``` bash
python -m bot fetch-funding --start 2023-01-01 --end 2026-02-16 --symbol BTCUSDT
```

------------------------------------------------------------------------

## Run Backtest

``` bash
python -m bot backtest \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml
```

Enable market structure:

``` bash
--ms-enable
--ms-gate-mode structure_trend
```

Enable MSB breakout:

``` bash
--msb-enable
--ms-gate-mode msb_only
```

------------------------------------------------------------------------

## Paper Trading

``` bash
python -m bot paper
```

Loop mode:

``` bash
python -m bot paper --loop --sleep 60
```

------------------------------------------------------------------------

## Live Trading

Dry run:

``` bash
python -m bot live --dry-run
```

Real execution:

``` bash
python -m bot live --no-dry-run
```

------------------------------------------------------------------------

## Outputs per Run

Each backtest produces:

    summary.json
    trades.csv
    equity.csv
    regime_stats.json
    direction_stats.json
    market_structure_stats.json
    pivots.csv
    metrics.md

------------------------------------------------------------------------

## Disclaimer

Margin trading carries liquidation risk.

Backtest performance does not guarantee live results.

Use paper trading before enabling live execution.
