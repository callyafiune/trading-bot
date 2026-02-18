# bot-btc-margin-1h

MVP funcional de bot BTCUSDT 1H para Binance com:
- coleta de dados OHLCV
- feature engineering determinístico
- detector de regime em 2 camadas (macro + micro) com histerese anti-churn
- estratégia baseline breakout + ATR
- modos opcionais de sinal: `ema`, `ema_macd`, `ml_gate`
- backtest com fricções realistas
- paper trading simulado
- conectores para execução real em margin Binance
- **infraestrutura de experiment loops** para gerar artefatos por run e comparar resultados

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## Configuração
Edite `config/settings.yaml` para ajustar símbolo, datas, risco, fricções e execução.

Novos defaults estruturais para o baseline breakout+ATR:
- `strategy_breakout.breakout_lookback_N: 72`
- `strategy_breakout.atr_k: 2.5`
- `regime.adx_trend_threshold: 28`
- `strategy_breakout.use_ma200_filter: true`
- `strategy_breakout.ma200_period: 200`

Filtro direcional MA200 (sem lookahead):
- LONG só entra quando `close[t] > ma_200[t]`
- SHORT só entra quando `close[t] < ma_200[t]`
- A decisão é feita no candle fechado `t`; execução permanece no `open[t+1]`.

## Baixar dados
```bash
python -m bot fetch-data --start 2023-01-01 --end 2026-02-16
```


## Baixar funding histórico (USDT-M perpetual)
```bash
python -m bot fetch-funding --start 2023-01-01 --end 2026-02-16 --symbol BTCUSDT
```

Artefatos gerados:
- `data/raw/funding/BTCUSDT_funding_2023-01-01_2026-02-16.parquet`
- `data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet`

O funding processado é alinhado para 1H com `forward-fill` seguro e flag de ausência.

## Rodar backtest com artefatos de run
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml
```

Backtest com funding e filtro habilitado (no YAML: `funding_filter.enabled: true`):
```bash
python -m bot backtest \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml
```

Também disponível:
```bash
python -m bot backtest \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml \
  --outdir runs \
  --run-name minha_run \
  --seed 42 \
  --tag baseline
```


Exemplos de modos estratégicos:
```bash
python -m bot backtest --data-path ... --mode ema
python -m bot backtest --data-path ... --mode ema_macd --atr-k 2.5 --adx-threshold 28
python -m bot backtest --data-path ... --mode ml_gate --ml-threshold 0.58 --use-ma200-filter
python -m bot backtest --data-path ... --short-only
python -m bot backtest --data-path ... --long-only
```

Para validar sem executar:
```bash
python -m bot backtest --data-path ... --dry-run
```

Comparação rápida before/after (smoke 2025-01-01 a 2025-04-01):
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --run-name baseline_after
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --run-name baseline_before --breakout-N 48 --adx-threshold 25 --no-use-ma200-filter
python -m bot compare --runs runs/<baseline_before> runs/<baseline_after>
```

Cada execução cria uma pasta única em `runs/` contendo:
- `config_used.yaml`
- `summary.json`
- `trades.csv`
- `equity.csv`
- `regime_stats.json`
- `direction_stats.json`
- `params_hash.txt`
- `run_meta.json`
- `metrics.md`

## Comparar runs
```bash
python -m bot compare --runs runs/<id1> runs/<id2>
```

Opcionalmente, salve a comparação:
```bash
python -m bot compare --runs runs/<id1> runs/<id2> --save-path runs/compare.json
```

## Rodar grid search simples
```bash
python -m bot grid \
  --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet \
  --config config/settings.yaml \
  --param strategy_breakout.breakout_lookback_N=48,72,96 \
  --param strategy_breakout.atr_k=2.0,2.5
```

Também aceita aliases:
```bash
python -m bot grid --data-path ... --param strategy.atr_k=2.0,2.5,3.0 --param breakout.breakout_lookback_N=48,72,96
```

## Rodar paper trading
```bash
python -m bot paper --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet
python -m bot paper --loop --sleep 60 --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet
```

## Rodar live (dry-run e real)
```bash
python -m bot live --dry-run
python -m bot live --no-dry-run
```

### Requisitos de API key Binance
- Enable Spot & Margin Trading
- Enable Margin Loan, Repay & Transfer
- Whitelist de IP recomendada.

## Observações de modelagem
- Sinal calculado no close de `t`; execução sempre no open de `t+1`.
- `ml_gate` usa classificador XGBoost com seleção de features por importância e validação walk-forward (double OOS).
- Fricções: fee, slippage e juros de empréstimo por hora.
- Logs estruturados em JSON.

## Testes
```bash
pytest -q
```


## Métricas de regime e switches
`summary.json` agora inclui:
- `trades_by_regime_final` e `pnl_by_regime_final`
- `regime_switch_count_macro`, `regime_switch_count_micro`, `regime_switch_count_total`
- `blocked_funding`, `blocked_macro`, `blocked_micro`, `blocked_chaos`

`regime_stats.json` inclui a distribuição dos regimes finais e contagem de switches.

## Fear & Greed (opcional)
```bash
python -m bot fetch-fng --start 2025-01-01 --end 2025-04-01
```

Saida:
- `data/raw/fng/fng_1d_2025-01-01_2025-04-01.parquet`
- `data/processed/fng_BTC_1h_2025-01-01_2025-04-01.parquet`

## Compare com multiplas runs
```bash
python -m bot compare --runs runs/A --runs runs/B --save-path runs/compare.json
```

## Experimentos recomendados
Smoke baseline vs funding gate:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_short_baseline --tag smoke
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --funding-path data/processed/funding_BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_short_funding --tag smoke
python -m bot compare --runs runs/smoke_short_baseline --runs runs/smoke_short_funding --save-path runs/compare_smoke_short_funding.json
```

Ablacao MA200:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --use-ma200-filter --run-name smoke_ma200_on --tag ablation
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --no-use-ma200-filter --run-name smoke_ma200_off --tag ablation
python -m bot compare --runs runs/smoke_ma200_off --runs runs/smoke_ma200_on --save-path runs/compare_ma200.json
```

Ablacao FNG:
```bash
python -m bot fetch-fng --start 2025-01-01 --end 2025-04-01
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --fng-path data/processed/fng_BTC_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_fng_on --tag fng
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2025-01-01_2025-04-01.parquet --config config/settings.yaml --short-only --run-name smoke_fng_off --tag fng
python -m bot compare --runs runs/smoke_fng_off --runs runs/smoke_fng_on --save-path runs/compare_fng.json
```

Router adaptativo:
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml --run-name full_router --tag router
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --funding-path data/processed/funding_BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml --run-name full_no_router --tag router --disable-router
python -m bot compare --runs runs/full_no_router --runs runs/full_router --save-path runs/compare_router.json
```
