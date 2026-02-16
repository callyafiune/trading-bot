# bot-btc-margin-1h

MVP funcional de bot BTCUSDT 1H para Binance com:
- coleta de dados OHLCV
- feature engineering determinístico
- detector de regime (TREND/RANGE/CHAOS)
- estratégia baseline breakout + ATR
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

## Baixar dados
```bash
python -m bot fetch-data --start 2023-01-01 --end 2026-02-16
```

## Rodar backtest com artefatos de run
```bash
python -m bot backtest --data-path data/processed/BTCUSDT_1h_2023-01-01_2026-02-16.parquet --config config/settings.yaml
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

Para validar sem executar:
```bash
python -m bot backtest --data-path ... --dry-run
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
- Sem ML e sem order book nesta versão.
- Fricções: fee, slippage e juros de empréstimo por hora.
- Logs estruturados em JSON.

## Testes
```bash
pytest -q
```
