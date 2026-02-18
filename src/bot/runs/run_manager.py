from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot.runs.schemas import BacktestSummary, RunMeta
from bot.runs.serializers import write_csv, write_json, write_text, write_yaml
from bot.utils.config import Settings


MIN_TRADE_COLUMNS = [
    "trade_id",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "qty",
    "notional",
    "stop_init",
    "stop_final",
    "reason_exit",
    "exit_reason",
    "pnl_gross",
    "pnl_net",
    "fees",
    "slippage",
    "interest",
    "regime_at_entry",
    "hold_hours",
    "holding_hours",
    "R_multiple_exit",
]


MIN_EQUITY_COLUMNS = ["timestamp", "equity", "position", "price", "drawdown"]


@dataclass
class RunContext:
    run_id: str
    run_name: str
    run_dir: Path
    params_hash: str


class RunManager:
    def __init__(self, outdir: str | Path = "runs") -> None:
        self.outdir = Path(outdir)

    @staticmethod
    def stable_hash_from_dict(payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def stable_hash_from_settings(settings: Settings) -> str:
        return RunManager.stable_hash_from_dict(settings.model_dump(mode="json"))

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return None

    @staticmethod
    def auto_run_name(settings: Settings, params_hash: str) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
        symbol = settings.symbol.lower()
        interval = settings.interval.lower()
        return f"{ts}_{symbol}_{interval}_{params_hash[:8]}"

    def build_context(self, settings: Settings, run_name: str | None = None) -> RunContext:
        params_hash = self.stable_hash_from_settings(settings)
        final_name = run_name or self.auto_run_name(settings, params_hash)
        run_dir = self.outdir / final_name
        return RunContext(run_id=final_name, run_name=final_name, run_dir=run_dir, params_hash=params_hash)

    def init_run(self, ctx: RunContext, *, settings: Settings, data_path: str, config_path: str, seed: int | None = None, tag: str | None = None) -> None:
        ctx.run_dir.mkdir(parents=True, exist_ok=False)
        config_dict = settings.model_dump(mode="json")
        write_yaml(ctx.run_dir / "config_used.yaml", config_dict)

        git_commit = self._git_commit()
        meta = RunMeta(
            run_id=ctx.run_id,
            run_name=ctx.run_name,
            created_at=datetime.now(UTC).isoformat(),
            outdir=str(self.outdir),
            data_path=data_path,
            config_path=config_path,
            git_commit=git_commit,
            python_version=sys.version,
            seed=seed,
            tag=tag,
            params_hash=ctx.params_hash,
        )
        write_json(ctx.run_dir / "run_meta.json", meta.model_dump(mode="json"))
        write_text(ctx.run_dir / "params_hash.txt", f"{ctx.params_hash}\n")

    @staticmethod
    def enforce_trade_schema(trades: pd.DataFrame) -> pd.DataFrame:
        df = trades.copy()
        for col in MIN_TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[MIN_TRADE_COLUMNS]

    @staticmethod
    def enforce_equity_schema(equity: pd.DataFrame) -> pd.DataFrame:
        df = equity.copy()
        for col in MIN_EQUITY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[MIN_EQUITY_COLUMNS]

    def persist_outputs(
        self,
        ctx: RunContext,
        *,
        summary: dict[str, Any],
        trades: pd.DataFrame,
        equity: pd.DataFrame,
        regime_stats: dict[str, Any],
        direction_stats: dict[str, Any],
    ) -> None:
        summary_model = BacktestSummary.model_validate(summary)
        write_json(ctx.run_dir / "summary.json", summary_model.model_dump(mode="json"))
        write_json(ctx.run_dir / "regime_stats.json", regime_stats)
        write_json(ctx.run_dir / "direction_stats.json", direction_stats)
        write_csv(ctx.run_dir / "trades.csv", self.enforce_trade_schema(trades))
        write_csv(ctx.run_dir / "equity.csv", self.enforce_equity_schema(equity))

        metrics_md = ["# Backtest Metrics", ""]
        for key, value in summary_model.model_dump(mode="json").items():
            if key == "counts":
                continue
            metrics_md.append(f"- **{key}**: {value}")
        metrics_md.append("- **counts**:")
        for c_key, c_value in summary_model.counts.model_dump(mode="json").items():
            metrics_md.append(f"  - {c_key}: {c_value}")
        write_text(ctx.run_dir / "metrics.md", "\n".join(metrics_md) + "\n")
