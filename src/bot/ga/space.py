from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bot.runs.run_manager import RunManager

SEARCH_ROOTS = {
    "strategy_breakout",
    "regime",
    "funding_filter",
    "fng_filter",
    "market_structure",
    "frictions",
    "strategy_router",
    "router",
    "multi_timeframe",
    "time_exit",
    "adaptive_trailing",
    "risk",
}

EXCLUDED_SUFFIXES = {
    "symbol",
    "interval",
    "start_date",
    "end_date",
    "path",
    "api_key_env",
    "api_secret_env",
}

CHOICE_KEYS = {
    "strategy_breakout.mode": ["breakout", "baseline", "ema", "ema_macd", "ml_gate"],
    "router.trend_mode": ["breakout", "baseline", "ema", "ema_macd", "ml_gate"],
    "router.range_mode": ["breakout", "baseline", "ema", "ema_macd", "ml_gate"],
    "strategy_breakout.trade_direction": ["both", "long", "short"],
    "strategy_breakout.ml_feature_selector": ["lightgbm", "xgboost"],
    "strategy_breakout.retest.confirmation": ["close_back", "wick_reject"],
    "market_structure.gate.mode": ["msb_only", "structure_trend", "hybrid"],
}


@dataclass(frozen=True)
class ParamSpec:
    key: str
    kind: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    step: float | int | None = None
    sigma: float | None = None

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "bool":
            return bool(rng.randint(0, 1))
        if self.kind == "choice":
            return rng.choice(list(self.choices))
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return int(rng.randint(int(self.low), int(self.high)))
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return float(rng.uniform(float(self.low), float(self.high)))
        raise ValueError(f"Unsupported spec kind: {self.kind}")

    def clamp(self, value: Any) -> Any:
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            if value in self.choices:
                return value
            return self.choices[0]
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            try:
                casted = int(round(float(value)))
            except Exception:
                casted = int(self.low)
            return max(int(self.low), min(int(self.high), casted))
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            try:
                casted = float(value)
            except Exception:
                casted = float(self.low)
            return max(float(self.low), min(float(self.high), casted))
        raise ValueError(f"Unsupported spec kind: {self.kind}")

    def mutate(self, value: Any, rng: random.Random) -> Any:
        if self.kind == "bool":
            return not bool(value)
        if self.kind == "choice":
            return rng.choice(list(self.choices))
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            span = int(self.high) - int(self.low)
            max_step = max(1, span // 12)
            step = int(self.step) if self.step is not None else max_step
            delta = rng.randint(-max(1, step), max(1, step))
            return self.clamp(int(value) + delta)
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            span = float(self.high) - float(self.low)
            sigma = self.sigma if self.sigma is not None else max(1e-6, span / 12.0)
            delta = rng.gauss(0.0, sigma)
            return self.clamp(float(value) + delta)
        raise ValueError(f"Unsupported spec kind: {self.kind}")


class SearchSpace:
    def __init__(self, base_config: dict[str, Any], specs: dict[str, ParamSpec]) -> None:
        self.base_config = copy.deepcopy(base_config)
        self.specs = specs

    def sample_individual(self, rng: random.Random) -> dict[str, Any]:
        sampled = {key: spec.sample(rng) for key, spec in self.specs.items()}
        return self.normalize_genes(sampled)

    def base_genes(self) -> dict[str, Any]:
        flat = _flatten(self.base_config)
        genes: dict[str, Any] = {}
        for key, spec in self.specs.items():
            if key in flat:
                genes[key] = spec.clamp(flat[key])
        return self.normalize_genes(genes)

    def normalize_genes(self, genes: dict[str, Any]) -> dict[str, Any]:
        normalized = {key: self.specs[key].clamp(value) for key, value in genes.items() if key in self.specs}
        return self._apply_constraints(normalized)

    def mutate_gene(self, key: str, value: Any, rng: random.Random) -> Any:
        return self.specs[key].mutate(value, rng)

    def apply_genes(self, genes: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(self.base_config)
        normalized = self.normalize_genes(genes)
        for key, value in normalized.items():
            if key not in self.specs:
                continue
            _set_nested(cfg, key, value)
        return cfg

    def config_hash(self, genes: dict[str, Any]) -> str:
        cfg = self.apply_genes(genes)
        return RunManager.stable_hash_from_dict(cfg)

    def _apply_constraints(self, genes: dict[str, Any]) -> dict[str, Any]:
        repaired = dict(genes)

        adx_exit_key = "regime.adx_exit_threshold"
        adx_enter_key = "regime.adx_enter_threshold"
        adx_trend_key = "regime.adx_trend_threshold"
        if adx_exit_key in repaired and adx_enter_key in repaired and adx_trend_key in repaired:
            exit_v = float(repaired[adx_exit_key])
            enter_v = float(repaired[adx_enter_key])
            trend_v = float(repaired[adx_trend_key])
            exit_v = min(exit_v, enter_v)
            trend_v = max(trend_v, enter_v)
            repaired[adx_exit_key] = self.specs[adx_exit_key].clamp(exit_v)
            repaired[adx_enter_key] = self.specs[adx_enter_key].clamp(enter_v)
            repaired[adx_trend_key] = self.specs[adx_trend_key].clamp(trend_v)
            repaired[adx_exit_key] = self.specs[adx_exit_key].clamp(
                min(float(repaired[adx_exit_key]), float(repaired[adx_enter_key]))
            )
            repaired[adx_trend_key] = self.specs[adx_trend_key].clamp(
                max(float(repaired[adx_trend_key]), float(repaired[adx_enter_key]))
            )

        z_key = "funding_filter.z_threshold"
        long_key = "funding_filter.block_long_if_z_gt"
        short_key = "funding_filter.block_short_if_z_lt"
        if z_key in repaired and long_key in repaired:
            z = abs(float(repaired[z_key]))
            repaired[z_key] = self.specs[z_key].clamp(z)
            repaired[long_key] = self.specs[long_key].clamp(max(float(repaired[long_key]), float(repaired[z_key])))
        if z_key in repaired and short_key in repaired:
            z = abs(float(repaired[z_key]))
            repaired[z_key] = self.specs[z_key].clamp(z)
            target_short = -max(abs(float(repaired[short_key])), float(repaired[z_key]))
            repaired[short_key] = self.specs[short_key].clamp(target_short)

        return repaired


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(value, child))
        return out
    out[prefix] = node
    return out


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: dict[str, Any] = payload
    keys = dotted_key.split(".")
    for token in keys[:-1]:
        current = cursor.get(token)
        if not isinstance(current, dict):
            current = {}
            cursor[token] = current
        cursor = current
    cursor[keys[-1]] = value


def _auto_numeric_spec(key: str, value: Any) -> ParamSpec | None:
    key_lower = key.lower()
    if isinstance(value, bool):
        return ParamSpec(key=key, kind="bool")

    if isinstance(value, int) and not isinstance(value, bool):
        low, high = _derive_bounds_int(key_lower, value)
        return ParamSpec(key=key, kind="int", low=low, high=high)

    if isinstance(value, float):
        low, high = _derive_bounds_float(key_lower, value)
        sigma = max(1e-6, (high - low) / 10.0)
        return ParamSpec(key=key, kind="float", low=low, high=high, sigma=sigma)

    choices = CHOICE_KEYS.get(key)
    if isinstance(value, str) and choices:
        return ParamSpec(key=key, kind="choice", choices=tuple(choices))
    return None


def _derive_bounds_int(key: str, value: int) -> tuple[int, int]:
    if key.endswith("breakout_lookback_n"):
        return 10, max(200, value * 3)
    if "adx" in key and "threshold" in key:
        return 10, 80
    if key.endswith("ma200_period") or key.endswith("multi_timeframe.ma_period"):
        return 50, max(400, value * 2)
    if key.endswith("left_bars") or key.endswith("right_bars"):
        return 1, max(12, value * 2)
    if any(token in key for token in ["period", "window", "bars", "hours", "count"]):
        lo = 1 if value > 0 else 0
        hi = max(value * 3 if value > 0 else 12, lo + 1)
        return int(lo), int(hi)
    lo = 0 if value >= 0 else int(math.floor(value * 1.5))
    hi = max(1, int(math.ceil(value * 2.0))) if value >= 0 else int(math.ceil(abs(value) * 1.5))
    return lo, hi


def _derive_bounds_float(key: str, value: float) -> tuple[float, float]:
    if key.endswith("atr_k"):
        return 0.1, max(8.0, value * 2.5)
    if key.endswith("min_break_atr"):
        return 0.0, max(3.0, value * 3.0 + 0.2)
    if 0.0 <= value <= 1.0:
        return 0.0, 1.0
    if "adx" in key and "threshold" in key:
        return 10.0, 80.0
    lo = 0.0 if value >= 0 else value * 2.0
    hi = max(1.0, value * 2.0 if value > 0 else abs(value) * 2.0)
    return float(lo), float(hi)


def _is_allowed_key(key: str) -> bool:
    root = key.split(".", 1)[0]
    if root not in SEARCH_ROOTS:
        return False
    if key.split(".")[-1] in EXCLUDED_SUFFIXES:
        return False
    return True


def _spec_from_payload(key: str, payload: dict[str, Any]) -> ParamSpec:
    kind = str(payload.get("type", "")).strip().lower()
    if kind not in {"int", "float", "bool", "choice"}:
        raise ValueError(f"Invalid type for {key}: {kind}")

    if kind == "bool":
        return ParamSpec(key=key, kind="bool")

    if kind == "choice":
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"choices missing for {key}")
        return ParamSpec(key=key, kind="choice", choices=tuple(choices))

    low = payload.get("min")
    high = payload.get("max")
    if low is None or high is None:
        raise ValueError(f"min/max missing for {key}")

    if kind == "int":
        step = payload.get("step")
        return ParamSpec(key=key, kind="int", low=int(low), high=int(high), step=int(step) if step is not None else None)

    sigma = payload.get("sigma")
    return ParamSpec(
        key=key,
        kind="float",
        low=float(low),
        high=float(high),
        sigma=float(sigma) if sigma is not None else None,
    )


def _load_space_yaml(path: Path) -> dict[str, ParamSpec]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    params = parsed.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("config/ga_space.yaml precisa de chave 'parameters' com um mapa.")

    specs: dict[str, ParamSpec] = {}
    for key, payload in params.items():
        if not isinstance(payload, dict):
            continue
        specs[key] = _spec_from_payload(key, payload)
    return specs


def discover_search_space(base_config: dict[str, Any], ga_space_path: str | Path | None = None) -> SearchSpace:
    flat = _flatten(base_config)
    specs: dict[str, ParamSpec] = {}

    if ga_space_path is not None:
        declared = _load_space_yaml(Path(ga_space_path))
        if declared:
            specs = declared

    if not specs:
        for key, value in flat.items():
            if not _is_allowed_key(key):
                continue
            spec = _auto_numeric_spec(key, value)
            if spec is not None:
                specs[key] = spec

    if not specs:
        raise ValueError("Nenhum parâmetro elegível foi encontrado para o GA.")

    # Normalize base values to satisfy declared bounds.
    for key, spec in list(specs.items()):
        value = flat.get(key)
        if value is None and spec.kind != "bool":
            continue
        try:
            spec.clamp(value)
        except Exception:
            del specs[key]

    return SearchSpace(base_config=base_config, specs=dict(sorted(specs.items())))


def genes_hash(genes: dict[str, Any]) -> str:
    normalized = json.dumps(genes, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return RunManager.stable_hash_from_dict({"genes": normalized})
