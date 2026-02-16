from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _prepare_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    _prepare_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    _prepare_parent(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    _prepare_parent(path)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.8f")


def write_text(path: Path, content: str) -> None:
    _prepare_parent(path)
    path.write_text(content, encoding="utf-8")
