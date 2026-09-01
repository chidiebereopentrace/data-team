"""Load per-class schema cards for class engines."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "schema_cards"

_TINY_ENUM_THRESHOLD = 40


@lru_cache(maxsize=32)
def load_schema_card(class_code: str) -> dict[str, Any] | None:
    code = (class_code or "").strip().upper()
    if not code:
        return None
    path = _DEFAULT_DIR / f"{code}.yaml"
    if not path.is_file():
        return _default_card(code)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else _default_card(code)


def _default_card(class_code: str) -> dict[str, Any]:
    from ml.rag.chatbot.mart_indicator_classes import facts_for_class

    tables = facts_for_class(class_code)
    default_table = tables[0] if tables else ""
    return {
        "class": class_code,
        "default_table": default_table,
        "tables": tables,
        "columns": {},
        "hard_rules": [],
        "exclude_prompt_columns": ["geo_key", "scenario_key", "classification_key", "product_key"],
    }


def prompt_mode_for_column(card: dict[str, Any], column: str, *, distinct_count: int | None = None) -> str:
    cols = card.get("columns") or {}
    spec = cols.get(column) if isinstance(cols, dict) else None
    if isinstance(spec, dict) and spec.get("prompt_mode"):
        return str(spec["prompt_mode"])
    if distinct_count is not None and distinct_count <= _TINY_ENUM_THRESHOLD:
        return "full_list"
    role = str((spec or {}).get("role") or "")
    if role in ("measure", "time_filter"):
        return "stats_only"
    if role == "filter_enum":
        return "resolved_only"
    return "resolved_only"


def card_maturity(card: dict[str, Any] | None) -> dict[str, Any]:
    """Inspector-facing maturity for schema cards (stub vs ready)."""
    if not card:
        return {"status": "missing", "column_count": 0, "reason": "no card loaded"}
    cols = card.get("columns") or {}
    count = len(cols) if isinstance(cols, dict) else 0
    if count == 0:
        return {"status": "stub", "column_count": 0, "reason": "columns empty"}
    if count < 3:
        return {"status": "partial", "column_count": count}
    return {"status": "ready", "column_count": count}


__all__ = ["load_schema_card", "prompt_mode_for_column", "card_maturity", "_TINY_ENUM_THRESHOLD"]
