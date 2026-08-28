"""Shared mart_dev column inventory for entity-dictionary Excel builders.

Parses column names from mart_dev SQL and applies curated descriptions from
mart_dictionary_data (COLUMN_OVERRIDES + ACF_COLUMN_DESC).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mart_dictionary_data import ACF_COLUMN_DESC, COLUMN_OVERRIDES, ENTITIES

ROOT = Path(__file__).resolve().parents[1]
MART_ROOT = ROOT / "dbt" / "models" / "mart_dev"

SKIP_SQL = {"gold_example.sql"}
LAYER_MAP = {"core": "dim", "facts": "fact", "aggregates": "agg", "bridges": "bridge"}
NOISE_COLS = {
    "string",
    "int64",
    "float64",
    "bool",
    "boolean",
    "date",
    "timestamp",
    "numeric",
    "bytes",
    "array",
    "struct",
    "x",
    "d",
}

COLUMN_HEADERS = [
    "table_name",
    "column_name",
    "data_type",
    "role",
    "description",
    "example",
]


def cols_from_sql(text: str) -> list[str]:
    text_nc = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text_nc = re.sub(r"--.*?$", "", text_nc, flags=re.M)
    text_nc = re.sub(r"\{#.*?#\}", "", text_nc, flags=re.S)
    found: list[str] = []
    for m in re.finditer(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", text_nc, re.I):
        name = m.group(1)
        if name.lower() in NOISE_COLS:
            continue
        found.append(name)
    return list(dict.fromkeys(found))


def infer_role(col: str, pk: str) -> str:
    if col == pk:
        return "PK"
    if col.endswith("_key"):
        return "FK"
    if col in ACF_COLUMN_DESC:
        return ACF_COLUMN_DESC[col]["role"]
    if col in {"value", "unit", "area_harvested", "production_qty", "yield_value", "price_avg"}:
        return "measure"
    if col in {"loaded_at"}:
        return "meta"
    return "dim"


def scan_sql_models(mart_root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = mart_root or MART_ROOT
    sql_models: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.sql")):
        if path.name in SKIP_SQL:
            continue
        layer = LAYER_MAP.get(path.parent.name, path.parent.name)
        cols = cols_from_sql(path.read_text(encoding="utf-8", errors="replace"))
        sql_models[path.stem] = {"layer": layer, "path": str(path), "columns": cols}
    return sql_models


def collect_column_rows(
    entities: list[dict[str, Any]] | None = None,
    overrides: dict[str, dict[str, dict[str, str]]] | None = None,
    mart_root: Path | None = None,
) -> tuple[list[str], list[list[Any]], dict[str, dict[str, Any]]]:
    """Return (headers, rows, sql_models). Rows match both Excel builders."""
    entity_list = entities if entities is not None else ENTITIES
    entities_by_name = {e["table_name"]: e for e in entity_list}
    over_map = overrides if overrides is not None else COLUMN_OVERRIDES
    sql_models = scan_sql_models(mart_root)

    col_rows: list[list[Any]] = []
    for name, info in sorted(sql_models.items()):
        meta = entities_by_name.get(name, {})
        pk = str(meta.get("primary_key") or "")
        table_over = over_map.get(name, {})
        # SQL `as` aliases + curated overrides (covers bare selects like p.production_grain)
        cols = list(info["columns"])
        for override_col in table_over:
            if override_col not in cols:
                cols.append(override_col)
        for col in cols:
            col_s = str(col)
            over = table_over.get(col_s, {})
            acf = ACF_COLUMN_DESC.get(col_s, {})
            role = over.get("role") or acf.get("role") or infer_role(col_s, pk)
            desc = over.get("description") or acf.get("description") or ""
            example = over.get("example") or acf.get("example") or ""
            dtype = over.get("data_type") or ""
            col_rows.append([name, col_s, dtype, role, desc, example])

    return COLUMN_HEADERS, col_rows, sql_models
