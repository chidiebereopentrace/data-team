"""Shared engine helpers: prompt pack, SQL validate hooks."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_sql_validate import (
    validate_sql_column_allowlist,
    validate_sql_table_allowlist,
)
from ml.rag.chatbot.schema_card import load_schema_card, prompt_mode_for_column
from ml.rag.chatbot.value_index import (
    complete_enum,
    numeric_stats,
    resolve_country,
    resolve_labels,
)

_PROJECT = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")
_DATASET = os.environ.get("BQ_DATASET", "mart_dev")

_EXTRACT_YEAR_WHERE_RE = re.compile(
    r"extract\s*\(\s*year\s+from\s+as_of_date\s*\)",
    re.I,
)


def mart_table_fqn(table_id: str) -> str:
    bare = table_id.split(".")[-1]
    return f"`{_PROJECT}.{_DATASET}.{bare}`"


def pack_engine_prompt(
    card: dict[str, Any],
    *,
    query: str,
    facets: dict[str, Any],
    value_hits: dict[str, Any],
) -> str:
    lines = [f"Class: {card.get('class')}", f"Default table: {card.get('default_table')}"]
    for rule in card.get("hard_rules") or []:
        lines.append(f"Rule: {rule}")
    table = str(card.get("default_table") or "")
    cols = card.get("columns") or {}
    if isinstance(cols, dict):
        for col, spec in cols.items():
            if not isinstance(spec, dict):
                continue
            mode = str(spec.get("prompt_mode") or "")
            if mode == "full_list":
                enums = value_hits.get(col) or complete_enum(table, col)
                lines.append(f"{col} (complete): {' | '.join(enums[:80])}")
            elif mode == "resolved_only" and value_hits.get(col):
                lines.append(f"Resolved {col}: {' | '.join(value_hits[col])}")
                lines.append(f"Do not invent other {col} values.")
            elif mode == "stats_only":
                st = numeric_stats(table, col)
                lines.append(f"{col} stats: {st}")
    lines.append(f"Question: {query[:500]}")
    return "\n".join(lines)


def bind_value_hits(
    card: dict[str, Any],
    *,
    query: str,
    facets: dict[str, Any],
) -> dict[str, Any]:
    table = str(card.get("default_table") or "")
    hits: dict[str, Any] = {}
    geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
    iso = resolve_country(query, geography=geography)
    if iso:
        hits["country_iso3"] = [iso]
    cols = card.get("columns") or {}
    if not isinstance(cols, dict):
        cols = {}
    for col, spec in cols.items():
        if not isinstance(spec, dict):
            continue
        mode = str(spec.get("prompt_mode") or prompt_mode_for_column(card, col))
        if col == "country_iso3":
            if iso:
                hits[col] = [iso]
            continue
        if mode == "full_list":
            hits[col] = complete_enum(table, col)
        elif mode == "resolved_only":
            scope = str(spec.get("scope") or "table")
            hits[col] = resolve_labels(
                table,
                col,
                query,
                scope=scope,
                geography=geography,
            )
    return hits


def validate_engine_sql(
    sql: str,
    *,
    table_id: str,
    selected_tables: list[str] | None = None,
) -> tuple[bool, str]:
    if _EXTRACT_YEAR_WHERE_RE.search(sql or ""):
        return False, "EXTRACT(YEAR FROM as_of_date) forbidden in WHERE"
    if re.search(r"\bgeo_key\s*=", sql or "", re.I):
        return False, "geo_key hash filter forbidden"
    if not re.search(r"\blimit\s+\d+", sql or "", re.I):
        return False, "LIMIT required"
    tables = set(selected_tables or [table_id])
    err = validate_sql_table_allowlist(sql, tables)
    if err:
        return False, err
    err = validate_sql_column_allowlist(sql, tables)
    if err:
        return False, err or ""
    return True, ""


__all__ = [
    "mart_table_fqn",
    "pack_engine_prompt",
    "bind_value_hits",
    "validate_engine_sql",
]
