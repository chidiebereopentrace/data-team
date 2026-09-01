"""Shared engine helpers: prompt pack, SQL validate hooks."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bq_engine_validate import validate_engine_sql
from ml.rag.chatbot.bq_mart_sql import mart_table_fqn
from ml.rag.chatbot.schema_card import prompt_mode_for_column
from ml.rag.chatbot.value_index import (
    complete_enum,
    numeric_stats,
    resolve_geography_iso3,
    resolve_labels,
)


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
    expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
    iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
    if iso_list:
        hits["country_iso3"] = iso_list
    cols = card.get("columns") or {}
    if not isinstance(cols, dict):
        cols = {}
    for col, spec in cols.items():
        if not isinstance(spec, dict):
            continue
        mode = str(spec.get("prompt_mode") or prompt_mode_for_column(card, col))
        if col == "country_iso3":
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


__all__ = [
    "mart_table_fqn",
    "pack_engine_prompt",
    "bind_value_hits",
    "validate_engine_sql",
]
