"""Shared engine helpers: prompt pack, bind contract, planned results."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bq_engine_validate import validate_engine_sql
from ml.rag.chatbot.bq_mart_sql import mart_table_fqn
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_intent_for_table,
    compile_table_bind_contract,
    pack_mart_table_hints,
)
from ml.rag.chatbot.class_engines.base import EngineResult
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


def _measure_id_for_class(class_code: str, facets: dict[str, Any]) -> str:
    pm = facets.get("primary_measures")
    if isinstance(pm, list) and pm:
        return str(pm[0]).strip().lower() or "value"
    code = (class_code or "").upper()
    if code == "PROD":
        return "production"
    if code == "PRC":
        return "market_price"
    if code == "FS":
        return "food_security"
    if code == "FVC":
        return "food_availability"
    return "value"


def build_planned_engine_result(
    *,
    class_code: str,
    table_id: str,
    query: str,
    facets: dict[str, Any],
    card: dict[str, Any],
    value_hits: dict[str, Any] | None = None,
    iso_list: list[str] | None = None,
    measure_id: str | None = None,
) -> EngineResult:
    """Return a planned engine result: bind contract + intents, no SELECT string."""
    hits = dict(value_hits or {})
    geos = list(iso_list or hits.get("country_iso3") or [])
    if geos:
        hits["country_iso3"] = geos

    contract = compile_table_bind_contract(
        table_id,
        facets=facets,
        card=card,
        query=query,
        country_labels=geos,
    )

    ts = str(facets.get("time_start") or "")[:10]
    te = str(facets.get("time_end") or "")[:10]
    africa_panel = bool(facets.get("africa_panel"))
    multi = africa_panel or len(geos) >= 2
    mid = measure_id or _measure_id_for_class(class_code, facets)
    year_hint = (te or ts or "")[:4] or "year from question"

    intent = compile_intent_for_table(
        table_id,
        measure_id=mid,
        query=query,
        geo_labels=geos,
        year_hint=year_hint,
        multi_country=multi,
        africa_panel=africa_panel,
        time_start=ts,
        time_end=te,
    )
    intent["goal"] = f"{class_code} from {table_id}"
    intent["notes"] = class_code
    if contract.required_filters_sql.strip():
        intent["filters"] = f"{intent.get('filters', '')}; bind={contract.nomenclature[:200]}"

    terms = [query[:80], *(geos[:3])]
    entities = facets.get("entities")
    if isinstance(entities, list):
        terms.extend(str(e) for e in entities[:3] if str(e).strip())
    hints, hints_truncated = pack_mart_table_hints([table_id], query_terms=terms)

    return EngineResult(
        class_code=class_code,
        status="planned",
        table_id=table_id,
        sql=None,
        value_hits=hits,
        bind_contract=contract.to_dict(),
        query_intents=[intent],
        table_hints=hints,
        hints_truncated=hints_truncated,
    )


__all__ = [
    "mart_table_fqn",
    "pack_engine_prompt",
    "bind_value_hits",
    "build_planned_engine_result",
    "validate_engine_sql",
]
