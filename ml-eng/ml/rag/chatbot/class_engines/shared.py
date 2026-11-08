"""Shared engine helpers: prompt pack, bind contract, planned results."""
from __future__ import annotations

from typing import Any, Sequence

from ml.rag.chatbot.bq_engine_validate import validate_engine_sql
from ml.rag.chatbot.bq_mart_sql import mart_table_fqn
from ml.rag.chatbot.bind_planning import compile_bind_for_request, effective_time_bounds
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_intent_for_table,
    pack_mart_table_hints,
)
from ml.rag.chatbot.class_engines.base import EngineResult
from ml.rag.chatbot.class_table_router import TablePlan
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

    product_hits = hits.get("product_name")
    product_labels = (
        [str(x).strip() for x in product_hits if str(x).strip()]
        if isinstance(product_hits, list)
        else None
    )
    contract = compile_bind_for_request(
        table_id,
        facets=facets,
        card=card,
        query=query,
        country_labels=geos,
        product_labels=product_labels,
    )

    ts, te = effective_time_bounds(contract, facets)
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

    contract_dict = contract.to_dict()
    return EngineResult(
        class_code=class_code,
        status="planned",
        table_id=table_id,
        sql=None,
        value_hits=hits,
        bind_contract=contract_dict,
        bind_contracts={table_id: contract_dict},
        query_intents=[intent],
        table_hints=hints,
        hints_truncated=hints_truncated,
    )


def _measure_id_for_table(table_id: str, class_code: str, facets: dict[str, Any]) -> str:
    bare = str(table_id or "").strip().split(".")[-1].lower()
    if "trade" in bare:
        return "trade"
    if "food_balance" in bare or "food_security" in bare:
        return "food_availability"
    if "price" in bare:
        return "market_price"
    if "production" in bare or "yield" in bare:
        return "production"
    return _measure_id_for_class(class_code, facets)


def build_planned_multi_table_result(
    *,
    class_code: str,
    plans: Sequence[TablePlan | str],
    query: str,
    facets: dict[str, Any],
    card: dict[str, Any],
    value_hits: dict[str, Any] | None = None,
    iso_list: list[str] | None = None,
    measure_id: str | None = None,
    measure_by_table: dict[str, str] | None = None,
) -> EngineResult:
    """Build one bind + intent per TablePlan; merge into bind_contracts (sql=None)."""
    table_ids: list[str] = []
    for p in plans:
        if isinstance(p, str):
            tid = p.strip().split(".")[-1]
        else:
            tid = str(getattr(p, "table_id", "") or "").strip().split(".")[-1]
        if tid and tid not in table_ids:
            table_ids.append(tid)
    if not table_ids:
        return EngineResult(
            class_code=class_code,
            status="planner_error",
            table_id="",
            sql=None,
            caveats=["no_table_plans"],
            value_hits=dict(value_hits or {}),
        )
    if len(table_ids) == 1:
        mid = None
        if measure_by_table and table_ids[0] in measure_by_table:
            mid = measure_by_table[table_ids[0]]
        else:
            mid = measure_id
        return build_planned_engine_result(
            class_code=class_code,
            table_id=table_ids[0],
            query=query,
            facets=facets,
            card=card,
            value_hits=value_hits,
            iso_list=iso_list,
            measure_id=mid,
        )

    hits = dict(value_hits or {})
    geos = list(iso_list or hits.get("country_iso3") or [])
    if geos:
        hits["country_iso3"] = geos
    product_hits = hits.get("product_name")
    product_labels = (
        [str(x).strip() for x in product_hits if str(x).strip()]
        if isinstance(product_hits, list)
        else None
    )
    africa_panel = bool(facets.get("africa_panel"))
    multi = africa_panel or len(geos) >= 2

    bind_contracts: dict[str, Any] = {}
    intents: list[dict[str, Any]] = []
    for tid in table_ids:
        mid = (measure_by_table or {}).get(tid) or measure_id or _measure_id_for_table(
            tid, class_code, facets
        )
        contract = compile_bind_for_request(
            tid,
            facets=facets,
            card=card,
            query=query,
            country_labels=geos,
            product_labels=product_labels,
        )
        bind_contracts[tid] = contract.to_dict()
        ts, te = effective_time_bounds(contract, facets)
        year_hint = (te or ts or "")[:4] or "year from question"
        intent = compile_intent_for_table(
            tid,
            measure_id=mid,
            query=query,
            geo_labels=geos,
            year_hint=year_hint,
            multi_country=multi,
            africa_panel=africa_panel,
            time_start=ts,
            time_end=te,
        )
        intent["goal"] = f"{class_code} from {tid}"
        intent["notes"] = class_code
        if contract.required_filters_sql.strip():
            intent["filters"] = f"{intent.get('filters', '')}; bind={contract.nomenclature[:200]}"
        intents.append(intent)

    terms = [query[:80], *(geos[:3])]
    entities = facets.get("entities")
    if isinstance(entities, list):
        terms.extend(str(e) for e in entities[:3] if str(e).strip())
    hints, hints_truncated = pack_mart_table_hints(table_ids, query_terms=terms)
    primary = table_ids[0]
    return EngineResult(
        class_code=class_code,
        status="planned",
        table_id=primary,
        sql=None,
        value_hits=hits,
        bind_contract=bind_contracts.get(primary),
        bind_contracts=bind_contracts,
        query_intents=intents,
        table_hints=hints,
        hints_truncated=hints_truncated,
    )


__all__ = [
    "mart_table_fqn",
    "pack_engine_prompt",
    "bind_value_hits",
    "build_planned_engine_result",
    "build_planned_multi_table_result",
    "validate_engine_sql",
]
