"""Thin 1–2 intent BQ plans for fact_lookup and data_export_only modes."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import (
    fallback_plan,
    resolve_measure,
    wants_africa_panel,
)
from ml.rag.chatbot.bq_table_schema_yaml import compile_intent_for_table, pack_mart_table_hints


def build_fact_bq_plan(
    query: str,
    *,
    decomposition: dict[str, Any],
    known_tables: set[str],
    task_mode: str = "fact_lookup",
) -> dict[str, Any] | None:
    """Ontology-aware forced plan (skip_bq=false) with one or two focused intents."""
    hit = resolve_measure(query, decomposition)
    if hit is not None:
        ontology_plan = fallback_plan(
            hit,
            query=query,
            decomposition=decomposition,
            known_tables=known_tables,
            task_mode=task_mode,
        )
        if ontology_plan is not None:
            ontology_plan["rationale"] = f"fact_forced_{task_mode}_{hit.measure.id}"
            return ontology_plan

    # Legacy production fallback when ontology misses but production table exists.
    if "fct_production" not in known_tables:
        return None

    geo_raw = decomposition.get("geography")
    geo = geo_raw if isinstance(geo_raw, list) else []
    countries = [str(g).strip() for g in geo if str(g).strip()]
    africa_panel = bool(decomposition.get("africa_panel")) or wants_africa_panel(query)
    ts = str(decomposition.get("time_start") or "")[:10]
    te = str(decomposition.get("time_end") or "")[:10]
    year_hint = (te or ts or "")[:4] or "year from question"

    entities_raw = decomposition.get("entities")
    entities = entities_raw if isinstance(entities_raw, list) else []
    item_hint = ", ".join(str(e).strip() for e in entities[:4] if str(e).strip()) or "crop from question"
    want_yield = bool(re.search(r"\byields?\b", query or "", re.IGNORECASE))
    table = "fct_yield" if want_yield else "fct_production"
    multi = africa_panel or bool(decomposition.get("africa_default")) or len(countries) != 1

    intent = compile_intent_for_table(
        table,
        measure_id="production" if not want_yield else "yield",
        query=query,
        geo_labels=countries,
        year_hint=year_hint,
        multi_country=multi,
        africa_panel=africa_panel,
        time_start=ts,
        time_end=te,
    )
    intent["goal"] = (
        f"Primary {'yield' if want_yield else 'production'} fact/rank for the asked geography/commodity"
    )
    intent["notes"] = f"fact_{task_mode}"
    intent["filters"] = f"{intent['filters']}; product≈{item_hint}"
    intents: list[dict[str, Any]] = [intent]

    if task_mode == "data_export_only":
        export = compile_intent_for_table(
            table,
            measure_id="production" if not want_yield else "yield",
            query=query,
            geo_labels=countries,
            year_hint=year_hint,
            multi_country=multi,
            africa_panel=africa_panel,
            time_start=ts,
            time_end=te,
        )
        export["goal"] = "Tabular series or multi-row export for the same filters"
        export["notes"] = "fact_export_table"
        export["pattern"] = "time_series"
        export["order_by"] = "year ASC"
        intents.append(export)

    selected = [table]
    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": f"fact_forced_{task_mode}",
        "task_mode": task_mode,
        "max_sql_queries": 3,
    }
    terms = [query[:80], item_hint, *countries[:5]]
    hints, hints_truncated = pack_mart_table_hints(selected, query_terms=terms)
    plan["table_hints"] = hints
    plan["index_truncated"] = False
    plan["hints_truncated"] = hints_truncated
    return plan


__all__ = ["build_fact_bq_plan"]
