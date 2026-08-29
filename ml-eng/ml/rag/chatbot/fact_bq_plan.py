"""Thin 1–2 intent BQ plans for fact_lookup and data_export_only modes."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import (
    fallback_plan,
    resolve_measure,
    wants_africa_panel,
)
from ml.rag.chatbot.bq_table_schema_yaml import pack_mart_table_hints


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
    geo_filter = (
        "Africa continental panel GROUP BY country_iso3"
        if africa_panel or (not countries and decomposition.get("africa_default"))
        else (
            f"country_iso3 in ({', '.join(countries[:8])})"
            if countries
            else "Africa continental ranking when unscoped"
        )
    )
    ts = str(decomposition.get("time_start") or "")[:10]
    te = str(decomposition.get("time_end") or "")[:10]
    year_hint = (te or ts or "")[:4] or "year from question"

    entities_raw = decomposition.get("entities")
    entities = entities_raw if isinstance(entities_raw, list) else []
    item_hint = ", ".join(str(e).strip() for e in entities[:4] if str(e).strip()) or "crop from question"
    want_yield = bool(re.search(r"\byields?\b", query or "", re.IGNORECASE))
    table = "fct_yield" if want_yield else "fct_production"
    grain_filter = "season_key/harvest_year" if want_yield else "production_grain='physical'"

    intents: list[dict[str, Any]] = [
        {
            "goal": f"Primary {'yield' if want_yield else 'production'} fact/rank for the asked geography/commodity",
            "tables": [table],
            "filters": f"{grain_filter}; {geo_filter}; year≈{year_hint}; product≈{item_hint}",
            "notes": f"fact_{task_mode}",
            "pattern": "rank_by_sum" if not countries or len(countries) > 1 or africa_panel else "custom",
            "metric": "value",
            "grain": ["country_iso3"]
            if not countries or len(countries) != 1 or africa_panel
            else ["country_iso3", "product_key"],
            "order_by": "value DESC",
        }
    ]
    if task_mode == "data_export_only":
        intents.append(
            {
                "goal": "Tabular series or multi-row export for the same filters",
                "tables": [table],
                "filters": f"{grain_filter}; {geo_filter}; year around {year_hint}",
                "notes": "fact_export_table",
                "pattern": "time_series",
                "metric": "value",
                "grain": ["country_iso3", "year"],
                "order_by": "year ASC",
            }
        )

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
