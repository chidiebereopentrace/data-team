"""Forced multi-intent BigQuery plans for analytical report mode."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import fallback_plan, resolve_measure
from ml.rag.chatbot.bq_table_schema_yaml import pack_selected_table_hints, table_supports_sql_pattern

_STAPLES = ("Maize", "Rice", "Cassava", "Sorghum", "Millet")

# Priority spine first, then cross-domain companions (not exclusive FEWS-only).
_FOOD_SECURITY_TABLES = (
    "stg_fews_food_security",
    "stg_fews_market_prices",
    "stg_faostat_production",
    "stg_ilri_household_food_security",
)


def analytical_sql_query_floor() -> int:
    try:
        env = int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10)
    except ValueError:
        env = 10
    try:
        floor = int(os.environ.get("RAG_ANALYTICAL_BQ_MIN_QUERIES", "8") or 8)
    except ValueError:
        floor = 8
    return max(env, floor, 8)


def build_food_security_bq_plan(
    query: str,
    *,
    decomposition: dict[str, Any],
    known_tables: set[str],
) -> dict[str, Any] | None:
    """FEWS/IPC priority spine plus prices, production, and ILRI companions.

    Does not pad ASTI or treat a single table as sufficient. Each companion is
    its own SQL intent (per-SQL validation applies per statement).
    """
    selected = [tid for tid in _FOOD_SECURITY_TABLES if tid in known_tables]
    if not selected:
        return None

    geo_raw = decomposition.get("geography")
    geo = geo_raw if isinstance(geo_raw, list) else []
    countries = [str(g).strip() for g in geo if str(g).strip()]
    geo_filter = (
        "countries=" + ",".join(countries[:16])
        if countries
        else "geography from question / expanded region"
    )
    ts = str(decomposition.get("time_start") or "")[:10] or "earliest"
    te = str(decomposition.get("time_end") or "")[:10] or "latest"
    y0 = ts[:4] if ts[:4].isdigit() else "start"
    y1 = te[:4] if te[:4].isdigit() else "end"
    multi = len(countries) != 1
    grain_country = ["country_name"] if multi else ["country_name", "year"]

    intents: list[dict[str, Any]] = []
    if "stg_fews_food_security" in selected:
        intents.append(
            {
                "goal": "FEWS / IPC food security phases by country in the scoped geography",
                "tables": ["stg_fews_food_security"],
                "filters": f"{geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_fews_food_security",
                "pattern": "custom",
                "metric": "value",
                "grain": grain_country,
                "order_by": "value DESC",
            }
        )
    if "stg_fews_market_prices" in selected:
        intents.append(
            {
                "goal": "Retail market prices companion for food-security assessment",
                "tables": ["stg_fews_market_prices"],
                "filters": f"price_type='Retail'; {geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_fews_market_prices",
                "pattern": "rank_by_sum" if multi else "custom",
                "metric": "value",
                "grain": ["country"],
                "order_by": "total DESC" if multi else "value DESC",
            }
        )
    if "stg_faostat_production" in selected:
        staples = ", ".join(_STAPLES[:4])
        intents.append(
            {
                "goal": (
                    f"Staple crop production pressure companion ({staples}) "
                    "by country — multi-country rank/IN, not single-country series"
                ),
                "tables": ["stg_faostat_production"],
                "filters": (
                    f"element='Production'; items in {list(_STAPLES[:4])}; "
                    f"{geo_filter}; year between {y0} and {y1}"
                ),
                "notes": "analytical_food_security_production_companion",
                "pattern": "rank_by_sum" if multi else "custom",
                "metric": "value",
                "grain": ["country_name", "product_name"] if multi else ["country_name", "year", "product_name"],
                "order_by": "total DESC" if multi else "value DESC",
            }
        )
    if "stg_ilri_household_food_security" in selected:
        intents.append(
            {
                "goal": "ILRI household food security companion signals",
                "tables": ["stg_ilri_household_food_security"],
                "filters": f"{geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_ilri_household_food_security",
                "pattern": "custom",
                "metric": "value",
                "grain": ["country_name"],
                "order_by": "value DESC",
            }
        )
    if not intents:
        return None

    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": "analytical_forced_food_security_ipc",
        "analytical_mode": True,
        "max_sql_queries": max(3, len(intents)),
        "measure_id": "food_security_ipc",
    }
    hints, hints_truncated = pack_selected_table_hints(
        selected,
        query_terms=_pack_terms(query, decomposition),
    )
    plan["table_hints"] = hints
    plan["index_truncated"] = False
    plan["hints_truncated"] = hints_truncated
    return plan


def build_analytical_bq_plan(
    query: str,
    *,
    decomposition: dict[str, Any],
    known_tables: set[str],
) -> dict[str, Any] | None:
    """
    Deterministic multi-intent plan for agricultural comparative / report queries.

    Prefer ontology composite (e.g. investor_best_country); food_security_ipc → FEWS;
    else production-centered multi-intent with trade/yield companions. Never sets skip_bq.
    """
    hit = resolve_measure(query, decomposition)
    if hit is not None and hit.measure.id == "food_security_ipc":
        fs = build_food_security_bq_plan(query, decomposition=decomposition, known_tables=known_tables)
        if fs is not None:
            return fs

    if hit is not None and (
        hit.measure.id == "investor_best_country"
        or hit.measure.default_task_mode == "analytical"
        and hit.measure.companions
    ):
        ontology_plan = fallback_plan(
            hit,
            query=query,
            decomposition=decomposition,
            known_tables=known_tables,
            task_mode="analytical",
        )
        if ontology_plan is not None and not ontology_plan.get("skip_bq"):
            floor = analytical_sql_query_floor()
            intents = list(ontology_plan.get("query_intents") or [])
            # Pad with companion table intents toward floor.
            for tid in list(ontology_plan.get("selected_tables") or []):
                if len(intents) >= floor:
                    break
                if any(tid in (i.get("tables") or []) for i in intents):
                    continue
                intents.append(
                    {
                        "goal": f"Companion analytical signal from {tid}",
                        "tables": [tid],
                        "filters": str((intents[0].get("filters") if intents else "") or ""),
                        "notes": f"analytical_companion_{tid}",
                        "pattern": "rank_by_sum" if table_supports_sql_pattern(tid) else "custom",
                        "metric": "value",
                        "grain": ["country_name"],
                        "order_by": "total DESC" if table_supports_sql_pattern(tid) else "value DESC",
                    }
                )
            ontology_plan["query_intents"] = intents[:floor]
            ontology_plan["analytical_mode"] = True
            ontology_plan["max_sql_queries"] = floor
            ontology_plan["rationale"] = f"analytical_forced_{hit.measure.id}"
            return ontology_plan

    if "stg_faostat_production" not in known_tables:
        return None

    geo_raw = decomposition.get("geography")
    geo = geo_raw if isinstance(geo_raw, list) else []
    countries = [str(g).strip() for g in geo if str(g).strip()]
    geo_filter = (
        "countries=" + ",".join(countries[:16])
        if countries
        else "Africa continental (expand region if present)"
    )
    ts = str(decomposition.get("time_start") or "")[:10] or "earliest"
    te = str(decomposition.get("time_end") or "")[:10] or "latest"
    y0 = ts[:4] if ts[:4].isdigit() else "start"
    y1 = te[:4] if te[:4].isdigit() else "end"
    want_yield = bool(re.search(r"\byields?\b", query or "", re.IGNORECASE))
    primary_element = "Yield" if want_yield else "Production"

    selected = ["stg_faostat_production"]
    for tid in ("stg_faostat_trade", "stg_faostat_prices", "stg_africa_gdp_ppp", "stg_faostat_investment_asti"):
        if tid in known_tables and tid not in selected:
            selected.append(tid)
        if len(selected) >= 5:
            break

    intents: list[dict[str, Any]] = [
        {
            "goal": f"Country agricultural {primary_element} ranking for the region/time window",
            "tables": ["stg_faostat_production"],
            "filters": f"element='{primary_element}'; {geo_filter}; year between {y0} and {y1}",
            "notes": f"analytical_rank_{primary_element.lower()}",
            "pattern": "rank_by_sum",
            "metric": "value",
            "grain": ["country_name"],
            "order_by": "total DESC",
        },
        {
            "goal": f"Production by country for staple crops ({', '.join(_STAPLES[:3])})",
            "tables": ["stg_faostat_production"],
            "filters": (
                f"element='Production'; items in {list(_STAPLES)}; {geo_filter}; "
                f"year≈{y1}"
            ),
            "notes": "analytical_staples_by_country",
            "pattern": "rank_by_sum",
            "metric": "value",
            "grain": ["country_name", "product_name"],
            "order_by": "total DESC",
        },
        {
            "goal": f"{primary_element} time series endpoints ({y0} vs {y1}) by country",
            "tables": ["stg_faostat_production"],
            "filters": f"element='{primary_element}'; {geo_filter}; years {y0} and {y1}",
            "notes": "analytical_series_endpoints",
            "pattern": "time_series",
            "metric": "value",
            "grain": ["country_name", "year"],
            "order_by": "year ASC",
        },
        {
            "goal": "Top commodities by production volume in the geography",
            "tables": ["stg_faostat_production"],
            "filters": f"element='Production'; {geo_filter}; year≈{y1}",
            "notes": "analytical_top_commodities",
            "pattern": "rank_by_sum",
            "metric": "value",
            "grain": ["product_name"],
            "order_by": "total DESC",
        },
    ]

    if "stg_faostat_trade" in selected:
        intents.append(
            {
                "goal": "Agricultural trade (export/import) comparison by country",
                "tables": ["stg_faostat_trade"],
                "filters": f"{geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_trade",
                "pattern": "rank_by_sum",
                "metric": "value",
                "grain": ["country_name"],
                "order_by": "total DESC",
            }
        )

    if want_yield:
        intents.append(
            {
                "goal": "Crop yield comparison across countries (element=Yield)",
                "tables": ["stg_faostat_production"],
                "filters": f"element='Yield'; {geo_filter}; year≈{y1}",
                "notes": "analytical_yield",
                "pattern": "rank_by_sum",
                "metric": "value",
                "grain": ["country_name", "product_name"],
                "order_by": "total DESC",
            }
        )

    for tid, goal in (
        ("stg_africa_gdp_ppp", "GDP PPP companion by country"),
        ("stg_faostat_investment_asti", "ASTI investment companion by country"),
    ):
        if tid in selected:
            intents.append(
                {
                    "goal": goal,
                    "tables": [tid],
                    "filters": f"{geo_filter}; year between {y0} and {y1}",
                    "notes": f"analytical_{tid}",
                    "pattern": "rank_by_sum" if table_supports_sql_pattern(tid) else "custom",
                    "metric": "value",
                    "grain": ["country_name"],
                    "order_by": "total DESC" if table_supports_sql_pattern(tid) else "value DESC",
                }
            )

    floor = analytical_sql_query_floor()
    intents = intents[:floor]

    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": "analytical_forced_multi_intent",
        "analytical_mode": True,
        "max_sql_queries": floor,
        "measure_id": hit.measure.id if hit else None,
    }
    hints, hints_truncated = pack_selected_table_hints(
        selected,
        query_terms=_pack_terms(query, decomposition),
    )
    plan["table_hints"] = hints
    plan["index_truncated"] = False
    plan["hints_truncated"] = hints_truncated
    return plan


def _pack_terms(query: str, decomposition: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    entities = decomposition.get("entities")
    if isinstance(entities, list):
        terms.extend(str(e).strip() for e in entities if str(e).strip())
    geo = decomposition.get("geography")
    if isinstance(geo, list):
        terms.extend(str(g).strip() for g in geo[:8] if str(g).strip())
    terms.extend(_STAPLES)
    if query:
        terms.append(query[:80])
    return terms


__all__ = [
    "analytical_sql_query_floor",
    "build_analytical_bq_plan",
    "build_food_security_bq_plan",
]
