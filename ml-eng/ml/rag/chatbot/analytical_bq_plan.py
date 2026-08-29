"""Forced multi-intent BigQuery plans for analytical report mode."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import fallback_plan, resolve_measure
from ml.rag.chatbot.bq_table_schema_yaml import (
    match_product_samples,
    pack_mart_table_hints,
    table_supports_sql_pattern,
)

_STAPLES = ("Maize", "Rice", "Cassava", "Sorghum", "Millet")

_FOOD_SECURITY_TABLES = (
    "fct_food_security",
    "fct_prices",
    "fct_production",
    "fct_household",
)


def analytical_sql_query_floor() -> int:
    try:
        env = int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10)
    except ValueError:
        env = 10
    try:
        floor = int(os.environ.get("RAG_ANALYTICAL_BQ_MIN_QUERIES", "5") or 5)
    except ValueError:
        floor = 5
    return max(3, min(max(env, floor), 8))


def _asked_products(query: str, decomposition: dict[str, Any]) -> list[str]:
    blob_parts = [query or ""]
    entities = decomposition.get("entities")
    if isinstance(entities, list):
        blob_parts.extend(str(e) for e in entities if str(e).strip())
    found = match_product_samples("fct_production", " ".join(blob_parts))
    return found[:6] if found else list(_STAPLES[:3])


def _wants_trade(query: str) -> bool:
    return bool(re.search(r"\b(trade|export|import)\b", query or "", re.IGNORECASE))


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
    grain_country = ["country_iso3"] if multi else ["country_iso3", "year"]

    intents: list[dict[str, Any]] = []
    if "fct_food_security" in selected:
        intents.append(
            {
                "goal": "IPC food security by country in the scoped geography",
                "tables": ["fct_food_security"],
                "filters": f"measure_type='population'; {geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_food_security",
                "pattern": "custom",
                "metric": "value",
                "grain": grain_country,
                "order_by": "value DESC",
            }
        )
    if "fct_prices" in selected:
        intents.append(
            {
                "goal": "Market prices companion for food-security assessment",
                "tables": ["fct_prices"],
                "filters": f"price_source='fews'; {geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_prices",
                "pattern": "rank_by_sum" if multi else "custom",
                "metric": "value",
                "grain": ["country_iso3"],
                "order_by": "total DESC" if multi else "value DESC",
            }
        )
    if "fct_production" in selected:
        staples = ", ".join(_STAPLES[:4])
        intents.append(
            {
                "goal": (
                    f"Staple crop production pressure companion ({staples}) "
                    "by country — multi-country rank/IN, not single-country series"
                ),
                "tables": ["fct_production"],
                "filters": (
                    f"production_grain='physical'; items in {list(_STAPLES[:4])}; "
                    f"{geo_filter}; year between {y0} and {y1}"
                ),
                "notes": "analytical_food_security_production_companion",
                "pattern": "rank_by_sum" if multi else "custom",
                "metric": "value",
                "grain": ["country_iso3", "product_key"] if multi else ["country_iso3", "year"],
                "order_by": "total DESC" if multi else "value DESC",
            }
        )
    if "fct_household" in selected:
        intents.append(
            {
                "goal": "Household food security companion signals",
                "tables": ["fct_household"],
                "filters": f"{geo_filter}; year between {y0} and {y1}",
                "notes": "analytical_household_food_security",
                "pattern": "custom",
                "metric": "value",
                "grain": ["country_iso3"],
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
    hints, hints_truncated = pack_mart_table_hints(
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
                        "grain": ["country_iso3"],
                        "order_by": "total DESC" if table_supports_sql_pattern(tid) else "value DESC",
                    }
                )
            ontology_plan["query_intents"] = intents[:floor]
            ontology_plan["analytical_mode"] = True
            ontology_plan["max_sql_queries"] = floor
            ontology_plan["rationale"] = f"analytical_forced_{hit.measure.id}"
            return ontology_plan

    contract_decomp = dict(decomposition)
    contract_decomp["task_mode"] = "analytical"
    from ml.rag.chatbot.retrieval_contract import build_retrieval_contract, contract_to_bq_plan

    contract = build_retrieval_contract(
        query,
        decomposition=contract_decomp,
        known_tables=known_tables,
    )
    if contract.bq_tables and contract.bq_intents and not contract.skip_bq:
        contract_plan = contract_to_bq_plan(
            contract,
            query=query,
            decomposition=contract_decomp,
        )
        if contract_plan is not None and not contract_plan.get("skip_bq"):
            floor = analytical_sql_query_floor()
            contract_plan["analytical_mode"] = True
            contract_plan["max_sql_queries"] = floor
            contract_plan["rationale"] = "analytical_forced_contract"
            return contract_plan

    if "fct_production" not in known_tables:
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
    products = _asked_products(query, decomposition)
    product_filter = "items in " + ", ".join(products)
    want_yield = bool(re.search(r"\byields?\b", query or "", re.IGNORECASE))
    want_trade = _wants_trade(query)
    prod_table = "fct_yield" if want_yield else "fct_production"
    grain_filter = "season_key" if want_yield else "production_grain='physical'"

    selected = [prod_table]
    if want_trade and "fct_trade" in known_tables:
        selected.append("fct_trade")

    rank_grain = ["country_iso3", "product_key"] if len(products) > 1 else ["country_iso3"]
    series_grain = (
        ["country_iso3", "product_key", "year"] if len(products) > 1 else ["country_iso3", "year"]
    )

    intents: list[dict[str, Any]] = [
        {
            "goal": f"Country {'yield' if want_yield else 'production'} ranking for {', '.join(products)}",
            "tables": [prod_table],
            "filters": f"{grain_filter}; {product_filter}; {geo_filter}; year≈{y1}",
            "notes": "analytical_products_by_country",
            "pattern": "rank_by_sum",
            "metric": "value",
            "grain": rank_grain,
            "order_by": "total DESC",
        },
        {
            "goal": f"{'Yield' if want_yield else 'Production'} time series by country for {', '.join(products)}",
            "tables": [prod_table],
            "filters": f"{grain_filter}; {product_filter}; {geo_filter}; years {y0} and {y1}",
            "notes": "analytical_series_endpoints",
            "pattern": "time_series",
            "metric": "value",
            "grain": series_grain,
            "order_by": "year ASC",
        },
    ]

    if "fct_trade" in selected:
        for trade_grain, notes in (
            ("faostat_country_year export", "analytical_trade_export"),
            ("faostat_country_year import", "analytical_trade_import"),
        ):
            intents.append(
                {
                    "goal": f"Trade {notes} by country for {', '.join(products)}",
                    "tables": ["fct_trade"],
                    "filters": f"trade_grain='{trade_grain.split()[0]}'; {product_filter}; {geo_filter}; year between {y0} and {y1}",
                    "notes": notes,
                    "pattern": "rank_by_sum",
                    "metric": "value",
                    "grain": rank_grain,
                    "order_by": "total DESC",
                }
            )

    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": "analytical_forced_multi_intent",
        "analytical_mode": True,
        "max_sql_queries": max(3, len(intents)),
        "measure_id": hit.measure.id if hit else None,
    }
    hints, hints_truncated = pack_mart_table_hints(
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
    terms.extend(_asked_products(query, decomposition))
    if query:
        terms.append(query[:80])
    return terms


__all__ = [
    "analytical_sql_query_floor",
    "build_analytical_bq_plan",
    "build_food_security_bq_plan",
]
