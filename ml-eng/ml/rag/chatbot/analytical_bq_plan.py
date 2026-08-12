"""Forced multi-intent BigQuery plans for analytical report mode."""
from __future__ import annotations

import os
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import pack_selected_table_hints

_STAPLES = ("Maize", "Rice", "Cassava", "Sorghum", "Millet")


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


def build_analytical_bq_plan(
    query: str,
    *,
    decomposition: dict[str, Any],
    known_tables: set[str],
) -> dict[str, Any] | None:
    """
    Deterministic multi-intent plan for agricultural comparative / report queries.

    Never sets skip_bq. Returns None only when no staging production table exists.
    """
    if "stg_faostat_production" not in known_tables:
        return None

    geo = decomposition.get("geography") if isinstance(decomposition.get("geography"), list) else []
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

    selected = ["stg_faostat_production"]
    for tid in ("stg_faostat_trade", "stg_faostat_prices", "stg_faostat_yield"):
        # Prefer real staging ids when present in catalog.
        if tid in known_tables and tid not in selected:
            selected.append(tid)
        if len(selected) >= 4:
            break

    intents: list[dict[str, Any]] = [
        {
            "goal": "Country agricultural production ranking for the region/time window",
            "tables": ["stg_faostat_production"],
            "filters": f"element='Production'; {geo_filter}; year between {y0} and {y1}",
            "notes": "analytical_rank_production",
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
            "pattern": "custom",
            "metric": "value",
            "grain": ["country_name", "item"],
            "order_by": "value DESC",
        },
        {
            "goal": f"Production time series endpoints ({y0} vs {y1}) by country",
            "tables": ["stg_faostat_production"],
            "filters": f"element='Production'; {geo_filter}; years {y0} and {y1}",
            "notes": "analytical_series_endpoints",
            "pattern": "custom",
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
            "grain": ["item"],
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
                "pattern": "custom",
                "metric": "value",
                "grain": ["country_name"],
                "order_by": "value DESC",
            }
        )

    # Pad toward floor with yield if available.
    if "stg_faostat_yield" in known_tables or any("yield" in t for t in known_tables):
        yield_tid = "stg_faostat_yield" if "stg_faostat_yield" in known_tables else None
        if yield_tid is None:
            for t in known_tables:
                if "yield" in t and t.startswith("stg_"):
                    yield_tid = t
                    break
        if yield_tid:
            if yield_tid not in selected:
                selected.append(yield_tid)
            intents.append(
                {
                    "goal": "Crop yield comparison across countries",
                    "tables": [yield_tid],
                    "filters": f"{geo_filter}; year≈{y1}",
                    "notes": "analytical_yield",
                    "pattern": "custom",
                    "metric": "value",
                    "grain": ["country_name", "item"],
                    "order_by": "value DESC",
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


__all__ = ["analytical_sql_query_floor", "build_analytical_bq_plan"]
