"""Thin 1–2 intent BQ plans for fact_lookup and data_export_only modes."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import pack_selected_table_hints


def build_fact_bq_plan(
    query: str,
    *,
    decomposition: dict[str, Any],
    known_tables: set[str],
    task_mode: str = "fact_lookup",
) -> dict[str, Any] | None:
    """Forced skip_bq=false plan with one or two focused intents."""
    if "stg_faostat_production" not in known_tables:
        return None

    geo = decomposition.get("geography") if isinstance(decomposition.get("geography"), list) else []
    countries = [str(g).strip() for g in geo if str(g).strip()]
    geo_filter = (
        f"country_name in ({', '.join(countries[:8])})"
        if countries
        else "Africa continental ranking when unscoped"
    )
    ts = str(decomposition.get("time_start") or "")[:10]
    te = str(decomposition.get("time_end") or "")[:10]
    year_hint = (te or ts or "")[:4] or "year from question"

    entities = decomposition.get("entities") if isinstance(decomposition.get("entities"), list) else []
    item_hint = ", ".join(str(e).strip() for e in entities[:4] if str(e).strip()) or "crop from question"

    intents: list[dict[str, Any]] = [
        {
            "goal": "Primary production (or ranking) fact for the asked geography/commodity",
            "tables": ["stg_faostat_production"],
            "filters": f"element='Production'; {geo_filter}; year≈{year_hint}; item≈{item_hint}",
            "notes": f"fact_{task_mode}",
            "pattern": "rank_by_sum" if not countries or len(countries) > 1 else "custom",
            "metric": "value",
            "grain": ["country_name"] if not countries or len(countries) != 1 else ["country_name", "item"],
            "order_by": "value DESC",
        }
    ]
    if task_mode == "data_export_only":
        intents.append(
            {
                "goal": "Tabular series or multi-row export for the same filters",
                "tables": ["stg_faostat_production"],
                "filters": f"element='Production'; {geo_filter}; year around {year_hint}",
                "notes": "fact_export_table",
                "pattern": "custom",
                "metric": "value",
                "grain": ["country_name", "item", "year"],
                "order_by": "year ASC",
            }
        )

    selected = ["stg_faostat_production"]
    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": f"fact_forced_{task_mode}",
        "task_mode": task_mode,
        "max_sql_queries": 3,
    }
    terms = [query[:80], item_hint, *countries[:5]]
    hints, hints_truncated = pack_selected_table_hints(selected, query_terms=terms)
    plan["table_hints"] = hints
    plan["index_truncated"] = False
    plan["hints_truncated"] = hints_truncated
    return plan


__all__ = ["build_fact_bq_plan"]
