"""Card-driven engine for indicator classes without bespoke panel logic."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import measure_columns_mart
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    validate_engine_sql,
)
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_compiler import compile_sql, sql_compiler_enabled
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.value_index import complete_enum, resolve_geography_iso3


def _legacy_card_sql(
    *,
    table: str,
    class_code: str,
    iso_vals: list[str],
    y0: int,
    y1: int,
) -> str:
    if len(iso_vals) >= 2:
        iso_in = ", ".join(f"'{c}'" for c in iso_vals)
        iso_clause = f"country_iso3 IN ({iso_in})"
        limit = "LIMIT 500"
    else:
        iso_clause = f"country_iso3 = '{iso_vals[0]}'"
        limit = "LIMIT 40"
    grain_clause = ""
    if complete_enum(table, "production_grain"):
        grain_clause = "\n  AND production_grain = 'physical'"
    elif complete_enum(table, "measure_type") and class_code == "FS":
        grain_clause = "\n  AND measure_type IN ('population', 'classification')"
    measures = measure_columns_mart(table)
    measure_col = measures[0] if measures else "value"
    metric_col = ", metric" if complete_enum(table, "metric") else ""
    return f"""SELECT country_iso3, year, {measure_col} AS value, unit{metric_col}
FROM {mart_table_fqn(table)}
WHERE {iso_clause}{grain_clause}
  AND year BETWEEN {y0} AND {y1}
ORDER BY year DESC
{limit}"""


class CardDrivenEngine(ClassEngine):
    """Schema-card + mart YAML engine for the 11 non-bespoke indicator classes."""

    def __init__(self, class_code: str) -> None:
        self.class_code = class_code.upper()

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card(self.class_code) or {}
        bundles = match_intent_bundles(query, facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)

        plans = select_table_plans(
            self.class_code,
            query=query,
            facets=facets,
            bundles=bundles,
            card=card,
            iso_list=iso_list,
        )
        if not plans:
            return EngineResult(
                class_code=self.class_code,
                status="planner_error",
                table_id="",
                sql=None,
                caveats=["no_table_plans"],
            )

        plan = plans[0]
        table = plan.table_id
        hits = bind_value_hits(card, query=query, facets=facets)
        if iso_list:
            hits["country_iso3"] = iso_list
        elif not hits.get("country_iso3"):
            return EngineResult(
                class_code=self.class_code,
                status="planner_error",
                table_id=table,
                sql=None,
                caveats=["missing_geography"],
                value_hits=hits,
            )

        if sql_compiler_enabled():
            req = build_sql_request_from_facets(
                class_code=self.class_code,
                table_id=table,
                query=query,
                facets=facets,
                card=card,
                value_hits=hits,
                iso_list=iso_list or list(hits.get("country_iso3") or []),
                bundles=bundles,
            )
            sql, reason = compile_sql(req, card)
            ok = sql is not None
        else:
            iso_vals = list(hits.get("country_iso3") or [])
            ts = str(facets.get("time_start") or "2010")[:4]
            te = str(facets.get("time_end") or "2024")[:4]
            try:
                y0, y1 = int(ts), int(te)
            except ValueError:
                y0, y1 = 2010, 2024
            sql = _legacy_card_sql(
                table=table,
                class_code=self.class_code,
                iso_vals=iso_vals,
                y0=y0,
                y1=y1,
            )
            ok, reason = validate_engine_sql(
                sql,
                table_id=table,
                selected_tables=[table],
                allowed_iso3=iso_vals,
            )

        return EngineResult(
            class_code=self.class_code,
            status="ready" if ok else "planner_error",
            table_id=table,
            sql=sql if ok else None,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


# Back-compat alias
GenericEngine = CardDrivenEngine

__all__ = ["CardDrivenEngine", "GenericEngine"]
