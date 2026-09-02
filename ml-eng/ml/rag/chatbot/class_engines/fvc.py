"""FVC engine: food balance, trade, import share."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, build_planned_engine_result
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3


class FvcEngine(ClassEngine):
    class_code = "FVC"

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card("FVC") or {}
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        bundles = match_intent_bundles(query, facets)
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        base_hits = bind_value_hits(card, query=query, facets=facets)
        if iso_list:
            base_hits["country_iso3"] = iso_list

        if not iso_list:
            return EngineResult(
                class_code="FVC",
                status="planner_error",
                table_id="fct_food_balance",
                sql=None,
                caveats=["no country_iso3 from facets or region expansion"],
                value_hits=base_hits,
            )

        plans = select_table_plans(
            "FVC",
            query=query,
            facets=facets,
            bundles=bundles,
            card=card,
            iso_list=iso_list,
        )
        if not plans:
            return EngineResult(
                class_code="FVC",
                status="planner_error",
                table_id="fct_food_balance",
                sql=None,
                caveats=["no_table_plans"],
                value_hits=base_hits,
            )

        return build_planned_engine_result(
            class_code="FVC",
            table_id=plans[0].table_id,
            query=query,
            facets=facets,
            card=card,
            value_hits=base_hits,
            iso_list=iso_list,
            measure_id="food_availability",
        )


__all__ = ["FvcEngine"]
