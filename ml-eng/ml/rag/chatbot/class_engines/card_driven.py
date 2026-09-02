"""Card-driven engine for indicator classes without bespoke panel logic."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, build_planned_engine_result
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3


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

        return build_planned_engine_result(
            class_code=self.class_code,
            table_id=table,
            query=query,
            facets=facets,
            card=card,
            value_hits=hits,
            iso_list=iso_list or list(hits.get("country_iso3") or []),
        )


GenericEngine = CardDrivenEngine

__all__ = ["CardDrivenEngine", "GenericEngine"]
