"""PROD engine: national production series and multi-country agg panels."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bundle_metrics import (
    is_multi_country_panel,
    resolve_staple_products,
    unsupported_grain_for_panel,
)
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, build_planned_engine_result
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3


class ProdEngine(ClassEngine):
    class_code = "PROD"

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card("PROD") or {}
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        bundles = match_intent_bundles(query, facets)
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        if not iso_list:
            return EngineResult(
                class_code="PROD",
                status="planner_error",
                table_id=str(card.get("default_table") or "agg_production_country_year"),
                sql=None,
                caveats=["no country_iso3 from facets or region expansion"],
                value_hits={},
            )

        plans = select_table_plans(
            "PROD",
            query=query,
            facets=facets,
            bundles=bundles,
            card=card,
            iso_list=iso_list,
        )
        table_id = plans[0].table_id if plans else str(card.get("default_table") or "agg_production_country_year")

        if unsupported_grain_for_panel(table_id, iso_count=len(iso_list)):
            return EngineResult(
                class_code="PROD",
                status="unsupported_grain",
                table_id=table_id,
                sql=None,
                caveats=[f"panel grain unsupported on {table_id}"],
                value_hits={"country_iso3": iso_list},
            )

        hits = bind_value_hits(card, query=query, facets=facets)
        hits["country_iso3"] = iso_list
        products = resolve_staple_products(query, facets, bundles=bundles)
        if products:
            hits["product_name"] = products

        return build_planned_engine_result(
            class_code="PROD",
            table_id=table_id,
            query=query,
            facets=facets,
            card=card,
            value_hits=hits,
            iso_list=iso_list,
            measure_id="production",
        )


__all__ = ["ProdEngine"]
