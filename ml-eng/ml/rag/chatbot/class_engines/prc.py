"""PRC engine: prices and markets (FEWS/WFP/FAOSTAT grains)."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bundle_metrics import resolve_staple_products
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, build_planned_engine_result
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3

_MARKET_DETAIL_RE = re.compile(r"\b(market|bamako|kano|nairobi|retail|wholesale|urban)\b", re.I)


class PrcEngine(ClassEngine):
    class_code = "PRC"

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card("PRC") or {}
        bundles = match_intent_bundles(query, facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        if not iso_list:
            return EngineResult(
                class_code="PRC",
                status="planner_error",
                table_id="",
                sql=None,
                caveats=["missing_geography"],
            )

        products = resolve_staple_products(query, facets)
        use_market = bool(_MARKET_DETAIL_RE.search(query)) and len(iso_list) == 1
        table = "fct_prices" if use_market else "agg_prices_country_month"

        hits = bind_value_hits(card, query=query, facets=facets)
        hits["country_iso3"] = iso_list
        if products:
            hits["product_name"] = products

        return build_planned_engine_result(
            class_code="PRC",
            table_id=table,
            query=query,
            facets=facets,
            card=card,
            value_hits=hits,
            iso_list=iso_list,
            measure_id="market_price",
        )


__all__ = ["PrcEngine"]
