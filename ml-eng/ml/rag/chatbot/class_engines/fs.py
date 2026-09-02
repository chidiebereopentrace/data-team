"""FS engine: food security IPC and household FIES."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, build_planned_engine_result
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3

_IPC_POP_RE = re.compile(r"\b(ipc|phase\s*[345]|population|people|humanitarian)\b", re.I)


def _agg_coverage_countries(card: dict[str, Any]) -> set[str]:
    coverage = card.get("coverage") or {}
    agg = coverage.get("agg_food_security_monthly") or coverage.get("agg_food_security_country_month")
    if not isinstance(agg, dict):
        return set()
    raw = agg.get("country_iso3") or []
    return {str(c).strip().upper() for c in raw if str(c).strip()}


class FsEngine(ClassEngine):
    class_code = "FS"

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card("FS") or {}
        bundles = match_intent_bundles(query, facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        if not iso_list:
            return EngineResult(
                class_code="FS",
                status="planner_error",
                table_id="",
                sql=None,
                caveats=["missing_geography"],
            )

        iso = iso_list[0]
        covered = _agg_coverage_countries(card)
        use_agg = iso.upper() in covered and _IPC_POP_RE.search(query)
        table = "agg_food_security_monthly" if use_agg else "fct_food_security"

        hits = bind_value_hits(card, query=query, facets=facets)
        hits["country_iso3"] = iso_list

        return build_planned_engine_result(
            class_code="FS",
            table_id=table,
            query=query,
            facets=facets,
            card=card,
            value_hits=hits,
            iso_list=iso_list,
            measure_id="food_security",
        )


__all__ = ["FsEngine"]
