"""Generic card-driven engine for remaining indicator classes."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    pack_engine_prompt,
    validate_engine_sql,
)
from ml.rag.chatbot.mart_indicator_classes import facts_for_class
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import complete_enum, resolve_country


class GenericEngine(ClassEngine):
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
        table = str(card.get("default_table") or "")
        if not table:
            facts = facts_for_class(self.class_code)
            table = facts[0] if facts else ""
        if not table:
            return EngineResult(
                class_code=self.class_code,
                status="planner_error",
                table_id="",
                sql=None,
                caveats=["no_default_table"],
            )
        hits = bind_value_hits(card, query=query, facets=facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        iso = resolve_country(query, geography=geography)
        if iso:
            hits.setdefault("country_iso3", [iso])

        ts = str(facets.get("time_start") or "2010")[:4]
        te = str(facets.get("time_end") or "2024")[:4]
        try:
            y0, y1 = int(ts), int(te)
        except ValueError:
            y0, y1 = 2010, 2024

        iso_val = (hits.get("country_iso3") or ["GHA"])[0]
        grain_clause = ""
        if complete_enum(table, "production_grain"):
            grain_clause = "\n  AND production_grain = 'physical'"
        elif complete_enum(table, "measure_type") and self.class_code == "FS":
            grain_clause = "\n  AND measure_type IN ('population', 'classification')"

        sql = f"""SELECT country_iso3, year, value, unit, metric
FROM {mart_table_fqn(table)}
WHERE country_iso3 = '{iso_val}'{grain_clause}
  AND year BETWEEN {y0} AND {y1}
ORDER BY year DESC
LIMIT 40"""

        ok, reason = validate_engine_sql(sql, table_id=table, selected_tables=[table])
        _ = pack_engine_prompt(card, query=query, facets=facets, value_hits=hits)
        return EngineResult(
            class_code=self.class_code,
            status="planned" if ok else "planner_error",
            table_id=table,
            sql=sql,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


__all__ = ["GenericEngine"]
