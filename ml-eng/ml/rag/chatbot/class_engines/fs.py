"""FS engine: food security IPC and household FIES."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, mart_table_fqn, validate_engine_sql
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_compiler import compile_sql, sql_compiler_enabled
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.value_index import resolve_geography_iso3

_IPC_POP_RE = re.compile(r"\b(ipc|phase\s*[345]|population|people|humanitarian)\b", re.I)


def _agg_coverage_countries(card: dict[str, Any]) -> set[str]:
    coverage = card.get("coverage") or {}
    agg = coverage.get("agg_food_security_monthly") or coverage.get("agg_food_security_country_month")
    if not isinstance(agg, dict):
        return set()
    raw = agg.get("country_iso3") or []
    return {str(c).strip().upper() for c in raw if str(c).strip()}


def _year_bounds(facets: dict[str, Any]) -> tuple[int, int]:
    ts = str(facets.get("time_start") or "")[:4]
    te = str(facets.get("time_end") or "")[:4]
    try:
        return int(ts), int(te)
    except ValueError:
        return 2020, 2024


def _build_food_security_sql(*, iso: str, y0: int, y1: int) -> str:
    return f"""SELECT
  country_iso3,
  year,
  month,
  measure_type,
  metric,
  value,
  unit,
  scenario_name
FROM {mart_table_fqn("fct_food_security")}
WHERE country_iso3 = '{iso}'
  AND measure_type = 'population'
  AND year BETWEEN {y0} AND {y1}
ORDER BY year DESC, month DESC
LIMIT 48"""


def _build_agg_food_security_sql(*, iso: str, y0: int, y1: int) -> str:
    return f"""SELECT
  country_iso3,
  year,
  month,
  measure_type,
  metric,
  value,
  unit
FROM {mart_table_fqn("agg_food_security_country_month")}
WHERE country_iso3 = '{iso}'
  AND measure_type = 'population'
  AND year BETWEEN {y0} AND {y1}
ORDER BY year DESC, month DESC
LIMIT 48"""


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
        y0, y1 = _year_bounds(facets)
        covered = _agg_coverage_countries(card)
        use_agg = iso.upper() in covered and _IPC_POP_RE.search(query)
        table = "agg_food_security_country_month" if use_agg else "fct_food_security"

        hits = bind_value_hits(card, query=query, facets=facets)
        hits["country_iso3"] = iso_list

        if sql_compiler_enabled():
            req = build_sql_request_from_facets(
                class_code="FS",
                table_id=table,
                query=query,
                facets=facets,
                card=card,
                value_hits=hits,
                iso_list=iso_list,
                bundles=bundles,
            )
            sql, reason = compile_sql(req, card)
            ok = sql is not None
            if not ok:
                sql = (
                    _build_agg_food_security_sql(iso=iso, y0=y0, y1=y1)
                    if use_agg
                    else _build_food_security_sql(iso=iso, y0=y0, y1=y1)
                )
                ok, reason = validate_engine_sql(
                    sql,
                    table_id=table,
                    selected_tables=[table],
                    allowed_iso3=iso_list,
                )
        else:
            sql = (
                _build_agg_food_security_sql(iso=iso, y0=y0, y1=y1)
                if use_agg
                else _build_food_security_sql(iso=iso, y0=y0, y1=y1)
            )
            ok, reason = validate_engine_sql(
                sql,
                table_id=table,
                selected_tables=[table],
                allowed_iso3=iso_list,
            )

        return EngineResult(
            class_code="FS",
            status="ready" if ok else "planner_error",
            table_id=table,
            sql=sql if ok else None,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


__all__ = ["FsEngine"]
