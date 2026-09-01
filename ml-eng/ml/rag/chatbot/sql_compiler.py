"""Card-driven SQL compiler — SqlRequest → assemble → validate."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ml.rag.chatbot.bq_table_schema_yaml import measure_columns_mart
from ml.rag.chatbot.bundle_metrics import is_agri_activities_panel, is_multi_country_panel
from ml.rag.chatbot.class_engines.shared import mart_table_fqn, validate_engine_sql
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.value_index import complete_enum

Shape = Literal["point", "series", "panel", "share", "rank"]


def sql_compiler_enabled() -> bool:
    return os.environ.get("RAG_SQL_COMPILER", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


@dataclass
class SqlRequest:
    class_code: str
    table_id: str
    geos: list[str] = field(default_factory=list)
    year_start: int = 2010
    year_end: int = 2024
    entities: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    grain: str = ""
    shape: Shape = "series"
    value_hits: dict[str, Any] = field(default_factory=dict)


def default_measures_for_shape(card: dict[str, Any], shape: Shape) -> list[str]:
    defaults = card.get("default_measures") or {}
    if isinstance(defaults, dict):
        for key in (shape, "series", "point", "panel"):
            vals = defaults.get(key)
            if isinstance(vals, list) and vals:
                return [str(v) for v in vals]
    table = str(card.get("default_table") or "")
    if complete_enum(table, "metric"):
        return []
    cols = measure_columns_mart(table)
    return [cols[0]] if cols else ["value"]


def _year_bounds_from_facets(
    facets: dict[str, Any],
    *,
    query: str,
    bundles: tuple[Any, ...],
    panel_default_start: int = 2015,
) -> tuple[int, int]:
    ts = str(facets.get("time_start") or "")[:4]
    te = str(facets.get("time_end") or "")[:4]
    if not ts or not te:
        now = datetime.now(timezone.utc).year
        y1 = now - 1
        y0 = panel_default_start if is_agri_activities_panel(query, facets, bundles=bundles) else y1 - 4
        return y0, y1
    try:
        return int(ts), int(te)
    except ValueError:
        now = datetime.now(timezone.utc).year
        return (panel_default_start, now - 1) if is_agri_activities_panel(query, facets, bundles=bundles) else (now - 5, now - 1)


def build_sql_request_from_facets(
    *,
    class_code: str,
    table_id: str,
    query: str,
    facets: dict[str, Any],
    card: dict[str, Any],
    value_hits: dict[str, Any] | None = None,
    iso_list: list[str] | None = None,
    bundles: tuple[Any, ...] | None = None,
) -> SqlRequest:
    """Build SqlRequest from supervisor/decompose facets — engines must not re-parse geo/time."""
    bundles = bundles or match_intent_bundles(query, facets)
    hits = dict(value_hits or {})
    geos = list(iso_list or hits.get("country_iso3") or [])
    y0, y1 = _year_bounds_from_facets(facets, query=query, bundles=bundles)
    multi = is_multi_country_panel(geos)
    panel = is_agri_activities_panel(query, facets, bundles=bundles)
    dec_shape = str(facets.get("reasoner_shape") or facets.get("shape") or "").strip().lower()
    if dec_shape in ("panel", "rank", "share", "point", "series"):
        shape: Shape = dec_shape  # type: ignore[assignment]
    elif panel or multi:
        shape = "panel"
    else:
        shape = "series"
    entities = list(hits.get("product_name") or [])
    measures = list(hits.get("metric") or [])
    if not measures:
        measures = default_measures_for_shape(card, shape)
    return SqlRequest(
        class_code=class_code,
        table_id=table_id,
        geos=geos,
        year_start=y0,
        year_end=y1,
        entities=entities,
        measures=measures,
        shape=shape,
        value_hits=hits,
    )


def assemble_sql(req: SqlRequest, card: dict[str, Any]) -> str:
    """Build SELECT from bound facets and card roles — no value-index sample literals."""
    table = req.table_id
    hits = dict(req.value_hits)
    iso_vals = list(req.geos) or list(hits.get("country_iso3") or [])
    if not iso_vals:
        iso_vals = ["GHA"]
    if len(iso_vals) >= 2:
        iso_in = ", ".join(f"'{c}'" for c in iso_vals)
        iso_clause = f"country_iso3 IN ({iso_in})"
        limit = "LIMIT 500" if req.shape in ("panel", "rank") else "LIMIT 40"
    else:
        iso_clause = f"country_iso3 = '{iso_vals[0]}'"
        limit = "LIMIT 40"

    grain_clause = ""
    if complete_enum(table, "production_grain"):
        grain_clause = "\n  AND production_grain = 'physical'"
    elif complete_enum(table, "measure_type") and req.class_code == "FS":
        grain_clause = "\n  AND measure_type IN ('population', 'classification')"

    measures = req.measures or default_measures_for_shape(card, req.shape)
    measure_col = measures[0] if measures else (measure_columns_mart(table)[0] if measure_columns_mart(table) else "value")
    metric_col = ", metric" if complete_enum(table, "metric") else ""

    entity_clause = ""
    products = hits.get("product_name") or req.entities
    if products and complete_enum(table, "product_name"):
        in_list = ", ".join(f"'{p.replace(chr(39), '')}'" for p in products[:20])
        entity_clause = f"\n  AND product_name IN ({in_list})"

    return f"""SELECT country_iso3, year, {measure_col} AS value, unit{metric_col}
FROM {mart_table_fqn(table)}
WHERE {iso_clause}{grain_clause}{entity_clause}
  AND year BETWEEN {req.year_start} AND {req.year_end}
ORDER BY year DESC
{limit}"""


def compile_sql(
    req: SqlRequest,
    card: dict[str, Any],
) -> tuple[str | None, str]:
    sql = assemble_sql(req, card)
    ok, reason = validate_engine_sql(
        sql,
        table_id=req.table_id,
        selected_tables=[req.table_id],
        allowed_iso3=list(req.geos) or list(req.value_hits.get("country_iso3") or []),
    )
    if not ok:
        return None, reason
    return sql, ""


__all__ = [
    "Shape",
    "SqlRequest",
    "assemble_sql",
    "build_sql_request_from_facets",
    "compile_sql",
    "default_measures_for_shape",
    "sql_compiler_enabled",
]
