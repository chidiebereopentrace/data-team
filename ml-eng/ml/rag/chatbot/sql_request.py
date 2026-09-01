"""SqlRequest facet binding — shared by all 15 indicator class engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ml.rag.chatbot.bq_table_schema_yaml import measure_columns_mart
from ml.rag.chatbot.bundle_metrics import is_agri_activities_panel, is_multi_country_panel
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.value_index import complete_enum

Shape = Literal["point", "series", "panel", "share", "rank"]


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


__all__ = [
    "Shape",
    "SqlRequest",
    "build_sql_request_from_facets",
    "default_measures_for_shape",
]
