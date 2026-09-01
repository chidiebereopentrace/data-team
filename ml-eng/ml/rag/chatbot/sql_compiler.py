"""Card-driven SQL compiler — SqlRequest → assemble → validate."""
from __future__ import annotations

import os
from typing import Any

from ml.rag.chatbot.bq_engine_validate import validate_engine_sql
from ml.rag.chatbot.bq_mart_sql import mart_table_fqn
from ml.rag.chatbot.bq_table_schema_yaml import measure_columns_mart
from ml.rag.chatbot.sql_request import SqlRequest, Shape, default_measures_for_shape
from ml.rag.chatbot.value_index import complete_enum


def sql_compiler_enabled() -> bool:
    return os.environ.get("RAG_SQL_COMPILER", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
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
    "compile_sql",
    "default_measures_for_shape",
    "sql_compiler_enabled",
]
