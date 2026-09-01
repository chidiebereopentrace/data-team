"""FVC engine: food balance, trade, import share."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    validate_engine_sql,
)
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import complete_enum, resolve_labels, resolve_metric

_SHARE_RE = re.compile(
    r"\b(import\s+(share|dependency|ratio)|domestic\s+supply|self[-\s]?sufficien)\b",
    re.I,
)


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
        table = "fct_food_balance"
        hits = bind_value_hits(card, query=query, facets=facets)
        iso_list = hits.get("country_iso3") or []
        iso = iso_list[0] if iso_list else "GHA"
        products = resolve_labels(
            table,
            "product_name",
            query,
            scope="fact_distinct",
            geography=facets.get("geography") if isinstance(facets.get("geography"), list) else None,
        )
        if not products and "wheat" in query.lower():
            products = resolve_labels(table, "product_name", "wheat", scope="fact_distinct")
        metrics = resolve_metric(query, class_code="FVC", table_id=table)
        if not metrics and _SHARE_RE.search(query):
            metrics = [
                "food_balance_import_quantity",
                "food_balance_domestic_supply_quantity",
            ]
        hits["metric"] = metrics or complete_enum(table, "metric")
        hits["product_name"] = products
        hits["source_natural_key"] = complete_enum(table, "source_natural_key")

        ts = str(facets.get("time_start") or "2010")[:4]
        te = str(facets.get("time_end") or "2024")[:4]
        try:
            y0, y1 = int(ts), int(te)
        except ValueError:
            y0, y1 = 2010, 2024

        product_clause = ""
        if products:
            in_list = ", ".join(f"'{p.replace(chr(39), '')}'" for p in products)
            product_clause = f"\n  AND p.product_name IN ({in_list})"

        metric_clause = ", ".join(
            f"'{m}'" for m in (metrics or ["food_balance_import_quantity", "food_balance_domestic_supply_quantity"])
        )

        sql = f"""SELECT
  f.country_iso3,
  p.product_name,
  f.metric,
  f.element,
  f.year,
  f.value,
  f.unit,
  f.source_natural_key
FROM {mart_table_fqn(table)} f
JOIN {mart_table_fqn('dim_product')} p
  ON f.product_key = p.product_key
WHERE f.country_iso3 = '{iso}'
  AND f.metric IN ({metric_clause}){product_clause}
  AND f.year BETWEEN {y0} AND {y1}
ORDER BY f.year DESC
LIMIT 40"""

        ok, reason = validate_engine_sql(sql, table_id=table, selected_tables=[table, "dim_product"])
        status = "planned" if ok else "planner_error"
        caveats = [] if ok else [reason]
        return EngineResult(
            class_code="FVC",
            status=status,
            table_id=table,
            sql=sql if ok else sql,
            caveats=caveats,
            value_hits=hits,
        )


__all__ = ["FvcEngine"]
