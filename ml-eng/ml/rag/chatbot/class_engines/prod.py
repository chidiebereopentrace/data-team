"""PROD engine: national production series."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    validate_engine_sql,
)
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_country, resolve_labels

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
        table = "agg_production_country_year"
        hits = bind_value_hits(card, query=query, facets=facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        iso = resolve_country(query, geography=geography) or (hits.get("country_iso3") or ["GHA"])[0]
        hits["country_iso3"] = [iso]

        products = resolve_labels(
            "fct_production",
            "product_name",
            query,
            scope="fact_distinct",
            geography=geography,
        )
        if not products:
            for ent in facets.get("entities") or []:
                products.extend(
                    resolve_labels("fct_production", "product_name", str(ent), scope="fact_distinct")
                )
        products = list(dict.fromkeys(products))
        hits["product_name"] = products

        ts = str(facets.get("time_start") or "")[:4]
        te = str(facets.get("time_end") or "")[:4]
        if not ts or not te:
            now = datetime.now(timezone.utc).year
            y1 = now - 1
            y0 = y1 - 4
        else:
            try:
                y0, y1 = int(ts), int(te)
            except ValueError:
                now = datetime.now(timezone.utc).year
                y1 = now - 1
                y0 = y1 - 4

        product_clause = ""
        if products:
            in_list = ", ".join(f"'{p.replace(chr(39), '')}'" for p in products)
            product_clause = f"\n  AND p.product_name IN ({in_list})"

        sql = f"""SELECT
  g.country_iso3,
  g.country_name,
  p.product_name,
  a.year,
  a.value,
  a.unit
FROM {mart_table_fqn(table)} a
JOIN {mart_table_fqn('dim_geography')} g ON a.geography_key = g.geography_key
JOIN {mart_table_fqn('dim_product')} p ON a.product_key = p.product_key
WHERE g.country_iso3 = '{iso}'{product_clause}
  AND a.year BETWEEN {y0} AND {y1}
ORDER BY a.year DESC
LIMIT 40"""

        ok, reason = validate_engine_sql(
            sql,
            table_id=table,
            selected_tables=[table, "dim_geography", "dim_product"],
        )
        return EngineResult(
            class_code="PROD",
            status="planned" if ok else "planner_error",
            table_id=table,
            sql=sql,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


__all__ = ["ProdEngine"]
