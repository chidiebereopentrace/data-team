"""PROD engine: national production series and multi-country agg panels."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ml.rag.chatbot.bundle_metrics import (
    is_agri_activities_panel,
    is_multi_country_panel,
    resolve_staple_products,
    unsupported_grain_for_panel,
)
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    validate_engine_sql,
)
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_compiler import SqlRequest, build_sql_request_from_facets
from ml.rag.chatbot.value_index import resolve_geography_iso3


def _year_bounds(
    req: SqlRequest,
    *,
    query: str,
    facets: dict[str, Any],
    bundles: tuple[Any, ...],
) -> tuple[int, int]:
    if req.year_start and req.year_end:
        return req.year_start, req.year_end
    ts = str(facets.get("time_start") or "")[:4]
    te = str(facets.get("time_end") or "")[:4]
    if not ts or not te:
        now = datetime.now(timezone.utc).year
        y1 = now - 1
        y0 = 2015 if is_agri_activities_panel(query, facets, bundles=bundles) else y1 - 4
        return y0, y1
    try:
        return int(ts), int(te)
    except ValueError:
        now = datetime.now(timezone.utc).year
        return (2015, now - 1) if is_agri_activities_panel(query, facets, bundles=bundles) else (now - 5, now - 1)


def _build_prod_sql(
    table_id: str,
    *,
    iso_list: list[str],
    products: list[str],
    y0: int,
    y1: int,
) -> str:
    product_clause = ""
    if products:
        in_list = ", ".join(f"'{p.replace(chr(39), '')}'" for p in products)
        product_clause = f"\n  AND p.product_name IN ({in_list})"

    multi = is_multi_country_panel(iso_list)

    if table_id == "agg_production_country_year":
        if multi:
            iso_in = ", ".join(f"'{c}'" for c in iso_list)
            return f"""SELECT
  g.country_iso3,
  p.product_name,
  a.year,
  SUM(a.production_qty) AS production_qty
FROM {mart_table_fqn(table_id)} a
JOIN {mart_table_fqn('dim_geography')} g ON a.geography_key = g.geography_key
JOIN {mart_table_fqn('dim_product')} p ON a.product_key = p.product_key
WHERE g.country_iso3 IN ({iso_in}){product_clause}
  AND a.year BETWEEN {y0} AND {y1}
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
LIMIT 500"""
        iso = iso_list[0]
        return f"""SELECT
  g.country_iso3,
  p.product_name,
  a.year,
  a.production_qty,
  a.area_harvested,
  a.yield_recomputed
FROM {mart_table_fqn(table_id)} a
JOIN {mart_table_fqn('dim_geography')} g ON a.geography_key = g.geography_key
JOIN {mart_table_fqn('dim_product')} p ON a.product_key = p.product_key
WHERE g.country_iso3 = '{iso}'{product_clause}
  AND a.year BETWEEN {y0} AND {y1}
ORDER BY a.year DESC
LIMIT 40"""

    if table_id == "fct_production":
        if multi:
            iso_in = ", ".join(f"'{c}'" for c in iso_list)
            iso_clause = f"g.country_iso3 IN ({iso_in})"
            limit = "LIMIT 500"
        else:
            iso_clause = f"g.country_iso3 = '{iso_list[0]}'"
            limit = "LIMIT 40"
        return f"""SELECT
  g.country_iso3,
  p.product_name,
  f.year,
  f.production_qty,
  f.area_harvested,
  f.yield_qty
FROM {mart_table_fqn(table_id)} f
JOIN {mart_table_fqn('dim_geography')} g ON f.geography_key = g.geography_key
JOIN {mart_table_fqn('dim_product')} p ON f.product_key = p.product_key
WHERE {iso_clause}
  AND f.production_grain = 'physical'{product_clause}
  AND f.year BETWEEN {y0} AND {y1}
ORDER BY f.year DESC
{limit}"""

    if table_id == "agg_production_annual":
        iso = iso_list[0]
        return f"""SELECT
  country_code,
  product_name,
  time_key AS year,
  total_production_qty AS production_qty
FROM {mart_table_fqn(table_id)}
WHERE country_code = '{iso}'{product_clause.replace('p.product_name', 'product_name')}
  AND time_key BETWEEN {y0} AND {y1}
ORDER BY time_key DESC
LIMIT 40"""

    raise ValueError(f"unsupported PROD table {table_id}")


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
        hits["product_name"] = products
        req = build_sql_request_from_facets(
            class_code="PROD",
            table_id=table_id,
            query=query,
            facets=facets,
            card=card,
            value_hits=hits,
            iso_list=iso_list,
            bundles=bundles,
        )
        y0, y1 = _year_bounds(req, query=query, facets=facets, bundles=bundles)

        try:
            sql = _build_prod_sql(
                table_id,
                iso_list=iso_list,
                products=products,
                y0=y0,
                y1=y1,
            )
        except ValueError as exc:
            return EngineResult(
                class_code="PROD",
                status="planner_error",
                table_id=table_id,
                sql=None,
                caveats=[str(exc)],
                value_hits=hits,
            )

        ok, reason = validate_engine_sql(
            sql,
            table_id=table_id,
            selected_tables=[table_id, "dim_geography", "dim_product"],
            allowed_iso3=iso_list,
        )
        return EngineResult(
            class_code="PROD",
            status="ready" if ok else "planner_error",
            table_id=table_id,
            sql=sql,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


__all__ = ["ProdEngine"]
