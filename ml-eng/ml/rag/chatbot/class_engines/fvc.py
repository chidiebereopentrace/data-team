"""FVC engine: food balance, trade, import share."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bundle_metrics import (
    default_fvc_metrics_for_panel,
    default_fvc_trade_metrics_for_panel,
    is_agri_activities_panel,
    is_multi_country_panel,
    resolve_staple_products,
)
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import (
    bind_value_hits,
    mart_table_fqn,
    validate_engine_sql,
)
from ml.rag.chatbot.class_table_router import TablePlan, select_table_plans
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.value_index import complete_enum, resolve_geography_iso3, resolve_labels, resolve_metric

_SHARE_RE = re.compile(
    r"\b(import\s+(share|dependency|ratio)|domestic\s+supply|self[-\s]?sufficien)\b",
    re.I,
)


def _year_bounds(facets: dict[str, Any], *, req_years: tuple[int, int] | None = None) -> tuple[int, int]:
    if req_years:
        return req_years
    ts = str(facets.get("time_start") or "2010")[:4]
    te = str(facets.get("time_end") or "2024")[:4]
    try:
        return int(ts), int(te)
    except ValueError:
        return 2010, 2024


def _product_clause(products: list[str]) -> str:
    if not products:
        return ""
    in_list = ", ".join(f"'{p.replace(chr(39), '')}'" for p in products)
    return f"\n  AND p.product_name IN ({in_list})"


def _build_food_balance_sql(
    *,
    iso_list: list[str],
    products: list[str],
    metrics: list[str],
    y0: int,
    y1: int,
    multi: bool,
    include_source_key: bool,
) -> str:
    table = "fct_food_balance"
    metric_clause = ", ".join(f"'{m}'" for m in metrics)
    product_clause = _product_clause(products)
    if multi:
        iso_in = ", ".join(f"'{c}'" for c in iso_list)
        return f"""SELECT
  f.country_iso3,
  p.product_name,
  f.metric,
  f.year,
  SUM(f.value) AS value,
  f.unit
FROM {mart_table_fqn(table)} f
JOIN {mart_table_fqn('dim_product')} p
  ON f.product_key = p.product_key
WHERE f.country_iso3 IN ({iso_in})
  AND f.metric IN ({metric_clause}){product_clause}
  AND f.year BETWEEN {y0} AND {y1}
GROUP BY 1, 2, 3, 4, 6
ORDER BY 1, 2, 3
LIMIT 500"""
    iso = iso_list[0]
    source_col = ",\n  f.source_natural_key" if include_source_key else ""
    return f"""SELECT
  f.country_iso3,
  p.product_name,
  f.metric,
  f.element,
  f.year,
  f.value,
  f.unit{source_col}
FROM {mart_table_fqn(table)} f
JOIN {mart_table_fqn('dim_product')} p
  ON f.product_key = p.product_key
WHERE f.country_iso3 = '{iso}'
  AND f.metric IN ({metric_clause}){product_clause}
  AND f.year BETWEEN {y0} AND {y1}
ORDER BY f.year DESC
LIMIT 40"""


def _build_trade_sql(
    *,
    iso_list: list[str],
    products: list[str],
    metrics: list[str],
    y0: int,
    y1: int,
    multi: bool,
) -> str:
    table = "fct_trade"
    metric_clause = ", ".join(f"'{m}'" for m in metrics)
    product_clause = _product_clause(products)
    grain_clause = "\n  AND f.trade_grain = 'faostat_country_year'"
    if multi:
        iso_in = ", ".join(f"'{c}'" for c in iso_list)
        return f"""SELECT
  f.country_iso3,
  p.product_name,
  f.metric,
  f.year,
  SUM(f.value) AS value,
  f.unit
FROM {mart_table_fqn(table)} f
JOIN {mart_table_fqn('dim_product')} p
  ON f.product_key = p.product_key
WHERE f.country_iso3 IN ({iso_in}){grain_clause}
  AND f.metric IN ({metric_clause}){product_clause}
  AND f.year BETWEEN {y0} AND {y1}
GROUP BY 1, 2, 3, 4, 6
ORDER BY 1, 2, 3
LIMIT 500"""
    iso = iso_list[0]
    return f"""SELECT
  f.country_iso3,
  p.product_name,
  f.metric,
  f.year,
  f.value,
  f.unit
FROM {mart_table_fqn(table)} f
JOIN {mart_table_fqn('dim_product')} p
  ON f.product_key = p.product_key
WHERE f.country_iso3 = '{iso}'{grain_clause}
  AND f.metric IN ({metric_clause}){product_clause}
  AND f.year BETWEEN {y0} AND {y1}
ORDER BY f.year DESC
LIMIT 40"""


def _plan_sql(
    plan: TablePlan,
    *,
    query: str,
    iso_list: list[str],
    products: list[str],
    panel: bool,
    multi: bool,
    y0: int,
    y1: int,
) -> tuple[str | None, dict[str, Any], list[str]]:
    caveats: list[str] = []
    hits: dict[str, Any] = {"country_iso3": iso_list, "product_name": products}

    if plan.table_id == "fct_food_balance":
        if panel or multi:
            metrics = default_fvc_metrics_for_panel()
        else:
            metrics = resolve_metric(query, class_code="FVC", table_id=plan.table_id)
            if not metrics and _SHARE_RE.search(query):
                metrics = [
                    "food_balance_import_quantity",
                    "food_balance_domestic_supply_quantity",
                ]
            if not metrics:
                metrics = [
                    "food_balance_import_quantity",
                    "food_balance_domestic_supply_quantity",
                ]
        hits["metric"] = metrics
        include_source = not panel and not multi
        if include_source:
            hits["source_natural_key"] = complete_enum(plan.table_id, "source_natural_key")
        sql = _build_food_balance_sql(
            iso_list=iso_list,
            products=products,
            metrics=metrics,
            y0=y0,
            y1=y1,
            multi=multi,
            include_source_key=include_source,
        )
        selected = [plan.table_id, "dim_product"]
    elif plan.table_id == "fct_trade":
        metrics = default_fvc_trade_metrics_for_panel() if (panel or multi) else resolve_metric(
            query, class_code="FVC", table_id=plan.table_id
        )
        if not metrics:
            metrics = default_fvc_trade_metrics_for_panel()
        hits["metric"] = metrics
        hits["trade_grain"] = ["faostat_country_year"]
        sql = _build_trade_sql(
            iso_list=iso_list,
            products=products,
            metrics=metrics,
            y0=y0,
            y1=y1,
            multi=multi,
        )
        selected = [plan.table_id, "dim_product"]
    else:
        return None, hits, ["unsupported_fvc_table"]

    ok, reason = validate_engine_sql(
        sql,
        table_id=plan.table_id,
        selected_tables=selected,
        allowed_iso3=iso_list,
    )
    if not ok:
        caveats.append(reason)
        return None, hits, caveats
    return sql, hits, caveats


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
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        bundles = match_intent_bundles(query, facets)
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        panel = is_agri_activities_panel(query, facets, bundles=bundles)
        multi = is_multi_country_panel(iso_list)
        base_hits = bind_value_hits(card, query=query, facets=facets)
        if iso_list:
            base_hits["country_iso3"] = iso_list
        req = build_sql_request_from_facets(
            class_code="FVC",
            table_id=str(card.get("default_table") or "fct_food_balance"),
            query=query,
            facets=facets,
            card=card,
            value_hits=base_hits,
            iso_list=iso_list,
            bundles=bundles,
        )
        y0, y1 = _year_bounds(facets, req_years=(req.year_start, req.year_end))

        if not iso_list:
            return EngineResult(
                class_code="FVC",
                status="planner_error",
                table_id="fct_food_balance",
                sql=None,
                caveats=["no country_iso3 from facets or region expansion"],
                value_hits=base_hits,
            )

        plans = select_table_plans(
            "FVC",
            query=query,
            facets=facets,
            bundles=bundles,
            card=card,
            iso_list=iso_list,
        )
        if not plans:
            return EngineResult(
                class_code="FVC",
                status="planner_error",
                table_id="fct_food_balance",
                sql=None,
                caveats=["no_table_plans"],
                value_hits=base_hits,
            )

        merged_hits = dict(base_hits)
        sql_plans: list[dict[str, Any]] = []
        all_caveats: list[str] = []
        primary_sql: str | None = None
        primary_table = plans[0].table_id

        for plan in plans:
            products = resolve_labels(
                plan.table_id,
                "product_name",
                query,
                scope="fact_distinct",
                geography=geography,
            )
            if not products and panel:
                products = resolve_staple_products(query, facets, bundles=bundles)
            elif not products and "wheat" in query.lower():
                products = resolve_labels(plan.table_id, "product_name", "wheat", scope="fact_distinct")

            sql, plan_hits, caveats = _plan_sql(
                plan,
                query=query,
                iso_list=iso_list,
                products=products,
                panel=panel,
                multi=multi,
                y0=y0,
                y1=y1,
            )
            merged_hits.update(plan_hits)
            all_caveats.extend(caveats)
            if not sql:
                continue
            if primary_sql is None:
                primary_sql = sql
                primary_table = plan.table_id
            sql_plans.append(
                {
                    "table_id": plan.table_id,
                    "sql": sql,
                    "value_hits": plan_hits,
                    "status": "ready",
                    "family_id": plan.family_id,
                    "role": plan.role,
                }
            )

        if not sql_plans:
            return EngineResult(
                class_code="FVC",
                status="planner_error",
                table_id=plans[0].table_id,
                sql=None,
                caveats=all_caveats or ["no_valid_sql"],
                value_hits=merged_hits,
            )

        status = "ready" if len(sql_plans) == len(plans) else "planner_error"
        return EngineResult(
            class_code="FVC",
            status=status,
            table_id=primary_table,
            sql=primary_sql,
            caveats=all_caveats if status != "ready" else [],
            value_hits=merged_hits,
            sql_plans=tuple(sql_plans),
        )


__all__ = ["FvcEngine"]
