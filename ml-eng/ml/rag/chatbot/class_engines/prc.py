"""PRC engine: prices and markets (FEWS/WFP/FAOSTAT grains)."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bundle_metrics import resolve_staple_products
from ml.rag.chatbot.class_engines.base import ClassEngine, EngineResult
from ml.rag.chatbot.class_engines.shared import bind_value_hits, mart_table_fqn, validate_engine_sql
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_compiler import compile_sql, sql_compiler_enabled
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.value_index import resolve_geography_iso3, resolve_labels

_MARKET_DETAIL_RE = re.compile(r"\b(market|bamako|kano|nairobi|retail|wholesale|urban)\b", re.I)


def _year_bounds(facets: dict[str, Any]) -> tuple[int, int]:
    ts = str(facets.get("time_start") or "")[:4]
    te = str(facets.get("time_end") or "")[:4]
    try:
        return int(ts), int(te)
    except ValueError:
        return 2018, 2024


def _product_clause(products: list[str]) -> str:
    if not products:
        return ""
    labels: list[str] = []
    for p in products:
        hits = resolve_labels("fct_prices", "product_name", p, scope="fact_distinct")
        labels.extend(hits[:2] if hits else [p])
    uniq = list(dict.fromkeys(labels))[:6]
    if not uniq:
        return ""
    in_list = ", ".join(f"'{x.replace(chr(39), '')}'" for x in uniq)
    return f"\n  AND product_name IN ({in_list})"


def _build_agg_prices_sql(
    *,
    iso_list: list[str],
    products: list[str],
    y0: int,
    y1: int,
) -> str:
    iso = iso_list[0]
    prod = _product_clause(products)
    return f"""SELECT
  country_iso3,
  year,
  month,
  product_name,
  price_type,
  value,
  unit,
  source_key
FROM {mart_table_fqn("agg_prices_country_month")}
WHERE country_iso3 = '{iso}'{prod}
  AND year BETWEEN {y0} AND {y1}
  AND month BETWEEN 1 AND 12
ORDER BY year DESC, month DESC
LIMIT 48"""


def _build_fct_prices_sql(
    *,
    iso_list: list[str],
    products: list[str],
    y0: int,
    y1: int,
) -> str:
    iso = iso_list[0]
    prod = ""
    if products:
        labels: list[str] = []
        for p in products:
            hits = resolve_labels("fct_prices", "product_name", p, scope="fact_distinct")
            labels.extend(hits[:2] if hits else [p])
        uniq = list(dict.fromkeys(labels))[:6]
        if uniq:
            in_list = ", ".join(f"'{x.replace(chr(39), '')}'" for x in uniq)
            prod = f"\n  AND p.product_name IN ({in_list})"
    return f"""SELECT
  g.country_iso3,
  p.product_name,
  f.year,
  f.month,
  f.price_type,
  f.value,
  f.unit,
  f.source_key
FROM {mart_table_fqn("fct_prices")} f
JOIN {mart_table_fqn("dim_geography")} g ON f.geography_key = g.geography_key
JOIN {mart_table_fqn("dim_product")} p ON f.product_key = p.product_key
WHERE g.country_iso3 = '{iso}'{prod}
  AND f.year BETWEEN {y0} AND {y1}
  AND (f.month IS NULL OR f.month BETWEEN 1 AND 12)
ORDER BY f.year DESC, f.month DESC
LIMIT 48"""


class PrcEngine(ClassEngine):
    class_code = "PRC"

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any] | None = None,
    ) -> EngineResult:
        card = card or load_schema_card("PRC") or {}
        bundles = match_intent_bundles(query, facets)
        geography = facets.get("geography") if isinstance(facets.get("geography"), list) else []
        expanded = facets.get("expanded_regions") if isinstance(facets.get("expanded_regions"), list) else None
        iso_list = resolve_geography_iso3(query, geography=geography, expanded_regions=expanded)
        if not iso_list:
            return EngineResult(
                class_code="PRC",
                status="planner_error",
                table_id="",
                sql=None,
                caveats=["missing_geography"],
            )

        products = resolve_staple_products(query, facets)
        y0, y1 = _year_bounds(facets)
        use_market = bool(_MARKET_DETAIL_RE.search(query)) and len(iso_list) == 1
        table = "fct_prices" if use_market else "agg_prices_country_month"

        hits = bind_value_hits(card, query=query, facets=facets)
        hits["country_iso3"] = iso_list
        if products:
            hits["product_name"] = products

        if sql_compiler_enabled():
            req = build_sql_request_from_facets(
                class_code="PRC",
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
                    _build_fct_prices_sql(iso_list=iso_list, products=products, y0=y0, y1=y1)
                    if use_market
                    else _build_agg_prices_sql(iso_list=iso_list, products=products, y0=y0, y1=y1)
                )
                ok, reason = validate_engine_sql(
                    sql,
                    table_id=table,
                    selected_tables=[table],
                    allowed_iso3=iso_list,
                )
        else:
            sql = (
                _build_fct_prices_sql(iso_list=iso_list, products=products, y0=y0, y1=y1)
                if use_market
                else _build_agg_prices_sql(iso_list=iso_list, products=products, y0=y0, y1=y1)
            )
            ok, reason = validate_engine_sql(
                sql,
                table_id=table,
                selected_tables=[table],
                allowed_iso3=iso_list,
            )

        return EngineResult(
            class_code="PRC",
            status="ready" if ok else "planner_error",
            table_id=table,
            sql=sql if ok else None,
            caveats=[] if ok else [reason],
            value_hits=hits,
        )


__all__ = ["PrcEngine"]
