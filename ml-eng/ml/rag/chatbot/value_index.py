"""Complete warehouse value index: closed-world enums + resolve_labels (no cap)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    load_mart_table_schema,
    value_samples_for_mart_tables,
)
from ml.rag.chatbot.geo_iso3 import infer_country_iso3_from_query, name_to_iso3

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "value_index"
_SAMPLE_SUFFIX = "_value_samples"
_STATS_SUFFIX = "_value_stats"

_FOOD_BALANCE_SOURCE_KEYS = (
    "Unpivoted_FAOstat_africa_Food_Balances_Food_Balances_1961-2013_old_methodology_and_population",
    "Unpivoted_FAOstat_africa_Food_Balances_Supply_Utilization_Accounts_2010-23",
    "Unpivoted_FAOstat_africa_Food Balances_Food_Balances_2010-23",
    "Unpivoted_FAOstat_africa_Food_Balances_Commodity_Balances_non-food_2010-23",
)
_COUNTRY_ALIASES: dict[str, str] = {
    "ghana": "GHA",
    "kenya": "KEN",
    "nigeria": "NGA",
    "somalia": "SOM",
    "malawi": "MWI",
}


def _bare_table(table_id: str) -> str:
    return (table_id or "").strip().split(".")[-1].lower()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


@lru_cache(maxsize=1)
def _load_index_file() -> dict[str, Any]:
    path = _DEFAULT_DIR / "mart_value_index.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {}


def _yaml_enums(table_id: str, column: str) -> list[str]:
    bare = _bare_table(table_id)
    samples_map = value_samples_for_mart_tables({bare}).get(bare) or {}
    vals = list(samples_map.get(column) or [])
    schema = load_mart_table_schema(bare) or {}
    sample_key = f"{column}{_SAMPLE_SUFFIX}"
    if sample_key in schema and isinstance(schema[sample_key], list):
        for v in schema[sample_key]:
            s = str(v).strip()
            if s and s not in vals:
                vals.append(s)
    stats_key = f"{column}{_STATS_SUFFIX}"
    stats = schema.get(stats_key) if isinstance(schema.get(stats_key), dict) else {}
    if stats.get("is_truncated") is False and vals:
        return vals
    index = _load_index_file()
    key = f"{bare}.{column}"
    indexed = index.get("enums", {}).get(key)
    if isinstance(indexed, list) and indexed:
        return [str(x) for x in indexed]
    return vals


def complete_enum(table_id: str, column: str) -> list[str]:
    """All labels for table.column from index (complete, no cap)."""
    bare = _bare_table(table_id)
    if bare == "fct_food_balance" and column == "source_natural_key":
        return list(_FOOD_BALANCE_SOURCE_KEYS)
    if column == "product_name":
        index = _load_index_file()
        dim_key = "dim_product.product_name"
        dim_labels = index.get("enums", {}).get(dim_key) or _yaml_enums("dim_product", column)
        if bare != "dim_product":
            scoped = index.get("fact_scoped", {}).get(f"{bare}.{column}")
            if isinstance(scoped, list) and scoped:
                return [str(x) for x in scoped]
            if bare == "fct_food_balance":
                return [l for l in dim_labels if "wheat" in _norm(l)] or dim_labels
        return dim_labels
    return list(_yaml_enums(table_id, column))


def numeric_stats(table_id: str, column: str) -> dict[str, Any]:
    bare = _bare_table(table_id)
    schema = load_mart_table_schema(bare) or {}
    stats_key = f"{column}{_STATS_SUFFIX}"
    raw = schema.get(stats_key)
    if isinstance(raw, dict):
        return {
            "dtype": _column_dtype(schema, column),
            "min": raw.get("min_value"),
            "max": raw.get("max_value"),
            "distinct_count": raw.get("distinct_count"),
        }
    return {"dtype": _column_dtype(schema, column), "min": None, "max": None, "distinct_count": None}


def _column_dtype(schema: dict[str, Any], column: str) -> str:
    for col in schema.get("columns") or []:
        if isinstance(col, dict) and str(col.get("name") or "").lower() == column.lower():
            return str(col.get("type") or "STRING")
    return "STRING"


def resolve_country(query: str, *, geography: list[str] | None = None) -> str | None:
    if geography:
        for g in geography:
            iso = name_to_iso3(str(g).strip())
            if iso:
                return iso
    return infer_country_iso3_from_query(query)


def resolve_labels(
    table_id: str,
    column: str,
    query: str,
    *,
    scope: str = "table",
    geography: list[str] | None = None,
) -> list[str]:
    """ALL matching labels for query — no score cutoff."""
    if column == "country_iso3":
        from ml.rag.chatbot.geo_iso3 import resolve_geography_iso3

        iso_list = resolve_geography_iso3(query, geography=geography)
        if iso_list:
            return iso_list
        iso = resolve_country(query, geography=geography)
        return [iso] if iso else []

    labels = complete_enum(table_id, column)
    if column == "product_name" and scope == "fact_distinct":
        labels = _fact_scoped_product_names(table_id, labels)
    q = _norm(query)
    if not q:
        return labels if len(labels) <= 40 else []
    tokens = [t for t in re.split(r"[\s,;/]+", q) if len(t) >= 2]
    matched: list[str] = []
    for label in labels:
        nl = _norm(label)
        if not nl:
            continue
        if nl == q or q in nl or nl in q:
            matched.append(label)
            continue
        if any(t in nl for t in tokens):
            matched.append(label)
    if matched:
        return matched
    if column == "metric" and "import" in q:
        return [m for m in labels if "import" in _norm(m)]
    if column == "metric" and "domestic" in q and "supply" in q:
        return [m for m in labels if "domestic" in _norm(m) and "supply" in _norm(m)]
    return matched


def _fact_scoped_product_names(table_id: str, global_labels: list[str]) -> list[str]:
    index = _load_index_file()
    bare = _bare_table(table_id)
    key = f"{bare}.product_name"
    scoped = index.get("fact_scoped", {}).get(key)
    if isinstance(scoped, list) and scoped:
        return [str(x) for x in scoped]
    wheat_like = [l for l in global_labels if "wheat" in _norm(l)]
    if wheat_like:
        return wheat_like
    return global_labels


def resolve_metric(query: str, *, class_code: str = "FVC", table_id: str = "fct_food_balance") -> list[str]:
    q = _norm(query)
    labels = complete_enum(table_id, "metric")
    if "import" in q and ("share" in q or "domestic" in q or "supply" in q):
        out = [
            m
            for m in labels
            if m in ("food_balance_import_quantity", "food_balance_domestic_supply_quantity")
            or ("import" in _norm(m) and "quantity" in _norm(m))
            or ("domestic" in _norm(m) and "supply" in _norm(m))
        ]
        if out:
            return out
    return resolve_labels(table_id, "metric", query)


def resolve_geography_iso3(
    query: str,
    *,
    geography: list[str] | None = None,
    expanded_regions: list[str] | None = None,
) -> list[str]:
    from ml.rag.chatbot.geo_iso3 import resolve_geography_iso3 as _resolve

    return _resolve(query, geography=geography, expanded_regions=expanded_regions)


def resolve_source_natural_key(table_id: str, query: str = "") -> list[str]:
    return complete_enum(table_id, "source_natural_key")


__all__ = [
    "complete_enum",
    "resolve_labels",
    "resolve_country",
    "resolve_metric",
    "resolve_source_natural_key",
    "numeric_stats",
    "resolve_geography_iso3",
]
