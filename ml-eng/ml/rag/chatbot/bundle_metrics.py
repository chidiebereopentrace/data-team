"""Bundle-default metrics and staple products for class engines."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import match_product_samples
from ml.rag.chatbot.intent_bundles import MatchedBundle, has_bundle, match_intent_bundles

AGRI_ACTIVITIES_STAPLES: tuple[str, ...] = (
    "Maize",
    "Rice",
    "Cassava",
    "Sorghum",
    "Millet",
    "Yam",
)

FVC_AGRI_ACTIVITIES_METRICS: tuple[str, ...] = (
    "food_balance_production",
    "food_balance_import_quantity",
    "food_balance_export_quantity",
    "food_balance_domestic_supply_quantity",
)

FVC_TRADE_PANEL_METRICS: tuple[str, ...] = (
    "trade_import_quantity",
    "trade_export_quantity",
)

_AGRI_PANEL_RE = re.compile(
    r"\b(agricultural\s+activities|agri\s+activities|agri\s+report|country\s+by\s+country)\b",
    re.I,
)

_UNSUPPORTED_PANEL_FACTS = frozenset(
    {
        "fct_soil",
        "fct_biodiversity",
        "fct_research_expenditure",
    }
)


def is_agri_activities_panel(
    query: str,
    facets: dict[str, Any],
    *,
    bundles: tuple[MatchedBundle, ...] | None = None,
) -> bool:
    b = bundles if bundles is not None else match_intent_bundles(query, facets)
    if has_bundle(b, "agricultural_activities"):
        return True
    return bool(_AGRI_PANEL_RE.search(query or ""))


def is_multi_country_panel(iso_list: list[str]) -> bool:
    return len(iso_list) >= 2


def default_fvc_metrics_for_panel() -> list[str]:
    return list(FVC_AGRI_ACTIVITIES_METRICS)


def default_fvc_trade_metrics_for_panel() -> list[str]:
    return list(FVC_TRADE_PANEL_METRICS)


def resolve_staple_products(
    query: str,
    facets: dict[str, Any],
    *,
    bundles: tuple[MatchedBundle, ...] | None = None,
) -> list[str]:
    """Staple IN-list for agri-activities when query has no explicit crop."""
    blob_parts = [query or ""]
    entities = facets.get("entities")
    if isinstance(entities, list):
        blob_parts.extend(str(e) for e in entities if str(e).strip())
    found = match_product_samples("fct_production", " ".join(blob_parts))
    if found:
        return found[:6]
    if is_agri_activities_panel(query, facets, bundles=bundles):
        return list(AGRI_ACTIVITIES_STAPLES)
    return []


def unsupported_grain_for_panel(table_id: str, *, iso_count: int) -> bool:
    bare = (table_id or "").split(".")[-1].lower()
    if bare in _UNSUPPORTED_PANEL_FACTS:
        return True
    if bare == "fct_production" and iso_count >= 2:
        return True
    return False


__all__ = [
    "AGRI_ACTIVITIES_STAPLES",
    "FVC_AGRI_ACTIVITIES_METRICS",
    "FVC_TRADE_PANEL_METRICS",
    "is_agri_activities_panel",
    "is_multi_country_panel",
    "default_fvc_metrics_for_panel",
    "default_fvc_trade_metrics_for_panel",
    "resolve_staple_products",
    "unsupported_grain_for_panel",
]
