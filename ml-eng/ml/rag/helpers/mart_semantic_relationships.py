"""Semantic relationship map for mart_dev BQ table YAMLs."""
from __future__ import annotations

from typing import Any


def _join(table: str, on: list[str], how: str, note: str) -> dict[str, Any]:
    return {"table": table, "on": on, "how": how, "note": note}


def _comp(table: str, when: str, role: str = "enrich_context") -> dict[str, Any]:
    return {"table": table, "when": when, "role": role}


def _avoid(table: str, reason: str) -> dict[str, Any]:
    return {"table": table, "reason": reason}


def _rels(
    joins: list[dict[str, Any]] | None = None,
    companions: list[dict[str, Any]] | None = None,
    do_not: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "joins_with": joins or [],
        "companions": companions or [],
        "do_not_join": do_not or [],
    }


# Standard dim join keys used across mart facts.
_DIM_GEO_JOIN = _join("dim_geography", ["geography_key"], "LEFT", "Country/admin labels via geography_key")
_DIM_PRODUCT_JOIN = _join("dim_product", ["product_key"], "LEFT", "Product labels via product_key")
_DIM_SOURCE_JOIN = _join("dim_source", ["source_key"], "LEFT", "Source lineage via source_key")
_DIM_INDICATOR_JOIN = _join("dim_indicator", ["indicator_key"], "LEFT", "Named indicator series via indicator_key")
_DIM_MARKET_JOIN = _join("dim_market", ["market_key"], "LEFT", "Market labels via market_key")
_DIM_ITEM_JOIN = _join("dim_item", ["item_key"], "LEFT", "Item labels via item_key")
_DIM_ELEMENT_JOIN = _join("dim_element", ["element_key"], "LEFT", "Element labels via element_key")
_DIM_UNIT_JOIN = _join("dim_unit", ["unit_key"], "LEFT", "Unit labels via unit_key")
_DIM_LAND_USE_JOIN = _join("dim_land_use", ["land_use_key"], "LEFT", "Land-use class/code via land_use_key")
_DIM_PEST_JOIN = _join("dim_pest", ["pest_key"], "LEFT", "Pest labels via pest_key")
_DIM_SEASON_JOIN = _join("dim_season", ["season_key"], "LEFT", "Season labels via season_key")
_DIM_CLASSIFICATION_JOIN = _join(
    "dim_classification", ["classification_key"], "LEFT", "IPC phase labels via classification_key"
)
_DIM_SCENARIO_JOIN = _join("dim_scenario", ["scenario_key"], "LEFT", "IPC scenario labels via scenario_key")
_DIM_LIVESTOCK_JOIN = _join("dim_livestock", ["livestock_key"], "LEFT", "Species labels via livestock_key")
_DIM_DISEASE_JOIN = _join("dim_disease", ["disease_key"], "LEFT", "Hazard/disease labels via disease_key")
_DIM_HOUSEHOLD_JOIN = _join("dim_household", ["household_key"], "LEFT", "Household attributes via household_key")
_DIM_ORGANISATION_JOIN = _join(
    "dim_organisation", ["organisation_key"], "LEFT", "Organisation labels via organisation_key"
)
_DIM_SEX_JOIN = _join("dim_sex", ["sex_key"], "LEFT", "Sex labels via sex_key")
_DIM_SOIL_PROPERTY_JOIN = _join(
    "dim_soil_property", ["soil_property_key"], "LEFT", "Soil property labels via soil_property_key"
)
_DIM_PERSON_JOIN = _join("dim_person", ["person_key"], "LEFT", "Person labels via person_key")

SEMANTIC_RELATIONSHIPS: dict[str, dict[str, Any]] = {
    "fct_production": _rels(
        joins=[
            _DIM_GEO_JOIN,
            _DIM_PRODUCT_JOIN,
            _DIM_ELEMENT_JOIN,
            _DIM_SEASON_JOIN,
            _DIM_SOURCE_JOIN,
            _join(
                "fct_yield",
                ["country_iso3", "product_key", "year≈harvest_year"],
                "compare",
                "National vs subnational — do not UNION",
            ),
        ],
        companions=[_comp("agg_production_annual", "national annual rollups")],
        do_not=[_avoid("fct_yield", "Different grain — never UNION with fct_production")],
    ),
    "fct_yield": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_SEASON_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_climate", "climate drivers of yield")],
        do_not=[_avoid("fct_production", "FAOSTAT national production — compare only")],
    ),
    "fct_prices": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_MARKET_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("agg_prices_country_month", "country monthly rollups")],
        do_not=[_avoid("fct_production", "prices vs production — separate queries unless analytical")],
    ),
    "fct_trade": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_food_balance", "trade in food balance context")],
        do_not=[_avoid("fct_production", "filter trade_grain before comparing to production")],
    ),
    "fct_food_security": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_CLASSIFICATION_JOIN],
        companions=[_comp("agg_food_security_monthly", "country monthly IPC rollups")],
        do_not=[_avoid("fct_household", "IPC vs household FIES — different grains")],
    ),
    "fct_food_balance": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_production", "supply vs production")],
    ),
    "fct_climate": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_INDICATOR_JOIN],
        companions=[_comp("fct_yield", "climate vs yield stress")],
    ),
    "fct_employment": _rels(
        joins=[
            _DIM_GEO_JOIN,
            _DIM_ITEM_JOIN,
            _DIM_ELEMENT_JOIN,
            _DIM_INDICATOR_JOIN,
            _DIM_SEX_JOIN,
            _DIM_SOURCE_JOIN,
        ],
        companions=[_comp("fct_economics", "macro context")],
    ),
    "fct_economics": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_hdi", "development context")],
    ),
    "fct_hdi": _rels(
        joins=[_DIM_GEO_JOIN],
        companions=[_comp("fct_economics", "development context"), _comp("agg_hdi_latest", "latest HDI snapshot")],
    ),
    "fct_household": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_HOUSEHOLD_JOIN],
        do_not=[_avoid("fct_gender_inclusion", "survey vs SDG official")],
    ),
    "fct_gender_inclusion": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN],
        do_not=[_avoid("fct_household", "SDG vs survey control scores")],
    ),
    "fct_emissions": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_production", "emissions intensity vs production")],
    ),
    "fct_animal_health": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_LIVESTOCK_JOIN, _DIM_HOUSEHOLD_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_vegetation": _rels(joins=[_DIM_GEO_JOIN]),
    "fct_biodiversity": _rels(joins=[_DIM_GEO_JOIN]),
    "fct_protected_areas": _rels(joins=[_DIM_GEO_JOIN, _DIM_SOURCE_JOIN]),
    "fct_germplasm": _rels(joins=[_DIM_GEO_JOIN, _DIM_SOURCE_JOIN]),
    "fct_humanitarian": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_forestry": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_fertilizer": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_machinery": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_pesticide": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_PEST_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_land_inputs": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_fertilizer", "legacy split table"), _comp("fct_pesticide", "legacy split table")],
        do_not=[_avoid("fct_fertilizer", "Prefer fct_land_inputs with input_grain filter when unified")],
    ),
    "fct_food_hazards": _rels(joins=[_DIM_GEO_JOIN, _DIM_DISEASE_JOIN, _DIM_SOURCE_JOIN]),
    "fct_insurance": _rels(joins=[_DIM_GEO_JOIN, _DIM_HOUSEHOLD_JOIN, _DIM_SOURCE_JOIN]),
    "fct_air_quality": _rels(joins=[_DIM_GEO_JOIN, _DIM_INDICATOR_JOIN, _DIM_SOURCE_JOIN]),
    "fct_land_use": _rels(joins=[_DIM_GEO_JOIN, _DIM_LAND_USE_JOIN, _DIM_SOURCE_JOIN]),
    "fct_market_access": _rels(joins=[_DIM_GEO_JOIN, _DIM_MARKET_JOIN, _DIM_SOURCE_JOIN]),
    "fct_population_dynamics": _rels(joins=[_DIM_GEO_JOIN, _DIM_INDICATOR_JOIN]),
    "fct_technology": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ORGANISATION_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_value_chain": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN]
    ),
    "fct_soil_health": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_SOIL_PROPERTY_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("agg_soil_admin", "admin rollups"), _comp("fct_land_inputs", "input context")],
    ),
    "fct_land_degradation": _rels(joins=[_DIM_GEO_JOIN, _DIM_UNIT_JOIN, _DIM_SOURCE_JOIN]),
    "fct_investment": _rels(
        joins=[
            _DIM_GEO_JOIN,
            _DIM_ITEM_JOIN,
            _DIM_ELEMENT_JOIN,
            _DIM_INDICATOR_JOIN,
            _DIM_SEX_JOIN,
            _DIM_ORGANISATION_JOIN,
            _DIM_SOURCE_JOIN,
        ]
    ),
    "fct_researchers": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ORGANISATION_JOIN, _DIM_PERSON_JOIN, _DIM_UNIT_JOIN]
    ),
    "fct_research_expenditure": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ORGANISATION_JOIN, _DIM_UNIT_JOIN]
    ),
    "agg_production_annual": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_production", "detail rows")],
    ),
    "agg_production_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_production", "detail rows"), _comp("agg_production_annual", "annual rollups")],
    ),
    "agg_production_country_season": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_SEASON_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_production", "FAOSTAT detail"), _comp("agg_production_country_year", "annual panel")],
    ),
    "agg_hdi_latest": _rels(
        joins=[_DIM_GEO_JOIN],
        companions=[_comp("fct_hdi", "time series detail")],
    ),
    "agg_soil_admin": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_soil_health", "plot-level detail")],
    ),
    "agg_food_security_monthly": _rels(
        joins=[
            _DIM_GEO_JOIN,
            _DIM_CLASSIFICATION_JOIN,
            _DIM_SCENARIO_JOIN,
            _join("dim_time", ["time_key"], "LEFT", "Calendar year for monthly observation keys"),
        ],
    ),
    "agg_food_security_country_month": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("agg_food_security_monthly", "national monthly rollups")],
    ),
    "agg_prices_country_month": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN],
        companions=[_comp("fct_prices", "detail rows")],
    ),
    "agg_emissions_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_ELEMENT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_emissions", "detail rows")],
    ),
    "agg_economics_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_economics", "detail rows")],
    ),
    "agg_forestry_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_ITEM_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_forestry", "detail rows")],
    ),
    "agg_food_balance_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_PRODUCT_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_food_balance", "detail rows")],
    ),
    "agg_employment_country_year": _rels(
        joins=[_DIM_GEO_JOIN, _DIM_SOURCE_JOIN],
        companions=[_comp("fct_employment", "detail rows")],
    ),
    "dim_geography": _rels(companions=[_comp("dim_geography", "join target for country_iso3 labels")]),
    "dim_product": _rels(),
    "dim_indicator": _rels(
        companions=[_comp("fct_employment", "employment indicators"), _comp("fct_climate", "climate series")]
    ),
    "dim_source": _rels(),
    "dim_market": _rels(),
}


def relationships_for(table_id: str) -> dict[str, Any]:
    bare = (table_id or "").strip().split(".")[-1].lower()
    rel = SEMANTIC_RELATIONSHIPS.get(bare)
    if rel:
        return rel
    return {"joins_with": [], "companions": [], "do_not_join": []}


def compact_rels_summary(table_id: str) -> str:
    rel = relationships_for(table_id)
    joins = [str(j.get("table") or "") for j in rel.get("joins_with") or [] if isinstance(j, dict)]
    comps = [str(c.get("table") or "") for c in rel.get("companions") or [] if isinstance(c, dict)]
    bits: list[str] = []
    if joins:
        bits.append("joins=" + ",".join(joins[:4]))
    if comps:
        bits.append("companions=" + ",".join(comps[:3]))
    return "; ".join(bits) if bits else "rels=none"


def _bare_table(table_id: str) -> str:
    return (table_id or "").strip().split(".")[-1].lower()


def _is_mart_table(table_id: str) -> bool:
    bare = _bare_table(table_id)
    return bare.startswith(("fct_", "agg_", "dim_", "bridge_"))


def documented_join_pairs(
    selected_tables: list[str] | set[str] | None,
) -> list[dict[str, Any]]:
    selected = {_bare_table(t) for t in (selected_tables or []) if _is_mart_table(t)}
    if len(selected) < 2:
        return []
    pairs: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for left in sorted(selected):
        rel = SEMANTIC_RELATIONSHIPS.get(left) or {}
        for join in rel.get("joins_with") or []:
            if not isinstance(join, dict):
                continue
            right = _bare_table(str(join.get("table") or ""))
            if right not in selected or right == left:
                continue
            on_keys = [
                str(k).strip()
                for k in (join.get("on") or [])
                if isinstance(k, (str, int, float)) and str(k).strip()
            ]
            if not on_keys:
                continue
            key = frozenset({left, right})
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "on": on_keys,
                    "how": str(join.get("how") or "LEFT").strip(),
                    "note": str(join.get("note") or "").strip(),
                }
            )
    return pairs


def format_join_fragments_for_nl2sql(
    selected_tables: list[str] | set[str] | None,
) -> str:
    selected = sorted({_bare_table(t) for t in (selected_tables or []) if _is_mart_table(t)})
    if len(selected) < 2:
        return ""
    pairs = documented_join_pairs(selected)
    if not pairs:
        return (
            "JOIN fragments: use standard mart dims when filtering by name: "
            "LEFT JOIN dim_geography g ON f.geography_key = g.geography_key for country names; "
            "LEFT JOIN dim_product p ON f.product_key = p.product_key for product names; "
            "LEFT JOIN dim_indicator i ON f.indicator_key = i.indicator_key for indicator names. "
            "Filter country_iso3 directly when ISO3 is known."
        )
    lines = ["JOIN fragments (required if multi-table):"]
    for pair in pairs:
        on_parts = []
        for on in pair["on"]:
            if "≈" in on:
                on_parts.append(on.replace("≈", " align "))
            elif on.endswith("_key"):
                on_parts.append(f"{pair['left']}.{on} = {pair['right']}.{on}")
            else:
                on_parts.append(on)
        on_sql = " AND ".join(on_parts)
        how = pair.get("how") or "LEFT"
        bit = f"  {pair['left']} {how} JOIN {pair['right']} ON {on_sql}"
        if pair.get("note"):
            bit += f"  -- {pair['note']}"
        lines.append(bit)
    return "\n".join(lines)


__all__ = [
    "SEMANTIC_RELATIONSHIPS",
    "compact_rels_summary",
    "documented_join_pairs",
    "format_join_fragments_for_nl2sql",
    "relationships_for",
]
