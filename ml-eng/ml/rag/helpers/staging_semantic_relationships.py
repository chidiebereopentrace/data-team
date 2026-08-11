"""Semantic relationship map for staging_dev BQ table YAMLs."""
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


# Domain-consistent multi-table guidance for the SQL reasoner.
SEMANTIC_RELATIONSHIPS: dict[str, dict[str, Any]] = {
    "stg_yield_raw_data": _rels(
        joins=[
            _join("stg_faostat_production", ["country", "product", "year≈harvest_year"], "compare", "National FAOSTAT production vs subnational yields"),
            _join("stg_fews_food_security", ["country", "fnid", "year"], "enrich", "Link productivity stress to IPC phases"),
            _join("stg_fews_market_prices", ["country", "product≈product_name", "year"], "enrich", "Yield outcomes vs local staple prices"),
            _join("stg_wfp_vampire_prices", ["country", "product≈product_name", "year"], "enrich", "Yield vs retail/market prices"),
        ],
        companions=[
            _comp("stg_nasa_power", "climate drivers of yield anomalies"),
            _comp("stg_isric_africa_soil", "soil constraints near yield geographies"),
        ],
        do_not=[
            _avoid("stg_openaire_projects", "research metadata; no shared agronomic grain"),
            _avoid("stg_nakuru_air_quality", "local air quality; not crop yield grain"),
        ],
    ),
    "stg_fews_food_security": _rels(
        joins=[
            _join("stg_yield_raw_data", ["country", "fnid", "year"], "enrich", "Food security vs production stress"),
            _join("stg_fews_market_prices", ["country", "year", "month"], "enrich", "IPC with contemporaneous prices"),
            _join("stg_wfp_vampire_prices", ["country", "year", "month"], "enrich", "IPC with WFP price series"),
            _join("stg_faostat_food_balances", ["country_name≈country", "year"], "compare", "National food availability context"),
        ],
        companions=[_comp("stg_climatewatch_health", "climate-health stress overlaps")],
        do_not=[_avoid("stg_openaire_persons", "no shared geo-time grain")],
    ),
    "stg_fews_market_prices": _rels(
        joins=[
            _join("stg_wfp_vampire_prices", ["country", "product_name", "year", "month"], "compare", "Align units/currency before comparing"),
            _join("stg_faostat_prices", ["country_name≈country", "product_name", "year"], "compare", "Producer/CPI vs FEWS market prices"),
            _join("stg_fews_food_security", ["country", "year", "month"], "enrich", "Prices with IPC context"),
            _join("stg_yield_raw_data", ["country", "product≈product_name", "year"], "enrich", "Prices with production outcomes"),
        ],
        companions=[_comp("stg_fews_cross_border_trade", "cross-border flow pressure on markets")],
        do_not=[_avoid("stg_isric_africa_soil", "soil grids; no market grain")],
    ),
    "stg_fews_cross_border_trade": _rels(
        joins=[
            _join("stg_fews_market_prices", ["country", "product_name", "year", "month"], "enrich", "Trade flows vs market prices"),
            _join("stg_faostat_trade", ["country_name≈country", "product_name", "year"], "compare", "FEWS border flows vs FAOSTAT trade"),
        ],
        companions=[_comp("stg_wfp_vampire_prices", "price shocks along corridors")],
        do_not=[_avoid("stg_ilri_dairy_genetics", "farm microdata; no trade grain")],
    ),
    "stg_faostat_production": _rels(
        joins=[
            _join("stg_yield_raw_data", ["country≈country_name", "product≈product_name", "year≈harvest_year"], "compare", "National vs subnational production"),
            _join("stg_faostat_trade", ["country_name", "product_name", "year"], "enrich", "Production with trade balance"),
            _join("stg_faostat_food_balances", ["country_name", "product_name", "year"], "enrich", "Production vs food supply accounts"),
            _join("stg_faostat_prices", ["country_name", "product_name", "year"], "enrich", "Production with price signals"),
        ],
        companions=[_comp("stg_africa_gdp_ppp", "macro context for production trends")],
        do_not=[_avoid("stg_vegetation_ndvi", "raster grids; prefer climate/soil companions instead")],
    ),
    "stg_faostat_emissions": _rels(
        joins=[
            _join("stg_faostat_production", ["country_name", "year"], "enrich", "Emissions intensity vs production"),
            _join("stg_faostat_land_inputs", ["country_name", "year"], "enrich", "Inputs/land-use drivers of emissions"),
        ],
        companions=[_comp("stg_climatewatch_health", "climate impact framing")],
        do_not=[_avoid("stg_wfp_vampire_prices", "prices; no emissions grain")],
    ),
    "stg_faostat_prices": _rels(
        joins=[
            _join("stg_fews_market_prices", ["country≈country_name", "product_name", "year"], "compare", "Watch currency/unit differences"),
            _join("stg_wfp_vampire_prices", ["country≈country_name", "product_name", "year"], "compare", "National vs market retail series"),
            _join("stg_faostat_production", ["country_name", "product_name", "year"], "enrich", "Prices with production"),
        ],
        companions=[_comp("stg_africa_gdp_ppp", "real income context")],
        do_not=[_avoid("stg_isda_soil_enriched", "soil long-format; no price grain")],
    ),
    "stg_faostat_trade": _rels(
        joins=[
            _join("stg_faostat_production", ["country_name", "product_name", "year"], "enrich", "Trade vs domestic production"),
            _join("stg_fews_cross_border_trade", ["country≈country_name", "product_name", "year"], "compare", "FAOSTAT vs FEWS corridor flows"),
            _join("stg_faostat_food_balances", ["country_name", "product_name", "year"], "enrich", "Trade in food balance context"),
        ],
        companions=[_comp("stg_faostat_prices", "unit value / price context")],
        do_not=[_avoid("stg_nakuru_air_quality", "local sensors only")],
    ),
    "stg_faostat_food_balances": _rels(
        joins=[
            _join("stg_faostat_production", ["country_name", "product_name", "year"], "enrich", "Supply accounts vs production"),
            _join("stg_faostat_trade", ["country_name", "product_name", "year"], "enrich", "Import/export in balances"),
            _join("stg_fews_food_security", ["country≈country_name", "year"], "compare", "National availability vs IPC"),
        ],
        companions=[_comp("stg_faostat_sdg_hdi", "diet cost / SDG nutrition indicators")],
        do_not=[_avoid("stg_germplasm", "accession geography; not food balances")],
    ),
    "stg_faostat_investment_asti": _rels(
        joins=[
            _join("stg_faostat_production", ["country_name", "year"], "enrich", "R&D/investment vs production outcomes"),
            _join("stg_africa_hdi", ["country≈country_name", "year"], "enrich", "Development context"),
        ],
        companions=[_comp("stg_openaire_projects", "project funding narratives (qualitative)")],
        do_not=[_avoid("stg_fews_market_prices", "market prices; different decision use")],
    ),
    "stg_faostat_land_inputs": _rels(
        joins=[
            _join("stg_faostat_production", ["country_name", "year"], "enrich", "Inputs vs production"),
            _join("stg_faostat_emissions", ["country_name", "year"], "enrich", "Fertilizer/land-use vs emissions"),
            _join("stg_isric_africa_soil", ["country via spatial"], "enrich", "Soil constraints (spatial join caution)"),
        ],
        companions=[_comp("stg_unccd_land_degradation", "degradation pressure")],
        do_not=[_avoid("stg_ilri_vendor_consumer", "survey microdata")],
    ),
    "stg_faostat_population_employment": _rels(
        joins=[
            _join("stg_africa_hdi", ["country≈country_name", "year"], "enrich", "Employment with HDI"),
            _join("stg_africa_gdp_ppp", ["country_code", "year≈observation_year"], "enrich", "Labor with income"),
            _join("stg_faostat_production", ["country_name", "year"], "enrich", "Ag employment vs production"),
        ],
        companions=[_comp("stg_faostat_sdg_hdi", "SDG labor/nutrition indicators")],
        do_not=[_avoid("stg_copernicus_era5", "reanalysis grids; no employment grain")],
    ),
    "stg_faostat_sdg_hdi": _rels(
        joins=[
            _join("stg_africa_hdi", ["country≈country_name", "year"], "compare", "FAOSTAT SDG/HDI-related vs Africa HDI series"),
            _join("stg_faostat_food_balances", ["country_name", "year"], "enrich", "Diet affordability with supply"),
            _join("stg_faostat_population_employment", ["country_name", "year"], "enrich", "SDG with employment"),
        ],
        companions=[_comp("stg_climatewatch_health", "health/climate SDG angles")],
        do_not=[_avoid("stg_biodiversity", "species occurrences; not SDG diet metrics")],
    ),
    "stg_ilri_dairy_genetics": _rels(
        joins=[
            _join("stg_ilri_household_food_security", ["country", "household_id"], "enrich", "Dairy outcomes with HH food security when IDs align"),
            _join("stg_ilri_animal_health", ["household_id", "species"], "enrich", "Genetics with health stress"),
        ],
        companions=[_comp("stg_ilri_vegetation_feed", "feed availability context")],
        do_not=[_avoid("stg_faostat_trade", "national trade; not farm genetics")],
    ),
    "stg_ilri_household_food_security": _rels(
        joins=[
            _join("stg_fews_food_security", ["country", "year via survey timing"], "compare", "Survey FIES vs FEWS IPC (interpret carefully)"),
            _join("stg_ilri_animal_health", ["household_id", "country"], "enrich", "HH welfare with livestock health"),
            _join("stg_ilri_vendor_consumer", ["country"], "enrich", "Food environment context"),
        ],
        companions=[_comp("stg_ilri_food_hazards", "food safety risk backdrop")],
        do_not=[_avoid("stg_protected_areas", "conservation polygons; not HH surveys")],
    ),
    "stg_ilri_animal_health": _rels(
        joins=[
            _join("stg_ilri_dairy_genetics", ["household_id"], "enrich", "Health with milk/calving records"),
            _join("stg_ilri_i4i_livestock_insurance", ["household_id", "country"], "enrich", "Health risk with insurance uptake"),
            _join("stg_ilri_vegetation_feed", ["household_id"], "enrich", "Health with feed/vegetation"),
        ],
        companions=[_comp("stg_ilri_household_food_security", "livelihood outcomes")],
        do_not=[_avoid("stg_faostat_emissions", "national emissions; not clinical scores")],
    ),
    "stg_ilri_vegetation_feed": _rels(
        joins=[
            _join("stg_ilri_animal_health", ["household_id"], "enrich", "Feed with animal health"),
            _join("stg_vegetation_ndvi", ["spatial proximity"], "compare", "Survey feed vs NDVI (spatial caution)"),
        ],
        companions=[_comp("stg_nasa_power", "climate stress on forage")],
        do_not=[_avoid("stg_faostat_prices", "national prices; not plot vegetation")],
    ),
    "stg_ilri_i4i_livestock_insurance": _rels(
        joins=[
            _join("stg_ilri_animal_health", ["household_id", "country"], "enrich", "Insurance with disease/mortality"),
            _join("stg_ilri_household_food_security", ["household_id", "country"], "enrich", "Insurance with HH food security"),
        ],
        companions=[_comp("stg_fews_food_security", "regional shock context")],
        do_not=[_avoid("stg_openaire_data_sources", "research infra metadata")],
    ),
    "stg_ilri_vendor_consumer": _rels(
        joins=[
            _join("stg_ilri_food_hazards", ["country"], "enrich", "Market actors with hazard prevalence"),
            _join("stg_ilri_household_food_security", ["country"], "enrich", "Demand-side food security"),
        ],
        companions=[_comp("stg_wfp_vampire_prices", "price environment")],
        do_not=[_avoid("stg_isric_africa_soil", "soil grids")],
    ),
    "stg_ilri_food_hazards": _rels(
        joins=[
            _join("stg_ilri_vendor_consumer", ["country"], "enrich", "Hazards with vendor/consumer surveys"),
        ],
        companions=[_comp("stg_faostat_food_balances", "national food supply backdrop")],
        do_not=[_avoid("stg_germplasm", "crop accessions; not hazards meta-analysis")],
    ),
    "stg_ilri_other_surveys": _rels(
        companions=[
            _comp("stg_ilri_household_food_security", "primary HH food security when survey_type overlaps"),
            _comp("stg_ilri_dairy_genetics", "dairy-related survey_type only"),
        ],
        do_not=[_avoid("stg_faostat_production", "national production; not sparse survey packs")],
    ),
    "stg_nasa_power": _rels(
        joins=[
            _join("stg_yield_raw_data", ["country_code≈country_code", "spatial+time"], "enrich", "Solar/climate drivers of yield"),
            _join("stg_copernicus_era5", ["latitude", "longitude", "time"], "compare", "NASA POWER vs ERA5 (align timestamps)"),
        ],
        companions=[
            _comp("stg_isric_africa_soil", "soil+climate constraints"),
            _comp("stg_vegetation_ndvi", "greenness response"),
        ],
        do_not=[_avoid("stg_openaire_organisations", "org registry; no climate grain")],
    ),
    "stg_copernicus_era5": _rels(
        joins=[
            _join("stg_nasa_power", ["latitude", "longitude", "time"], "compare", "Reanalysis vs POWER"),
            _join("stg_yield_raw_data", ["spatial+season"], "enrich", "Temperature stress on yields"),
        ],
        companions=[_comp("stg_climatewatch_health", "health impact scenarios")],
        do_not=[_avoid("stg_wfp_vampire_prices", "market prices; different use")],
    ),
    "stg_climatewatch_health": _rels(
        joins=[
            _join("stg_africa_hdi", ["country_code", "year"], "enrich", "Health impacts with development"),
            _join("stg_faostat_sdg_hdi", ["country_name", "year"], "enrich", "SDG health/nutrition angles"),
        ],
        companions=[_comp("stg_fews_food_security", "food security co-stress")],
        do_not=[_avoid("stg_cifor_icraf", "plot nutrients; not national health scenarios")],
    ),
    "stg_isric_africa_soil": _rels(
        joins=[
            _join("stg_isda_soil_enriched", ["latitude≈latitude", "longitude≈longitude"], "compare", "Prefer one primary soil source per question"),
            _join("stg_yield_raw_data", ["spatial proximity to fnid"], "enrich", "Soil constraints on yield (spatial join)"),
        ],
        companions=[
            _comp("stg_nasa_power", "climate+soil"),
            _comp("stg_s4a_field_surveys", "field validation plots"),
        ],
        do_not=[_avoid("stg_faostat_trade", "national trade aggregates")],
    ),
    "stg_isda_soil_enriched": _rels(
        joins=[
            _join("stg_isric_africa_soil", ["latitude", "longitude"], "compare", "Long vs wide soil representations — pick one"),
            _join("stg_yield_raw_data", ["spatial proximity"], "enrich", "Property/depth filters for crop stress"),
        ],
        companions=[_comp("stg_cifor_icraf", "plot nutrient samples")],
        do_not=[_avoid("stg_fews_cross_border_trade", "trade flows")],
    ),
    "stg_unccd_land_degradation": _rels(
        joins=[
            _join("stg_faostat_land_inputs", ["geo_area≈country", "year≈time"], "enrich", "Degradation with land/input pressure"),
            _join("stg_africa_hdi", ["country≈geo_area_name", "year"], "enrich", "Degradation with development"),
        ],
        companions=[_comp("stg_isric_africa_soil", "soil property backdrop")],
        do_not=[_avoid("stg_ilri_dairy_genetics", "farm genetics")],
    ),
    "stg_s4a_field_surveys": _rels(
        joins=[
            _join("stg_isric_africa_soil", ["latitude", "longitude"], "validate", "Field plots vs ISRIC grids"),
            _join("stg_isda_soil_enriched", ["latitude", "longitude"], "validate", "Field plots vs iSDA"),
        ],
        companions=[_comp("stg_cifor_icraf", "other plot nutrient surveys")],
        do_not=[_avoid("stg_faostat_prices", "national prices")],
    ),
    "stg_cifor_icraf": _rels(
        joins=[
            _join("stg_s4a_field_surveys", ["plot_id≈plot_code"], "compare", "Plot nutrient protocols differ"),
            _join("stg_isda_soil_enriched", ["spatial proximity"], "enrich", "Plot nutrients vs regional iSDA"),
        ],
        companions=[_comp("stg_isric_africa_soil", "regional soil context")],
        do_not=[_avoid("stg_wfp_vampire_prices", "market prices")],
    ),
    "stg_protected_areas": _rels(
        joins=[
            _join("stg_biodiversity", ["spatial overlap"], "enrich", "Protection with species richness/rarity"),
            _join("stg_vegetation_ndvi", ["spatial overlap"], "enrich", "Protection with vegetation condition"),
        ],
        companions=[_comp("stg_germplasm", "in-situ vs accession geography")],
        do_not=[_avoid("stg_faostat_production", "national production aggregates")],
    ),
    "stg_vegetation_ndvi": _rels(
        joins=[
            _join("stg_nasa_power", ["spatial+time"], "enrich", "NDVI with climate forcing"),
            _join("stg_yield_raw_data", ["spatial proximity", "season"], "enrich", "Greenness vs yield (caution)"),
            _join("stg_protected_areas", ["spatial overlap"], "enrich", "NDVI inside protected areas"),
        ],
        companions=[_comp("stg_ilri_vegetation_feed", "field forage assessments")],
        do_not=[_avoid("stg_openaire_projects", "research projects")],
    ),
    "stg_biodiversity": _rels(
        joins=[
            _join("stg_protected_areas", ["spatial overlap"], "enrich", "Occurrences/rarity with protection"),
            _join("stg_germplasm", ["taxon≈scientific_name/geography"], "compare", "Wild occurrence vs germplasm holdings"),
        ],
        companions=[_comp("stg_vegetation_ndvi", "habitat greenness")],
        do_not=[_avoid("stg_faostat_prices", "commodity prices")],
    ),
    "stg_germplasm": _rels(
        joins=[
            _join("stg_biodiversity", ["taxon"], "compare", "Accessions vs occurrence records"),
            _join("stg_faostat_production", ["taxon≈product_name"], "enrich", "Genetic resources vs production crops"),
        ],
        companions=[_comp("stg_protected_areas", "conservation geography")],
        do_not=[_avoid("stg_nakuru_air_quality", "urban air quality")],
    ),
    "stg_wfp_vampire_prices": _rels(
        joins=[
            _join("stg_fews_market_prices", ["country", "product_name", "year", "month"], "compare", "Align currency and unit"),
            _join("stg_faostat_prices", ["country≈country_name", "product_name", "year"], "compare", "Retail/market vs FAOSTAT prices"),
            _join("stg_yield_raw_data", ["country", "product≈product_name", "year"], "enrich", "Prices with production"),
            _join("stg_fews_food_security", ["country", "year", "month"], "enrich", "Prices with IPC"),
        ],
        companions=[_comp("stg_africa_gdp_ppp", "purchasing power context")],
        do_not=[_avoid("stg_isric_africa_soil", "soil grids")],
    ),
    "stg_openaire_projects": _rels(
        joins=[
            _join("stg_openaire_organisations", ["via product links / funder"], "enrich", "Projects with organisations"),
            _join("stg_openaire_product_links", ["project_id≈target/source"], "enrich", "Projects with linked products"),
            _join("stg_openaire_data_sources", ["shared research graph"], "enrich", "Projects with data sources"),
        ],
        companions=[_comp("stg_openaire_persons", "contributors via links")],
        do_not=[_avoid("stg_yield_raw_data", "agronomic observations; not research graph")],
    ),
    "stg_openaire_organisations": _rels(
        joins=[
            _join("stg_openaire_projects", ["shared research graph"], "enrich", "Orgs with projects"),
            _join("stg_openaire_product_links", ["organisation links"], "enrich", "Orgs with products"),
            _join("stg_openaire_persons", ["coauthor/affiliation via links"], "enrich", "Orgs with people"),
        ],
        companions=[_comp("stg_openaire_data_sources", "hosting repositories")],
        do_not=[_avoid("stg_fews_food_security", "IPC time series")],
    ),
    "stg_openaire_persons": _rels(
        joins=[
            _join("stg_openaire_product_links", ["person links"], "enrich", "People with products"),
            _join("stg_openaire_organisations", ["affiliation via graph"], "enrich", "People with orgs"),
            _join("stg_openaire_projects", ["participation via graph"], "enrich", "People with projects"),
        ],
        do_not=[_avoid("stg_wfp_vampire_prices", "market prices")],
    ),
    "stg_openaire_product_links": _rels(
        joins=[
            _join("stg_openaire_projects", ["target/source ids"], "enrich", "Links resolve project entities"),
            _join("stg_openaire_organisations", ["target/source ids"], "enrich", "Links resolve organisations"),
            _join("stg_openaire_persons", ["target/source ids"], "enrich", "Links resolve persons"),
            _join("stg_openaire_data_sources", ["target/source ids"], "enrich", "Links resolve data sources"),
        ],
        do_not=[_avoid("stg_faostat_production", "production facts")],
    ),
    "stg_openaire_data_sources": _rels(
        joins=[
            _join("stg_openaire_product_links", ["openaire_id"], "enrich", "Sources with linked products"),
            _join("stg_openaire_projects", ["shared research graph"], "enrich", "Sources with projects"),
        ],
        do_not=[_avoid("stg_isda_soil_enriched", "soil properties")],
    ),
    "stg_africa_hdi": _rels(
        joins=[
            _join("stg_africa_gdp_ppp", ["country_code", "year≈observation_year"], "enrich", "HDI with income"),
            _join("stg_faostat_population_employment", ["country≈country_name", "year"], "enrich", "HDI with employment"),
            _join("stg_faostat_sdg_hdi", ["country≈country_name", "year"], "compare", "Parallel development indicators"),
        ],
        companions=[_comp("stg_climatewatch_health", "health/climate development stress")],
        do_not=[_avoid("stg_ilri_vegetation_feed", "plot vegetation surveys")],
    ),
    "stg_africa_gdp_ppp": _rels(
        joins=[
            _join("stg_africa_hdi", ["country_code", "observation_year≈year"], "enrich", "Income with HDI"),
            _join("stg_faostat_production", ["country_code/name", "year"], "enrich", "Macro context for ag production"),
            _join("stg_wfp_vampire_prices", ["country_code≈country", "year"], "enrich", "Affordability backdrop"),
        ],
        companions=[_comp("stg_faostat_prices", "price level context")],
        do_not=[_avoid("stg_germplasm", "accession geography")],
    ),
    "stg_nakuru_air_quality": _rels(
        companions=[
            _comp("stg_climatewatch_health", "health framing for air quality (national scenarios differ in grain)"),
        ],
        do_not=[
            _avoid("stg_yield_raw_data", "national/subnational yields; not Nakuru sensors"),
            _avoid("stg_faostat_trade", "national trade"),
            _avoid("stg_openaire_projects", "research projects"),
        ],
    ),
}


def relationships_for(table_id: str) -> dict[str, Any]:
    bare = (table_id or "").strip().split(".")[-1]
    rel = SEMANTIC_RELATIONSHIPS.get(bare)
    if rel:
        return rel
    return _rels(
        do_not=[_avoid("stg_openaire_projects", "default: avoid inventing cross-domain joins")],
    )


def compact_rels_summary(table_id: str) -> str:
    """One-line summary for the reasoner index."""
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


def documented_join_pairs(
    selected_tables: list[str] | set[str] | None,
) -> list[dict[str, Any]]:
    """Pairwise joins where both ends are selected and YAML lists an explicit ``on=``."""
    selected = {_bare_table(t) for t in (selected_tables or []) if _bare_table(t).startswith("stg_")}
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
                    "how": str(join.get("how") or "").strip(),
                    "note": str(join.get("note") or "").strip(),
                }
            )
    return pairs


def format_join_fragments_for_nl2sql(
    selected_tables: list[str] | set[str] | None,
) -> str:
    """Mandatory JOIN glue for NL2SQL (never invent keys outside YAML ``on=`` lists)."""
    selected = sorted(
        {_bare_table(t) for t in (selected_tables or []) if _bare_table(t).startswith("stg_")}
    )
    if len(selected) < 2:
        return ""
    pairs = documented_join_pairs(selected)
    if not pairs:
        return (
            "JOIN fragments: none documented between the selected tables. "
            "Run SEPARATE SELECTs per table — do NOT invent JOINs or CROSS JOIN."
        )
    lines = ["JOIN fragments (required if multi-table):"]
    for pair in pairs:
        on_sql = " AND ".join(pair["on"])
        bit = f"  {pair['left']} JOIN {pair['right']} ON {on_sql}"
        if pair.get("note"):
            bit += f"  -- {pair['note']}"
        lines.append(bit)
    return "\n".join(lines)
