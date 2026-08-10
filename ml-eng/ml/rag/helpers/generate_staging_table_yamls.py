"""Generate staging_dev per-table YAMLs for BQ RAG (replaces bronze/raw YAMLs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ml.rag.helpers.staging_semantic_relationships import relationships_for

OUT = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"
PROJECT = "opentrace-prod-5ga4"
DATASET = "staging_dev"


def _col(name: str) -> dict[str, str]:
    typ = "STRING"
    if name in {
        "year",
        "month",
        "planting_year",
        "harvest_year",
        "observation_year",
        "planting_month",
        "harvest_month",
        "qc_flag",
        "hh_size",
        "individual_count",
    }:
        typ = "INT64"
    elif name in {
        "value",
        "yield",
        "area",
        "production",
        "hdi_value",
        "gdp_per_capita_ppp",
        "temperature_2m",
        "pm10",
        "pm2_5",
        "latitude",
        "longitude",
        "pct_phase3",
        "pct_phase4",
        "pct_phase5",
    } or name.endswith("_pct") or name.startswith(("bdod_", "cec_", "clay_", "nitrogen_", "phh2o_", "sand_", "silt_", "soc_")):
        typ = "FLOAT64"
    return {"name": name, "type": typ, "description": name.replace("_", " ")}


def _table(
    name: str,
    domain: str,
    desc: str,
    grain: str,
    tags: list[str],
    cols: list[str],
    questions: list[str] | None = None,
    hints: list[str] | None = None,
    filters: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "table_name": f"{PROJECT}.{DATASET}.{name}",
        "description": desc,
        "grain": grain,
        "entity_type": domain,
        "source": {"layer": "staging_dev", "domain": domain},
        "relationships": {"primary_key": {"type": "logical", "fields": grain}},
        "semantic_role": {"primary_domain": domain, "supports": tags},
        "business_questions_supported": questions
        or [f"What insights does {name} provide for African agriculture?"],
        "filtering_guidance": filters or ["Filter by geography and time when present"],
        "sql_generation_hints": hints
        or [f"Query only `{PROJECT}.{DATASET}.{name}`", "Always include LIMIT"],
        "semantic_relationships": relationships_for(name),
        "columns": [_col(c) for c in cols],
        "_index_domain": domain,
        "_index_tags": tags,
    }


def _definitions() -> list[dict[str, Any]]:
    fao_cols = [
        "area_code",
        "area_code_m49",
        "country_name",
        "item_code",
        "item_code_cpc",
        "product_name",
        "element_code",
        "element",
        "year",
        "unit",
        "value",
        "source_natural_key",
        "loaded_at",
    ]
    fao_hints = [
        "Filter country_name and year",
        "Filter element for the metric needed",
        "GROUP BY country_name, product_name, year when comparing",
    ]
    out: list[dict[str, Any]] = []
    add = out.append

    add(
        _table(
            "stg_fews_food_security",
            "fews",
            "FEWS NET food security / IPC phases and insecure population estimates.",
            "fnid/admin × year × month × measure_type × scenario",
            ["food security", "ipc", "fews"],
            [
                "fnid",
                "country",
                "country_code",
                "admin_0",
                "admin_1",
                "admin_2",
                "admin_3",
                "admin_4",
                "geographic_unit_name",
                "fewsnet_region",
                "phase_code",
                "phase_name",
                "classification_scale",
                "scenario_name",
                "value",
                "low_value",
                "high_value",
                "pct_phase3",
                "pct_phase4",
                "pct_phase5",
                "is_allowing_for_assistance",
                "year",
                "month",
                "measure_type",
                "source_natural_key",
                "loaded_at",
            ],
            [
                "Which districts are in IPC phase 3+?",
                "How has food insecurity changed recently?",
            ],
            [
                "Filter by country/fnid and year/month",
                "Use phase_code or pct_phase3-5 for severity",
                "GROUP BY geography for which-region questions",
            ],
        )
    )
    add(
        _table(
            "stg_fews_market_prices",
            "fews",
            "FEWS NET market prices for staple commodities.",
            "country × market × product × year × month × price_type",
            ["markets", "prices", "fews"],
            [
                "country",
                "country_code",
                "admin_1",
                "admin_2",
                "market_name",
                "product_name",
                "cpcv2",
                "price_type",
                "unit",
                "currency",
                "value",
                "common_unit_price",
                "common_currency_price",
                "year",
                "month",
                "source_natural_key",
                "loaded_at",
            ],
            ["What are maize prices in key markets?"],
            ["Prefer common_currency_price for cross-market compare"],
        )
    )
    add(
        _table(
            "stg_fews_cross_border_trade",
            "fews",
            "FEWS NET cross-border trade flows for agricultural products.",
            "border_point × product × year × month × trade_flow",
            ["trade", "markets", "fews"],
            [
                "country",
                "country_code",
                "border_point",
                "source_country",
                "source_country_code",
                "destination_country",
                "destination_country_code",
                "product_name",
                "cpcv2",
                "trade_flow",
                "trade_type",
                "unit",
                "value",
                "common_unit_quantity",
                "year",
                "month",
                "source_natural_key",
                "loaded_at",
            ],
            ["Which border points move the most maize?"],
            ["Filter trade_flow and product_name"],
        )
    )
    add(
        _table(
            "stg_faostat_production",
            "faostat",
            "FAOSTAT crop/livestock production, indices, and value of production.",
            "country × product × element × year",
            ["production", "yield", "faostat"],
            fao_cols,
            ["What is production of a crop in a country?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_emissions",
            "faostat",
            "FAOSTAT agrifood systems emissions.",
            "country × item × element × year",
            ["emissions", "climate", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "item_code",
                "item_code_cpc",
                "item",
                "element_code",
                "element",
                "source_code",
                "source",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What are agrifood emissions for a country?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_prices",
            "faostat",
            "FAOSTAT producer prices, CPI, deflators, exchange rates.",
            "country × product × element × year × month",
            ["prices", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "item_code",
                "item_code_cpc",
                "product_name",
                "element_code",
                "element",
                "months_code",
                "months",
                "iso_currency_code",
                "currency",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What are producer prices for a crop?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_trade",
            "faostat",
            "FAOSTAT crop and livestock trade products and indices.",
            "country × product × element × year",
            ["trade", "faostat"],
            fao_cols,
            ["What are import/export volumes?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_food_balances",
            "faostat",
            "FAOSTAT food balances and supply utilization accounts.",
            "country × product × element × year",
            ["food balances", "nutrition", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "item_code",
                "item_code_fbs",
                "item_code_cpc",
                "product_name",
                "element_code",
                "element",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is food supply for a commodity?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_investment_asti",
            "faostat",
            "FAOSTAT agricultural investment flows and ASTI indicators.",
            "country × indicator/purpose × year",
            ["investment", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "donor_code",
                "donor_code_m49",
                "donor",
                "purpose_code",
                "purpose",
                "item_code",
                "item",
                "element_code",
                "element",
                "indicator_code",
                "indicator",
                "institution_code",
                "institution",
                "degree_code",
                "degree",
                "sex_code",
                "sex",
                "cost_category_code",
                "cost_category",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["How much ag R&D spending is there?"],
            ["Filter indicator/purpose and year"],
        )
    )
    add(
        _table(
            "stg_faostat_land_inputs",
            "faostat",
            "FAOSTAT land use, fertilizers, pesticides, manure, nutrient balance.",
            "country × item × element × year",
            ["land", "fertilizer", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "partner_country_code",
                "partner_country_code_m49",
                "partner_countries",
                "item_code",
                "item_code_cpc",
                "item",
                "months_code",
                "months",
                "element_code",
                "element",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is fertilizer use by country?"],
            fao_hints,
        )
    )
    add(
        _table(
            "stg_faostat_population_employment",
            "faostat",
            "FAOSTAT population and agricultural/rural employment.",
            "country × indicator × year",
            ["population", "employment", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "item_code",
                "item",
                "element_code",
                "element",
                "indicator_code",
                "indicator",
                "source_code",
                "source",
                "sex_code",
                "sex",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What share of employment is in agriculture?"],
            ["Filter indicator and sex when relevant"],
        )
    )
    add(
        _table(
            "stg_faostat_sdg_hdi",
            "faostat",
            "FAOSTAT SDG indicators, healthy diet cost, food value chain, census.",
            "country × indicator × year",
            ["sdg", "nutrition", "faostat"],
            [
                "area_code",
                "area_code_m49",
                "country_name",
                "item_code",
                "item_code_sdg",
                "item",
                "food_value_code",
                "food_value",
                "industry_code",
                "industry",
                "factor_code",
                "factor",
                "release_code",
                "release",
                "census_year_code",
                "census_year",
                "element_code",
                "element",
                "year",
                "unit",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is the cost of a healthy diet?"],
            ["Filter item/indicator carefully"],
        )
    )
    add(
        _table(
            "stg_yield_raw_data",
            "production",
            "Crop yield, area, and production with season and production system.",
            "fnid/admin × product × season × harvest_year",
            ["yield", "production", "crops"],
            [
                "fnid",
                "country",
                "country_code",
                "admin_1",
                "admin_2",
                "product",
                "season_name",
                "planting_year",
                "planting_month",
                "harvest_year",
                "harvest_month",
                "crop_production_system",
                "qc_flag",
                "area",
                "production",
                "yield",
                "source_natural_key",
                "loaded_at",
            ],
            [
                "Which districts have highest maize yield?",
                "How did yields change over the past decade?",
            ],
            [
                "Prefer harvest_year for time filters",
                "Recompute yield as production/area when aggregating",
                "GROUP BY product + geography + year",
            ],
            ["Prefer clean qc_flag rows when available"],
        )
    )
    add(
        _table(
            "stg_ilri_dairy_genetics",
            "production",
            "ILRI dairy genetics East Africa milk and calving records.",
            "farm/cow × record_type × date",
            ["dairy", "livestock", "ilri"],
            [
                "client_id",
                "client_name",
                "country",
                "farm_code",
                "cow_number",
                "breed",
                "number_of_lactations",
                "date_of_milking",
                "test_date",
                "yield_afternoon",
                "yield_morning",
                "parity_number",
                "initial_calving_date",
                "record_type",
                "source_natural_key",
                "loaded_at",
            ],
            ["What are milk yields by breed?"],
            ["Filter record_type milk vs calving"],
        )
    )
    add(
        _table(
            "stg_ilri_household_food_security",
            "ilri",
            "ILRI household food security survey indicators.",
            "household_id",
            ["food security", "household", "ilri"],
            [
                "household_id",
                "country",
                "region",
                "district",
                "village_code",
                "hh_size",
                "hh_size_mae",
                "household_type",
                "head_education_level",
                "land_cultivated",
                "livestock_holdings",
                "fies_score",
                "total_income",
                "farm_income",
                "offfarm_income",
                "value_farm_produce",
                "crop_sales",
                "value_crop_produce",
                "value_crop_consumed",
                "livestock_product_sales",
                "value_livestock_production",
                "value_livestock_consumed",
                "food_availability",
                "food_self_sufficiency",
                "total_energy_available",
                "gender_male_control",
                "gender_female_control",
                "gender_male_youth_control",
                "gender_female_youth_control",
                "crop_diversity",
                "livestock_diversity",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is average FIES by region?"],
            ["Aggregate carefully; survey microdata"],
        )
    )
    add(
        _table(
            "stg_ilri_animal_health",
            "ilri",
            "ILRI animal health surveys (disease, acaricide, AMUSE).",
            "household × species",
            ["animal health", "livestock", "ilri"],
            [
                "household_id",
                "sub_county",
                "village_name",
                "village_code",
                "farmer_sex",
                "education_level",
                "housing_type",
                "breed_type",
                "animals_sold",
                "cash_received",
                "clinical_disease_score",
                "positive_cases",
                "incidence_rate",
                "treatment_cost",
                "average_herd_size",
                "mortality_count",
                "species",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is disease incidence by species?"],
        )
    )
    add(
        _table(
            "stg_ilri_vegetation_feed",
            "ilri",
            "ILRI vegetation survey and feed assessment.",
            "lat/lon or household × record_type",
            ["feed", "vegetation", "ilri"],
            [
                "latitude",
                "longitude",
                "survey_date",
                "quantity_trees",
                "quantity_shrubs",
                "quantity_grass",
                "leaves_trees",
                "leaves_shrubs",
                "leaves_grass",
                "palatability_trees",
                "palatability_shrubs",
                "palatability_grass",
                "carrying_capacity",
                "currently_grazing",
                "photo_id",
                "household_id",
                "agro_climatic_zone_id",
                "record_type",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is carrying capacity in surveyed sites?"],
            ["Filter record_type"],
        )
    )
    add(
        _table(
            "stg_ilri_i4i_livestock_insurance",
            "ilri",
            "ILRI index-based livestock insurance farmer/vaccinator data.",
            "household/vaccinator × record_type",
            ["insurance", "livestock", "ilri"],
            [
                "household_id",
                "country",
                "region_id",
                "farmer_location",
                "farmer_location_new",
                "farmer_category",
                "categorisation",
                "herd_size_category",
                "herd_size_cat",
                "herd_size_cat_new",
                "insurance_start_year",
                "itm_start_year",
                "insurance_start_year_new",
                "vaccinator_id",
                "local_currency",
                "currency_label",
                "record_type",
                "source_natural_key",
                "loaded_at",
            ],
            ["How many insured farmers by category?"],
            ["Filter record_type farmer vs vaccinator"],
        )
    )
    add(
        _table(
            "stg_ilri_vendor_consumer",
            "ilri",
            "ILRI vendor and consumer surveys (BF, ET).",
            "respondent × country × type",
            ["markets", "consumers", "ilri"],
            [
                "outlet_id",
                "survey_date",
                "respondent_age",
                "respondent_sex",
                "consumer_id",
                "respondent_type",
                "country",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_ilri_food_hazards",
            "ilri",
            "ILRI foodborne hazards meta-analysis (Burkina Faso).",
            "study × hazard",
            ["food safety", "hazards", "ilri"],
            [
                "author_year",
                "publication_year",
                "study_site",
                "sampling_points",
                "samples_type",
                "sample_subgroup",
                "foodborne_hazard",
                "total_samples",
                "positive_samples",
                "mean_cfu_per_g_log",
                "standard_deviations",
                "country",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_ilri_other_surveys",
            "ilri",
            "ILRI lower-priority survey packs.",
            "household × survey_type",
            ["surveys", "ilri"],
            [
                "enumerator",
                "survey_start",
                "deviceid",
                "village",
                "block",
                "household_id",
                "country",
                "survey_type",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Filter survey_type"],
        )
    )
    add(
        _table(
            "stg_nasa_power",
            "climate",
            "NASA POWER daily/hourly solar and climate observations.",
            "lat/lon × observation_time",
            ["climate", "solar", "nasa"],
            [
                "country_code",
                "country_name",
                "admin_region",
                "latitude",
                "longitude",
                "elevation_meters",
                "par_solar_at_noon",
                "shortwave_irradiance_at_noon",
                "uva_radiation_at_noon",
                "uvb_radiation_at_noon",
                "observation_time",
                "fetched_at",
                "processed_at",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_copernicus_era5",
            "climate",
            "Copernicus ERA5 reanalysis temperature fields.",
            "lat/lon × time × ensemble",
            ["climate", "temperature", "era5"],
            [
                "time",
                "valid_time",
                "latitude",
                "longitude",
                "ensemble_member",
                "step",
                "surface",
                "temperature_2m",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_climatewatch_health",
            "climate",
            "ClimateWatch climate–health impact indicators.",
            "country × indicator × year × scenario",
            ["climate", "health"],
            [
                "country_name",
                "country_code",
                "model",
                "scenario",
                "category",
                "subcategory",
                "indicator",
                "unit",
                "year",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Filter scenario and indicator"],
        )
    )
    soil_cols = [
        "latitude",
        "longitude",
        "fetched_date",
        "bdod_0_5cm",
        "bdod_5_15cm",
        "bdod_15_30cm",
        "bdod_30_60cm",
        "bdod_60_100cm",
        "cec_0_5cm",
        "cec_5_15cm",
        "cec_15_30cm",
        "cec_30_60cm",
        "cec_60_100cm",
        "clay_0_5cm",
        "clay_5_15cm",
        "clay_15_30cm",
        "clay_30_60cm",
        "clay_60_100cm",
        "nitrogen_0_5cm",
        "nitrogen_5_15cm",
        "nitrogen_15_30cm",
        "nitrogen_30_60cm",
        "nitrogen_60_100cm",
        "phh2o_0_5cm",
        "phh2o_5_15cm",
        "phh2o_15_30cm",
        "phh2o_30_60cm",
        "phh2o_60_100cm",
        "sand_0_5cm",
        "sand_5_15cm",
        "sand_15_30cm",
        "sand_30_60cm",
        "sand_60_100cm",
        "silt_0_5cm",
        "silt_5_15cm",
        "silt_15_30cm",
        "silt_30_60cm",
        "silt_60_100cm",
        "soc_0_5cm",
        "soc_5_15cm",
        "soc_15_30cm",
        "soc_30_60cm",
        "soc_60_100cm",
        "source_natural_key",
        "loaded_at",
    ]
    add(
        _table(
            "stg_isric_africa_soil",
            "soil_and_land",
            "ISRIC Africa soil properties by depth (wide).",
            "lat/lon",
            ["soil", "isric"],
            soil_cols,
            hints=["Select only needed depth columns"],
        )
    )
    add(
        _table(
            "stg_isda_soil_enriched",
            "soil_and_land",
            "iSDA soil properties in long format (property × depth × value).",
            "lat/lon × soil_property × depth",
            ["soil", "isda"],
            [
                "longitude",
                "latitude",
                "country",
                "city",
                "soil_property",
                "depth",
                "value",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Filter soil_property and depth"],
        )
    )
    add(
        _table(
            "stg_unccd_land_degradation",
            "soil_and_land",
            "UNCCD land degradation indicators.",
            "geo_area × indicator × time",
            ["land degradation", "unccd"],
            [
                "goal",
                "target",
                "indicator",
                "series",
                "series_description",
                "series_count",
                "geo_area_code",
                "geo_area_name",
                "time_period_start",
                "value",
                "value_type",
                "time_detail",
                "time_coverage",
                "upper_bound",
                "lower_bound",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_s4a_field_surveys",
            "soil_and_land",
            "S4A field survey soil surface plots.",
            "plot",
            ["soil", "field survey"],
            [
                "plot_code",
                "plot_id",
                "country_code",
                "survey_date",
                "altitude",
                "longitude",
                "latitude",
                "tsu_id",
                "obstruction_layer",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_cifor_icraf",
            "soil_and_land",
            "CIFOR-ICRAF plot soil nutrients.",
            "plot_id",
            ["soil", "nutrients"],
            [
                "plot_id",
                "treatment",
                "soil_type",
                "carbon_pct",
                "nitrogen_pct",
                "phosphorus_pct",
                "potassium_pct",
                "calcium_pct",
                "magnesium_pct",
                "source_doi",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_protected_areas",
            "spatial",
            "Protected area features and area protected.",
            "protected_area",
            ["biodiversity", "protected areas"],
            [
                "objectid",
                "protected_area_name",
                "feature_count",
                "area_protected",
                "geometry_wkt",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Avoid selecting large geometry_wkt unless needed"],
        )
    )
    add(
        _table(
            "stg_vegetation_ndvi",
            "spatial",
            "Vegetation NDVI grid means.",
            "grid_id",
            ["ndvi", "vegetation"],
            [
                "objectid",
                "grid_id",
                "ndvi_mean",
                "ndvi_mean_secondary",
                "shape_area",
                "shape_length",
                "geometry_wkt",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Avoid large geometry unless needed"],
        )
    )
    add(
        _table(
            "stg_biodiversity",
            "spatial",
            "GBIF occurrences and rarity-80 bird richness layers.",
            "occurrence or feature",
            ["biodiversity", "species"],
            [
                "gbif_id",
                "dataset_key",
                "occurrence_id",
                "kingdom",
                "phylum",
                "classification",
                "taxonomic_order",
                "family",
                "genus",
                "species",
                "infraspecific_epithet",
                "taxon_rank",
                "scientific_name",
                "verbatim_scientific_name",
                "country_code",
                "locality",
                "state_province",
                "occurrence_status",
                "individual_count",
                "latitude",
                "longitude",
                "coordinate_uncertainty_m",
                "elevation_m",
                "year",
                "month",
                "day",
                "event_date",
                "basis_of_record",
                "taxon_key",
                "species_key",
                "ingested_at",
                "feature_name",
                "area_km2",
                "prop_gadm",
                "rich_all",
                "rich_amph",
                "rich_bird",
                "rich_cact",
                "rich_coni",
                "rich_mamm",
                "rich_rept",
                "rar_all",
                "rar_amph",
                "rar_bird",
                "rar_cact",
                "rar_coni",
                "rar_mamm",
                "rar_rept",
                "area_protected",
                "geometry_wkt",
                "source_natural_key",
                "loaded_at",
            ],
            hints=["Prefer species + country_code filters"],
        )
    )
    add(
        _table(
            "stg_germplasm",
            "spatial",
            "Crop and rice germplasm accessions with geography.",
            "germplasm_id",
            ["germplasm", "crops"],
            [
                "germplasm_id",
                "taxon",
                "objectid",
                "geography",
                "geometry_wkt",
                "source_natural_key",
                "loaded_at",
            ],
        )
    )
    add(
        _table(
            "stg_wfp_vampire_prices",
            "market_prices",
            "WFP VAMPIRE global food prices curated for market analytics.",
            "country × market × product × year × month × price_type",
            ["prices", "markets", "wfp"],
            [
                "country_id",
                "country",
                "admin1_id",
                "admin1_name",
                "market_id",
                "market_name",
                "product_id",
                "product_name",
                "currency_id",
                "currency",
                "price_type_id",
                "price_type",
                "unit_id",
                "unit",
                "year",
                "month",
                "value",
                "commodity_source",
                "source_natural_key",
                "loaded_at",
            ],
            [
                "What are food prices in a market?",
                "How did prices change over months?",
            ],
            [
                "Filter country, product_name, year/month",
                "GROUP BY market or month for trends",
            ],
            ["Do not average across currencies or units without care"],
        )
    )
    for name, desc, cols in [
        (
            "stg_openaire_projects",
            "OpenAIRE research projects (agriculture and environment).",
            [
                "ingestion_id",
                "project_id",
                "project_code",
                "acronym",
                "project_title",
                "start_date",
                "currency",
                "total_cost",
                "funded_amount",
                "primary_funder_name",
                "jurisdiction",
                "funding_stream_desc",
                "fetched_at",
                "source_natural_key",
                "loaded_at",
            ],
        ),
        (
            "stg_openaire_organisations",
            "OpenAIRE organisations.",
            [
                "organisation_id",
                "legal_name",
                "short_name",
                "website_url",
                "country_code",
                "country_name",
                "alternative_names",
                "pids",
                "fetched_at",
                "source_natural_key",
                "loaded_at",
            ],
        ),
        (
            "stg_openaire_persons",
            "OpenAIRE persons.",
            [
                "person_id",
                "given_name",
                "family_name",
                "biography",
                "coauthor_count",
                "fetched_at",
                "source_natural_key",
                "loaded_at",
            ],
        ),
        (
            "stg_openaire_product_links",
            "OpenAIRE product relationship links.",
            [
                "openaire_id",
                "title",
                "entity_type",
                "pub_date",
                "publisher",
                "language",
                "rel_name",
                "target_id",
                "target_type",
                "fetched_at",
                "source_natural_key",
                "loaded_at",
            ],
        ),
        (
            "stg_openaire_data_sources",
            "OpenAIRE data sources.",
            [
                "ingestion_id",
                "fetched_at",
                "openaire_id",
                "official_name",
                "english_name",
                "website_url",
                "source_type",
                "compatibility",
                "subjects",
                "issn_online",
                "issn_printed",
                "source_natural_key",
                "loaded_at",
            ],
        ),
    ]:
        add(_table(name, "research", desc, "entity row", ["research", "openaire"], cols))
    add(
        _table(
            "stg_africa_hdi",
            "socio_economic",
            "Africa Human Development Index by country and year.",
            "country × year",
            ["hdi", "development"],
            ["country", "country_code", "year", "hdi_value", "source_natural_key", "loaded_at"],
            ["How has HDI changed for a country?"],
            ["Filter country_code and year"],
        )
    )
    add(
        _table(
            "stg_africa_gdp_ppp",
            "socio_economic",
            "Africa GDP per capita PPP by country and year.",
            "country × observation_year",
            ["gdp", "economics"],
            [
                "country_name",
                "country_code",
                "observation_year",
                "gdp_per_capita_ppp",
                "ingested_at",
                "source_natural_key",
                "loaded_at",
            ],
            ["What is GDP per capita PPP?"],
            ["Filter country_code"],
        )
    )
    add(
        _table(
            "stg_nakuru_air_quality",
            "socio_economic",
            "Nakuru air quality sensor archive.",
            "sensor × observation_timestamp",
            ["air quality", "nakuru"],
            [
                "sensor_id",
                "sensor_type",
                "location",
                "latitude",
                "longitude",
                "observation_timestamp",
                "pm10",
                "pm2_5",
                "humidity_pct",
                "temperature_c",
                "source_archive",
                "source_natural_key",
                "loaded_at",
            ],
            ["What are recent PM2.5 levels in Nakuru?"],
            ["Filter time range"],
        )
    )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*.yml"):
        path.unlink()
    defs = _definitions()
    assert len(defs) == 42, len(defs)
    for doc in defs:
        name = str(doc["table_name"]).rsplit(".", 1)[-1]
        # Strip private index helpers from on-disk YAML? Keep them for index builder convenience.
        path = OUT / f"{name}.yml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(defs)} yamls to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
