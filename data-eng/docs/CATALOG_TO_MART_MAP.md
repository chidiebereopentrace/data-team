# Catalog → mart map

Business glossary (catalog) vs physical dbt mart (`mart_dev`). Use this when names disagree; do not invent empty dims to match CSV labels.

**Analyst handoff docs**

| Doc | Purpose |
|-----|---------|
| [mart_dev_entity_dictionary.xlsx](./mart_dev_entity_dictionary.xlsx) | Full entity / column dictionary (all `mart_dev` tables) |
| [MART_DEV_OTA_ANALYST_GUIDE.docx](./MART_DEV_OTA_ANALYST_GUIDE.docx) | DOCX playbook for analysts writing OTA insights |
| [MART_DEV_OTA_ANALYST_GUIDE.md](./MART_DEV_OTA_ANALYST_GUIDE.md) | Markdown twin of the OTA guide |
| [mart_entity_dictionary_seed.yaml](./mart_entity_dictionary_seed.yaml) | Curated seed used to regenerate the workbook |
| [bq_mart_tables_yaml_files/](../../ml-eng/ml/rag/bq_mart_tables_yaml_files/) | Per-table mart_dev YAML with live `{column}_value_samples` (RAG BQ catalog) |

Regenerate entity docs: `python scripts/build_mart_entity_dictionary.py` and `python scripts/build_mart_ota_analyst_guide_docx.py` from `data-eng/`.

Regenerate mart column YAMLs (after `dbt build mart_dev`): `python data-eng/data/local/scripts/regenerate_mart_table_yamls.py` — see [MART_QA_NOTES.md](./MART_QA_NOTES.md).

| Catalog concept | Mart object | Notes |
|-----------------|-------------|-------|
| country | `dim_geography` (`geo_level = 'country'`) | No `dim_country` view unless product asks |
| admin / FNID / site | `dim_geography` (other `geo_level`) | Join facts on `geography_key` |
| crop / commodity | `dim_product` | Production, prices, food balance, trade |
| FAOSTAT item (long) | `dim_item` | Emissions, employment, land inputs, investment |
| FAOSTAT element | `dim_element` | Same long facts |
| period / as-of | `as_of_date`, `date_key` | Observation date; `loaded_at` is pipeline |
| agricultural season | `dim_season` | Used by `fct_yield` (`fnid_season`) — do not mix with FAOSTAT year grain |
| FNID–season yield | `fct_yield` | `yield_raw_data` only — separate from `fct_production` for citation SoT |
| ILRI dairy milk/calving | *(not in production/yield facts)* | Cow-day grain — leave out |
| land use / fertilizer / pesticide | `fct_land_inputs` (`input_grain`) | One fact, filter by grain — not three tables |
| crops & livestock trade + FEWS border | `fct_trade` (`trade_grain`) | Separate from land-input trade and forestry trade |
| forestry production / trade flows | `fct_forestry` (`forestry_grain`) | Keep grain filter before charts |
| national SDG gender | `fct_gender_inclusion` | Do not union household gender scores |
| household survey gender / FIES | `fct_household` | Community grain; keep separate |
| organisation (research / ASTI) | `dim_organisation` (`org_source`) | OpenAIRE + ASTI institutions |
| data lineage / ACF source | `dim_source` ← `int_source_registry` | Every fact keeps `source_key` + `source_natural_key` |

## Grain rules

- FAOSTAT country–year is `fct_production` (long-form: `element_key` + `value`, filter `production_grain`); FNID–season yield is `fct_yield` — never average across those tables.
- `fct_production` grains: `physical` (area/production/yield elements), `index`, `gross_value`. Use `production_grain = 'physical'` for area/qty aggregates (`agg_production_country_year`).
- Yield units on `fct_yield` are documented defaults (`ha` / `tonnes` / `t/ha`); FAOSTAT units on `fct_production.unit`.
- Price months must be calendar 1–12 or null (FAOSTAT `months_code` like `7004` is label-only).
- Aggregates always group by `source_key` — no silent source blend.
- Fact geo packaging: `geography_key` / `geo_level` / `place_scope` / `data_level` / `geo_scope` are dim-only (`acf_*_strict` — null scope when key null). `country_iso3` may be filled from extractable geo (dim or intermediate) even when `geography_key` is null.

## ACF warehouse contract

Every `fct_*` exposes columns so cited BigQuery rows map to ACF without adapter heuristics. The chatbot adapter is `warehouse_row_to_acf_record` in `ml-eng/ml/rag/chatbot/acf_metadata.py`.

| Column | Meaning | Example |
|--------|---------|---------|
| `tier` | producer scale (1=global, 2=national ministry, 3=community survey) | `1` |
| `data_level` | resolution of the number | `sub_national` |
| `place_scope` | ARRAY of comparable place labels (ACF geo overlap) | `['ETH', 'ETR103', 'Oromia']` |
| `metric` | stable slug for the measure | `price_retail_maize` |
| `source_id` | lineage key (= `dim_source.source_key`) | `faostat_prices_eth` |
| `value` / `unit` | measure | `42.5`, `ETB/kg` |
| `as_of_date` | observation date when known; else `DATE(loaded_at)` | `2024-06-30` |
| `as_of_date_basis` | `observation` or `loaded_at` | `observation` |
| `geo_scope` | national / sub_national / community / point | `national` |

**Column label dictionaries:** live distinct values for every mart column are profiled into per-table YAML under [`bq_mart_tables_yaml_files/`](../../ml-eng/ml/rag/bq_mart_tables_yaml_files/) (regenerate via [`regenerate_mart_table_yamls.py`](../data/local/scripts/regenerate_mart_table_yamls.py)). Load with [`bq_table_schema_yaml.load_mart_table_schema()`](../../ml-eng/ml/rag/chatbot/bq_table_schema_yaml.py) when extending the mart BQ reasoner.

**Naming:** warehouse `data_level` is row resolution. It is **not** Qdrant coverage-class `geo_scope` (`country|multi_country|regional|global`). ACF place overlap uses `place_scope` → adapter `geo_scope` list.

**Unscorable rows:** historically null `as_of_date` on household/animal_health/NDVI/iSDA/protected/germplasm — Phase 2 fills via `DATE(loaded_at)` with `as_of_date_basis='loaded_at'`. Adapter still treats `loaded_at` basis as pipeline time, not observation freshness.

**Direction:** warehouse v1 rows are **snapshots** — adapter sets `direction='unknown'` unless a temporal pair is present. Direction (D) is derived from: (1) `value` + `prior_value` on the row (including `yoy_delta` shapes `total_curr`/`total_prev`), (2) ranked-table **trend companion** on `fct_production` (Y−1 vs Y+1 bracketing years), or (3) vector claim extract on news/research. **Alignment (A)** requires multiple **cited** claims with distinct `source_id`s at generation time (`acf_scoring.score_cited_evidence`); uncited BQ rows do not affect the score.

Macros: `acf_data_level`, `acf_place_scope`, `acf_row_data_level` in `data-eng/dbt/macros/acf_contract.sql`.

## Geo enrichment

Facts must not join geography by name in the mart. Intermediate models emit `geo_key`; facts join `dim_geography` on `geography_key = geo_key` only.

```text
stg_geo (source table in staging_dev)
  → int_geography_conformed (city + admin + FAOSTAT country spine)
    → int_latlon_country_grid (1dp cells → country for point fallback)
    → int_*_with_geo (prices, ISRIC/iSDA soil, yield, …)
      → fct_* + dim_geography
```

| Domain | Intermediate geo model | Join strategy |
|--------|------------------------|---------------|
| Prices | `int_prices_with_geo` | Three branches by `price_source`: **FAOSTAT** — `area_code` → country spine; **FEWS** — ISO2 `country_code` + admin2 → admin1 → country; **WFP** — resolved ISO3 (`int_ref_country` + `geo_faostat_country_name_iso3`) → country spine, admin1 by ISO3 |
| ISRIC soil | `int_isric_soil_with_geo` | City 2dp lat/lon match → `int_latlon_country_grid` 1dp country fallback |
| iSDA soil | `int_isda_soil_with_geo` | City match + country name + grid fallback |

Shared macros: `geo_lookup.sql` (`geo_country_by_iso2_cte`, `geo_city_by_latlon_cte`, FEWS admin joins).

**Prerequisite:** `staging_dev.stg_geo` must be loaded before `int_geography_conformed`. Unmatched rows keep null `geo_key` — check coverage with QA queries on `int_prices_with_geo` / `fct_soil_health`.

## Physical layout (BigQuery)

| Layer | Partition | Cluster |
|-------|-----------|---------|
| Time-series facts (Phase 1 + 2) | `as_of_date` (DATE, month) | `data_level`, `country_iso3`, `source_key` (+ grain/product/item as needed) |
| Cluster-only facts | none | see exceptions below |
| Dimensions | none | `dim_geography`: `geo_level`, `country_iso3` |
| Aggregates | none | Prefer `agg_*` for country rollups |

**Partitioned facts (Phase 1):** `fct_prices`, `fct_production`, `fct_yield`, `fct_food_security`, `fct_climate`.

**Partitioned facts (Phase 2):** `fct_trade`, `fct_emissions`, `fct_food_balance`, `fct_land_inputs`, `fct_forestry`, `fct_economics`, `fct_employment`, `fct_investment`, `fct_machinery`, `fct_humanitarian`, `fct_gender_inclusion`, `fct_hdi`, `fct_food_hazards`, `fct_air_quality`, `fct_biodiversity`, `fct_insurance`.

**Cluster-only exceptions** (no partition until real observation dates exist):

| Fact | Cluster | Reason |
|------|---------|--------|
| `fct_soil_health` | `source_key`, `soil_property`, `data_level` | iSDA branch: `as_of_date` null |
| `fct_vegetation` | `vegetation_grain`, `data_level`, `source_key` | NDVI branch has no `as_of_date` |
| `fct_household` | `data_level`, `country_iso3`, `source_key` | survey year not yet in staging |
| `fct_animal_health` | `source_key` | survey date not yet in staging |
| `fct_protected_areas` | `source_key` | no observation date |
| `fct_germplasm` | `source_key` | no observation date |

**Hierarchy** lives on `dim_geography`: `data_level`, `country_key`, `parent_geography_key` (one hop). Facts denormalize `geo_level` / `data_level` / `country_iso3` for clustering — grain place remains `geography_key`.

**Query habits**

```sql
-- good: prunes partitions
where as_of_date between '2018-01-01' and '2024-12-31'
  and data_level = 'sub_national'
  and country_iso3 = 'ETH'
  and source_key = '…'

-- bad: hides the partition column
where extract(year from as_of_date) = 2020
```

Never partition by STRING year or country code. Cluster max 4 columns; data level before source.
