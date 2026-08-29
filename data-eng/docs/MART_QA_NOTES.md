# Mart QA inventory (Phase 0)

Snapshot: `opentrace-prod-5ga4.mart_dev` via BigQuery client (2026-08-25).

## Flags

| Finding | Status |
|---------|--------|
| Production only `yield_raw_data`? | **No** — historically unioned; **now split**: `fct_production` = FAOSTAT only, `fct_yield` = `yield_raw_data` |
| HDI `geo_null` = 100% | **P0 confirmed** — `fct_hdi` joins `dim_geography` on `geo_level='country' AND country_iso3`; all 57 country-level dim rows have **null** `country_iso3`. Cities have ISO3. `int_hdi_conformed.geo_key` is fully populated but unused by the fact. |
| Employment FAOSTAT geo | Mostly OK (0–5.2% null by source) |
| WFP `common_unit_price` null | **Expected** — do not invent |

Root cause for HDI / country ISO: `int_geography_conformed` FAOSTAT/FEWS country spines set `country_iso3` null; only `stg_geo` cities carry ISO3.

## Inventory by fact

See also `_mart_qa_inventory_raw.md` (full tables).

### fct_production (~2.0m FAOSTAT long-form)

Long-form grain: one row per `area_code + item + element + year + source`. `production_grain`: `physical` (Crops_and_livestock), `index` (Production Indices), `gross_value` (Value of Agricultural Production). Every row has non-null `value` + `element_key`. Physical rows optionally expose `area_harvested` / `production_qty` / `yield_value` when element matches.

### fct_yield (~203k)
`yield_raw_data` only (`fnid_season`).

### fct_hdi (1,597)
`africa_Human_development_index` — **geo_null 100%**, asof/src OK. Fact `country_iso3` column is null (taken from failed join).

### fct_employment (81,586)
geo_null 0–5.2% by FAOSTAT source. asof/src OK.

### fct_prices (~1.96M)

Unified fact over **three** harmonised feeds (`price_source`: `fews`, `wfp`, `faostat`) via `int_prices_harmonised` → `int_prices_with_geo`. Always filter by `price_source` and `source_key` when comparing prices.

| Source | Rows (mart_dev_sa) | geo_null | Grain | Geo attach |
|--------|-------------------:|---------:|-------|------------|
| **fews** | 548,863 | 0% | market × admin × product × month | `country_code` (ISO2) + admin2 → admin1 → country spine |
| **wfp** | 1,093,209 | 0% | market × country × product × month (Africa-scoped) | Resolved ISO3 (`int_ref_country` + `geo_faostat_country_name_iso3`) → country spine; admin1 by ISO3 |
| **faostat** | 321,816 | 0% | country × item × element × month | `area_code` → `int_geography_conformed.native_id` at country level |

**Field contracts:** `common_unit_price` / `common_currency_price` are **FEWS-only** — null on WFP and FAOSTAT is expected; use `value` + `currency` + `unit`. WFP non-Africa rows (e.g. AFN) are dropped at harmonise. Month must be calendar 1–12 (or null), not FAOSTAT label codes like 7004.

**Geo caveat (all sources):** when `geography_key` is null, `place_scope` may still cite a raw country name — that is citation fallback (`acf_place_scope_coalesce`), not resolved geo. Fake `national` without `geography_key` is blocked via `acf_*_strict`.

### fct_food_security / fct_food_balance / fct_trade / fct_investment / fct_gender_inclusion

src/asof OK. **FAOSTAT country-grain** rows on `fct_trade` / `fct_investment` resolve via `int_faostat_area_iso` → `dim_geography` by ISO3 (0% geo null post 2026-08-28 rebuild). `fct_gender_inclusion` residual **52 rows (0.07%)** — all **Chagos Archipelago** (`area_code` 24), outside Africa `in_africa_scope`; flagged in `ref_faostat_aggregate_areas` / `dim_faostat_area.is_aggregate_area`. FEWS trade border rows 0% geo null.

### Residual geo_null (post 2026-08-28 fix pass)

| Fact / source | geo_null | Cause | Action |
|---------------|----------|-------|--------|
| **fct_germplasm** `crop_germplasm_africa` | ~6.4% | Duplicate of ArcGIS rice germplasm on `objectid`; raw `geography` backfilled from WKT | Residual bad/off-shore WKT |
| **fct_germplasm** ArcGIS rice | ~6.2% | Off-shore / bad WKT | Residual |
| **fct_climate** ERA5 | ~1.1% | Ocean / fringe reanalysis cells | geoBoundaries ADM0 + city-bbox grid fallback |
| **fct_climate** NASA | ~3% | Same point resolver | Residual ocean/fringe |
| **fct_biodiversity** GBIF | ~1.0% | Off-shore occurrences | Residual |
| **fct_climate** NASA | ~3.2% | Same point resolver; smaller due to `country_code` fallback | Residual ocean/fringe |
| **fct_biodiversity** GBIF | ~0.9% | Off-shore occurrences, bad coords | Residual |
| **fct_gender_inclusion** | 52 rows | `area_code` 24 Chagos Archipelago — no Africa spine ISO3 | Expected null; `dim_faostat_area.is_aggregate_area` |

### High asof_null / geo_null (Phase 2–3 — updated)

| Fact | Issue |
|------|--------|
| fct_soil_health iSDA | asof_null 100% (~70M); geo_null **0%** (2026-08-28) |
| fct_soil_health ISRIC | geo_null **0%** |
| fct_animal_health | geo resolved via ILRI country stamp + admin1; asof_null 100% |
| fct_protected_areas | geo_null **0%** |
| fct_germplasm | see residual table above (source null geometry, not join bug) |
| fct_biodiversity | geo_null **~0.9%** (was misreported as 100%) |
| fct_vegetation | geo_null **0%** on ILRI + NDVI |
| fct_air_quality | geo_null **0%** (Nakuru) |
| fct_climate ERA5/NASA | ERA5 ~9%; NASA ~3%; ClimateWatch 0% |
| fct_insurance | geo_null **0%**; asof partial |

### ILRI dairy
Cow-day grain stays **out** of `fct_production` (by design).

## Phase 3 notes (code)

- `int_faostat_area_iso`: FAOSTAT `area_code` → ISO3 (aliases + `stg_geo` name match); unmapped kept null.
- `int_geography_conformed`: country grain from cities (`countries_from_cities`) + FAOSTAT rows enriched via area ISO map.
- `fct_hdi`: joins `dim_geography` on `h.geo_key` (ISO3 → country grain in `int_hdi_conformed`).
- FAOSTAT country facts: native_id join + ISO3 fallback via `int_faostat_area_iso`.
- Point: air/ERA5/NASA/biodiversity/germplasm use city 2dp → `int_latlon_country_grid` (geoBoundaries bbox + city-bbox fallback) → **`int_admin0_africa` bbox fallback** → nearest city ≤100 km → country ISO2 spine.
- `fct_animal_health`: country stamped from ILRI source (KEN/UGA/ETH); `geo_key` via admin1 name match else country spine; `place_scope` from dim + county/village.
- All facts expose `geo_scope` + `as_of_date_basis`.

## Phase 4 notes

- **Split:** `fct_production` = FAOSTAT `country_year` only; `fct_yield` = `yield_raw_data` `fnid_season` (citation SoT — no domain UNION).
- Unit columns: FAOSTAT from row `unit`; yield defaults `ha` / `tonnes` / `t/ha`.
- **Production long-form:** `int_faostat_production_conformed` → `fct_production` with `element_key` + `value` (replaces crops-only wide pivot). `agg_production_country_year` sums physical grain only.
- **Food balance long-form:** `int_faostat_food_balances_conformed` → `fct_food_balance` with `element_key` + `value`. Convenience food/feed/losses/etc. map by **element_code** (5141/5142 food; 5520/5521 feed; 5016/5123 losses)—not English labels. Kcal / kg-capita stay in `value` only.
- Geography: `native_id` join plus `int_faostat_area_iso` → country ISO3 fallback.
- **ILRI dairy (milk/calving) stays out** of both facts — cow-day grain.

## FEWS food security measure contract

- **classification:** IPC phase is source `value` (1–5) → staging maps `phase_code` / `phase_name`; fact `unit` = `IPC phase`; `classification_key` joins `dim_classification`.
- **population:** `value` = person count; `unit` = `persons`; may include `phase_code` from source `phase`.
- **`pct_phase3/4/5`:** present on population product when source fills them; typically **empty on classification** — do not treat null pct columns as a fact bug.

## Expected nulls / grain contracts (RAG)

- **`fct_prices`:** Three sources — **`fews`** (sub-national market prices, `common_unit_price` populated), **`wfp`** (Africa-scoped VAMPIRE; `common_unit_price` null), **`faostat`** (national producer/CPI/deflator/exchange-rate series). Geo paths differ per source in `int_prices_with_geo` (FEWS: ISO2 + admin hierarchy; WFP: ISO3 alias e.g. Swaziland → SWZ; FAOSTAT: `area_code`). Do not blend without `price_source` + `source_key`. When `geography_key` is null, `place_scope` may cite raw country name only — citation fallback, not resolved geo.
- **`fct_insurance`:** Categorical (herd band + `insurance_start_year` as int64). `value`/`unit` always null — do not invent premiums. `household_key` from `dim_household` (food-security ILRI + i4i farmers).
- **`fct_investment`:** Development Flows grain has donor/purpose/item/element; **indicator/sex/institution are ASTI-only** — null on Flows rows is correct, not a join miss.
- **`fct_trade`:** FAOSTAT is in the same fact (`trade_grain = 'faostat_country_year'`). FEWS border rows (`fews_border_month`) intentionally null `item_key`/`element_key` — use `product_name` / `trade_flow`.
- **`fct_vegetation` (ILRI):** No species in source. Fact exposes lifeform structure as categorical strings (`quantity_trees|shrubs|grass`, `palatability_*` e.g. Low/High density), derived `lifeform_mix` (grass/shrub/tree dominant), plus `carrying_capacity` as `value`.
- **`fct_protected_areas`:** Point resolve uses 5° nearest-city bbox; null geo keeps null scope (`acf_*_strict`), place_scope can cite protected area name.
- **`fct_germplasm`:** `crop_germplasm_africa` shares **100% `objectid` overlap** with `arcgis_layer_rice_germplasm_in_africa_3d2a9` (ArcGIS rice germplasm export). Raw `geography` was null; staging coalesces WKT from ArcGIS. Residual null geo = bad/off-shore WKT only.
- **`fct_climate` (ERA5):** Residual null geo on open-ocean grid cells is expected for `climate_grain=point_obs`; land cells should resolve via geoBoundaries ADM0 in `int_admin0_africa`.
- **FAOSTAT expected-null geo** (`ref_faostat_aggregate_areas` → `dim_faostat_area.is_aggregate_area`): **`area_code` 24 — Chagos Archipelago** (outside Africa scope). Regional aggregates (World, Africa, …) also stay null when present in other facts.

## Nearest-city point geo (post P0)

**Root cause of ILRI rubbish:** exact `round(lat/lon, 2)` city match misses rural points; `int_latlon_country_grid` only has cells that already contain a city — so `geo_key` stayed null while fact faked `geo_scope=community`.

**Fix:** intermediates use exact city → nearest Africa city within 100 km (`ST_DISTANCE`, ±1.2° bbox) → else country `geo_key` from nearest city’s ISO2. WKT sources (NDVI, protected, germplasm) derive centroid first.

**Facts:** `geography_key` / `geo_level` / `place_scope` / `geo_scope` come from `dim_geography` join only (no fake national when key null). `country_iso3` prefers dim, else intermediate/FAOSTAT extract via `acf_country_iso3`.

Macros: `geo_africa_cities_cte`, `geo_resolve_point_geo_key`, `geo_centroid_*`, `geo_africa_iso2_*`, `geo_africa_soil_fringe_exclude` in `macros/geo_lookup.sql`.

**Soil:** nearest-city bbox **5°** (default elsewhere stays 1.2°); drop Arabian fringe `lat > 12.5 and lon > 43`; reject resolved non-Africa ISO2; `fct_soil_health` uses `acf_*_strict` (no fake `point` without key).

### Rebuild

```powershell
dbt build --target mart_dev_sa --select int_ilri_vegetation_conformed int_ndvi_conformed int_nakuru_air_conformed int_nasa_power_conformed int_era5_conformed int_biodiversity_conformed int_isric_soil_long int_isric_soil_with_geo int_isda_soil_with_geo int_protected_areas int_germplasm_conformed fct_vegetation fct_air_quality fct_climate fct_biodiversity fct_soil_health fct_protected_areas fct_germplasm
```

Soil-only refresh:

```powershell
dbt build --target mart_dev_sa --select int_isric_soil_long int_isric_soil_with_geo int_isda_soil_with_geo fct_soil_health
```

### Acceptance

```sql
SELECT COUNTIF(geography_key IS NULL) / COUNT(*) AS geo_null_pct,
       COUNTIF(country_iso3 IS NULL) / COUNT(*) AS iso_null_pct
FROM mart_dev.fct_vegetation
WHERE vegetation_grain = 'ilri_site';

SELECT geography_key, geo_level, country_iso3, geo_scope, place_scope, latitude, longitude
FROM mart_dev.fct_vegetation
WHERE vegetation_grain = 'ilri_site'
LIMIT 20;

-- Country ISO may appear without key; scope must not
SELECT COUNT(*) AS bad_scope_without_key
FROM mart_dev.fct_vegetation
WHERE geography_key IS NULL AND geo_scope IS NOT NULL;

-- Soil: no fake scope; fringe gone; matched keys usually have ISO
SELECT
  COUNTIF(geography_key IS NULL) / COUNT(*) AS geo_null_pct,
  COUNTIF(geography_key IS NULL AND geo_scope IS NOT NULL) AS fake_scope,
  COUNTIF(geography_key IS NOT NULL AND country_iso3 IS NULL) AS key_without_iso
FROM mart_dev.fct_soil_health
WHERE source_natural_key = 'ISRIC_Africa_Soil';
```

Expect ILRI: non-null keys, `country_iso3` ≈ `KEN` near 0.6°N / 37°E, non-empty `place_scope`.

## Geo spine + dim-only facts (post nearest-city)

**Contract:** spine (`int_geography_conformed` → `dim_geography`) first; intermediates attach `geo_key` (and optional extractable ISO); facts take `geo_level` / `place_scope` / `data_level` / `geo_scope` from dim only. `country_iso3` = `acf_country_iso3(dim, fallback)` — may be present without `geography_key`. No fake `national` when key is null (`acf_row_data_level_strict` / `acf_geo_scope_strict`).

**WKT:** `SAFE` only on `ST_GEOGFROMTEXT`; centroids via `ST_Y(ST_CENTROID(...))`. Germplasm coalesces GEOGRAPHY column then WKT.

**Spine:** `countries_from_cities` + `countries_from_faostat_iso` (territories e.g. MYT/YT, ATF/TF). After `geo_key` hash, backfill `country_iso3` from ISO2 via cities map (keys stay stable). Employment emits ISO on int only when `geo_key` matched; facts still coalesce dim + int.

**Production:** `fct_production` = FAOSTAT only; `fct_yield` = `yield_raw_data` (`ha` / `tonnes` / `t/ha`). Do not union for citation.

## Fill `country_iso3` (spine + facts)

### Rebuild

```powershell
dbt build --target mart_dev_sa --select int_geography_conformed dim_geography int_climatewatch_conformed int_fews_food_security_with_geo int_ilri_household_conformed int_yield_raw_with_geo fct_climate fct_food_security fct_household fct_yield fct_insurance
```

`dim_geography+` alone does **not** rebuild upstream ints that newly gained `country_iso3` — include those ints explicitly (or rebuild them before the facts). For a full downstream refresh after spine/dim: `int_geography_conformed dim_geography+` plus the four ints above.

### Acceptance

```sql
-- Spine: FNID/admin with ISO2 should have ISO3
SELECT COUNT(*) FROM mart_dev.dim_geography
WHERE country_iso2 IS NOT NULL AND country_iso3 IS NULL;

-- Facts: matched geo should rarely lack ISO
SELECT COUNTIF(geography_key IS NOT NULL AND country_iso3 IS NULL)
FROM mart_dev.fct_yield;

-- Scope still null when key null (ISO alone is OK)
SELECT COUNT(*) FROM mart_dev.fct_employment
WHERE geography_key IS NULL AND geo_scope = 'national';
```

## FAOSTAT territory geo (e.g. French Southern Territories)

**Root cause:** FAOSTAT country facts join `int_faostat_area_iso` → `dim_geography` (native_id + ISO3). Area names absent from `stg_geo` and without a manual alias (e.g. French Southern Territories → ATF) leave `iso.country_iso3` null → no `geography_key` / `country_iso3` on facts like `fct_land_inputs`.

**Fix:** territory aliases in `int_faostat_area_iso` (incl. ATF/TF); `TF` on Africa ISO2 allowlist; FAOSTAT dual-join facts pass `iso.country_iso3` + row `country_name` into `acf_place_scope_coalesce` for citation when dim join lags.

### Rebuild

```powershell
dbt build --target mart_dev_sa --select int_faostat_area_iso int_geography_conformed dim_geography fct_land_inputs fct_production fct_emissions fct_food_balance fct_forestry fct_humanitarian fct_investment fct_gender_inclusion fct_machinery fct_trade
```

Land inputs only:

```powershell
dbt build --target mart_dev_sa --select int_faostat_area_iso int_geography_conformed dim_geography fct_land_inputs
```

### Acceptance

```sql
SELECT area_code, country_name, country_iso3
FROM mart_dev.int_faostat_area_iso
WHERE lower(country_name) LIKE '%french southern%';

SELECT
  COUNTIF(country_name LIKE 'French Southern%' AND country_iso3 IS NULL) AS fst_iso_null,
  COUNTIF(geography_key IS NULL AND geo_scope IS NOT NULL) AS fake_scope
FROM mart_dev.fct_land_inputs;

-- Remaining unmapped FAOSTAT areas (add aliases as needed)
SELECT country_name, COUNT(*) AS n
FROM mart_dev.int_faostat_area_iso
WHERE country_iso3 IS NULL
GROUP BY 1 ORDER BY n DESC LIMIT 20;
```

### Rebuild (earlier geo spine select)

```powershell
dbt build --target mart_dev_sa --select int_faostat_area_iso int_geography_conformed dim_geography int_employment_conformed int_ndvi_conformed int_protected_areas int_germplasm_conformed fct_employment fct_hdi fct_production fct_emissions fct_food_balance fct_investment fct_gender_inclusion fct_forestry fct_humanitarian fct_land_inputs fct_machinery fct_vegetation fct_protected_areas fct_germplasm
```

### Acceptance (earlier)

```sql
-- Employment: no fake national without key (ISO without key is allowed)
SELECT COUNT(*) FROM mart_dev.fct_employment
WHERE geography_key IS NULL AND geo_scope = 'national';

-- Production sources (FAOSTAT only)
SELECT source_natural_key, COUNT(*)
FROM mart_dev.fct_production GROUP BY 1;

-- Yield sources
SELECT source_natural_key, COUNT(*)
FROM mart_dev.fct_yield GROUP BY 1;

-- Yield units non-null
SELECT COUNTIF(area_unit IS NULL) FROM mart_dev.fct_yield;
```

**Production split (citation SoT):** `fct_production` has only `Unpivoted_FAOstat_africa_production_*` keys; `fct_yield` has only `yield_raw_data`. No domain UNION.

## Production / yield split

### Rebuild

```powershell
dbt build --target mart_dev_sa --select fct_production fct_yield agg_production_country_year agg_production_country_season
```

### Acceptance

```sql
SELECT source_natural_key, COUNT(*) FROM mart_dev.fct_production GROUP BY 1;
-- expect only Unpivoted_FAOstat_africa_production_* (3 keys)

SELECT source_natural_key, COUNT(*) FROM mart_dev.fct_yield GROUP BY 1;
-- expect only yield_raw_data
```

## Orphan geography FKs after spine rebuild

Rebuilding `dim_geography` changes `geography_key` hashes when ISO3 is filled on country grain. Facts not rebuilt keep orphan keys → relationship tests fail.

**Fix:** rebuild intermediates that stamp `geo_key`, then facts; dim-only `acf_*_strict` on economics/household/trade.

### Rebuild

```powershell
dbt build --target mart_dev_sa --select int_faostat_area_iso int_faostat_macro_conformed int_gdp_ppp_conformed int_ilri_household_conformed fct_economics fct_household fct_trade
```

### Acceptance

```sql
-- Orphan keys must be 0 (null geography_key is OK)
SELECT 'household' AS f, COUNT(*) FROM mart_dev.fct_household h
LEFT JOIN mart_dev.dim_geography d ON d.geography_key = h.geography_key
WHERE h.geography_key IS NOT NULL AND d.geography_key IS NULL
UNION ALL
SELECT 'economics', COUNT(*) FROM mart_dev.fct_economics e
LEFT JOIN mart_dev.dim_geography d ON d.geography_key = e.geography_key
WHERE e.geography_key IS NOT NULL AND d.geography_key IS NULL
UNION ALL
SELECT 'trade', COUNT(*) FROM mart_dev.fct_trade t
LEFT JOIN mart_dev.dim_geography d ON d.geography_key = t.geography_key
WHERE t.geography_key IS NOT NULL AND d.geography_key IS NULL;
```

## Geo architecture (reference layer)

**Root cause:** Geography was assembled from per-source UNIONs and ad-hoc matchers (city names, nearest city, manual aliases)—not a canonical place registry.

**Layers added:**

| Layer | Model | Role |
|-------|--------|------|
| Seed | `ref_m49_country` | UN M49 → ISO2/ISO3 + `in_africa_scope` (55 countries + TF/YT/RE/EH/SH) |
| Reference | `int_ref_country` / `dim_ref_country` | Canonical country list; replaces hardcoded ISO2 allowlists |
| FAOSTAT | `int_faostat_areas_distinct` → `int_faostat_area_reference` → `int_faostat_area_iso` / `dim_faostat_area` | area_code + M49 → ISO; `match_method` logged |
| Point grid | `int_admin0_africa` → `int_latlon_country_grid` | 0.1° cells via `ST_CONTAINS` on **geoBoundaries ADM0** polygons (`raw_dev.geoboundaries_admin0_africa`) |
| Spine | `int_geography_conformed` | FEWS/yield ISO3 from ref country; FAOSTAT `native_id = area_code` |
| Resolver | `macros/geo_resolve.sql` | `geo_latlon_grid_cte`, `geo_faostat_area_cte`, coalesce helpers |

**Point resolution order:** exact city → nearest city (100 km) → lat/lon grid country → dim country by ISO2.

**Scope filter:** `geo_africa_iso2_in()` now subqueries `ref_m49_country.in_africa_scope` (not a static list).

### Rebuild order (two steps + orphan refresh)

Relationship tests on facts pointing at `dim_geography` run whenever `dim_geography` is in a `dbt build` select—even if those facts were not rebuilt. Either rebuild the affected facts in the same run or expect orphan-key failures until you do.

**Step A — reference + grid + spine (fast)**

```powershell
dbt seed --target mart_dev_sa
dbt build --target mart_dev_sa --select ref_m49_country int_faostat_areas_distinct int_ref_country int_faostat_area_reference int_faostat_area_iso int_admin0_africa int_latlon_country_grid int_geography_conformed dim_ref_country dim_faostat_area dim_geography
```

**Step B — point intermediates + spatial/climate/soil facts (slow; iSDA ~35 min)**

```powershell
dbt build --target mart_dev_sa --select int_isric_soil_long int_isric_soil_with_geo int_isda_soil_with_geo int_era5_conformed int_nasa_power_conformed int_nakuru_air_conformed int_ndvi_conformed int_ilri_vegetation_conformed int_biodiversity_conformed int_germplasm_conformed int_protected_areas fct_soil_health fct_climate fct_air_quality fct_vegetation fct_biodiversity
```

**Step C — refresh facts with stale `geography_key` after spine rebuild**

```powershell
dbt build --target mart_dev_sa --select int_geography_conformed dim_geography int_climatewatch_conformed int_fews_food_security_with_geo int_ilri_household_conformed int_yield_raw_with_geo int_faostat_macro_conformed int_employment_conformed fct_economics fct_employment fct_food_security fct_production fct_trade fct_yield fct_household fct_climate fct_insurance
```

### Rebuild (full geo stack — single command, long)

```powershell
dbt seed --target mart_dev_sa
dbt build --target mart_dev_sa --select ref_m49_country int_faostat_areas_distinct int_ref_country int_faostat_area_reference int_faostat_area_iso int_admin0_africa int_latlon_country_grid int_geography_conformed dim_ref_country dim_faostat_area dim_geography int_isric_soil_with_geo int_isda_soil_with_geo int_era5_conformed int_nasa_power_conformed int_nakuru_air_conformed int_ndvi_conformed int_ilri_vegetation_conformed int_biodiversity_conformed int_germplasm_conformed int_protected_areas fct_soil_health fct_land_inputs+
```

### Acceptance

```sql
-- French Southern Territories (FAOSTAT area) must resolve
SELECT area_code, country_iso3, match_method
FROM intermediate_dev.int_faostat_area_reference
WHERE lower(country_name) LIKE '%french southern%';
-- expect ATF, match_method in (m49, alias)

-- Grid-backed point geo (ISRIC sample)
SELECT COUNT(*) AS total,
  COUNTIF(country_iso3 IS NOT NULL) AS with_iso3
FROM intermediate_dev.int_isric_soil_with_geo
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
```

**Note:** `int_admin0_africa` uses **geoBoundaries ADM0** polygons (loaded by `data/local/scripts/load_geoboundaries_admin0.py`). Replaces interim city-bbox envelopes from `stg_geo`.

---

## Mart column label dictionaries (RAG BQ reasoner)

Live distinct values for **every column** on each `mart_dev` table (cap 500 labels per column; stats-only above that).

**Run sequence** (after mart build):

```powershell
cd data-eng/dbt
dbt build --target mart_dev_sa --select mart_dev.*

cd ../..
python data-eng/data/local/scripts/regenerate_mart_table_yamls.py
```

**Outputs**

| Artifact | Path |
|----------|------|
| Per-table YAML (parallel to staging) | [`ml-eng/ml/rag/bq_mart_tables_yaml_files/`](../../ml-eng/ml/rag/bq_mart_tables_yaml_files/) |
| BigQuery audit table | `{BQ_PROJECT}.mart_dev.audit_mart_column_labels` |
| Human review (Markdown) | [`_mart_column_labels_raw.md`](./_mart_column_labels_raw.md) |

**dbt source:** `mart_dev_audit.audit_mart_column_labels` in [`dbt/models/mart_dev/sources_audit.yml`](../dbt/models/mart_dev/sources_audit.yml) (populated by script, not `dbt run`).

**RAG loader:** [`bq_table_schema_yaml.py`](../../ml-eng/ml/rag/chatbot/bq_table_schema_yaml.py) — `load_mart_table_schema()`, `list_mart_table_index()`, `pack_mart_table_hints()`; merge curated semantics after regen via [`patch_mart_yaml_semantics.py`](../../ml-eng/ml/rag/helpers/patch_mart_yaml_semantics.py).
