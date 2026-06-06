# OpenTrace data modeling & transformation guide

Use this document to explain your work to the team. It has **two parts**:

| Part | Topic | Best for |
|------|--------|----------|
| **Part 1** | Facts & dimension tables (star schema) | Country/year dashboards, FAO, climate, production, humanitarian |
| **Part 2** | `soil_dashboard_base` (ISDA + ISRIC) | Map-based soil dashboard in Power BI |

**dbt folder:** `dbt/models/silver/`  
**BigQuery project:** `opentrace-prod-5ga4`  
**Silver / staging dataset (team convention):** `staging_dev` (set via `BQ_DATASET_SILVER=staging_dev`)

---

# Part 1: How facts and dimension tables were created (star schema)

## 1.1 What you built and why

You implemented a **dimensional model (star schema)** in dbt so Power BI and analysts can:

- Filter by **country**, **year**, **indicator**, **crop**, etc. (dimensions)
- Measure **values** in separate **fact** tables by business area (climate, land use, production, …)
- Reuse the **same surrogate keys** across facts and dimensions (joins stay consistent)

This is **separate from** `soil_dashboard_base` (Part 2). The star schema is **country / admin-region / year** grain. The soil table is **latitude / longitude** grid grain.

---

## 1.2 Star schema architecture (build order)

Always build in this order — later models depend on earlier ones:

```
Silver source tables (FAO, NASA POWER, yield, FEWS NET, …)
                    |
                    v
        stg_silver_star_metrics   ← VIEW: one canonical column layout
                    |
        +-----------+-----------+
        |                       |
        v                       v
   dim_* tables            fact_* tables
   (14 dimensions)        (10 facts, filtered by domain_name)
        |                       |
        +-----------+-----------+
                    v
           Power BI: relate facts to dims on *_key columns
```

**Files:** `stg_silver_star_metrics.sql` → `dim_*.sql` + `fact_*.sql` → documented in `star_schema.yml`

---

## 1.3 Step 1 — Canonical staging: `stg_silver_star_metrics`

**File:** `dbt/models/silver/stg_silver_star_metrics.sql`  
**Materialization:** `view` (not a physical table; rebuilt each run)  
**Role:** Unify many different silver sources into **one standard shape** before building dims/facts.

### Standard columns (every domain uses the same layout)

| Column | Purpose |
|--------|---------|
| `source_name` | Which upstream table (e.g. `fao_rl_landuse`, `yield_raw_data_silver`) |
| `domain_name` | Business area — **used to split facts** (`climate`, `land_use`, `production`, …) |
| `indicator_name` | Metric name |
| `country_code`, `country_name` | Country |
| `admin_region` | Sub-national region when available |
| `period_year`, `period_date` | Time |
| `unit_name` | Unit of measure |
| `crop_name`, `season_name`, `practice_name`, … | Domain-specific attributes (often NULL) |
| `metric_value` | The numeric measure |

### How each domain is loaded (UNION ALL)

Inside the view, one CTE per domain selects from a silver source and maps columns into the standard layout. Examples:

| `domain_name` | Silver source (`source('silver', …)`) | What it represents |
|---------------|----------------------------------------|---------------------|
| `climate` | `africa_nasa_power_silver` | Solar / UV metrics (unpivoted with `union all`) |
| `land_use` | `fao_rl_landuse` | FAO land-use statistics |
| `enterprise_investment` | `fao_qi` | FAO investment |
| `technology` | `fao_ti_trade_indices` | Trade / technology indices |
| `market_access` | `fao_qv` | Market access |
| `production` | `yield_raw_data_silver` | Area, production, yield (unpivoted) |
| `policy` | `fao_rp_pesticides` | Pesticide / policy metrics |
| `nutrition` | `fao_rhn` | Nutrition |
| `value_chain` | `fao_fbs` | Food balance / value chain |
| `humanitarian` | `fews_net_food_security_master_silver` | IPC food security phases |

Final step in the view:

```sql
select * from climate
union all select * from land_use
union all …
union all select * from humanitarian
```

**Why this step matters for your presentation:** You did not build 24 unrelated models. You built **one staging contract**, then **thin** dimension and fact models on top.

---

## 1.4 Step 2 — Dimension tables (14 tables)

**Pattern (same for every `dim_*` model):**

1. `select distinct` from `{{ ref('stg_silver_star_metrics') }}`
2. Build a **surrogate key** with `to_hex(md5(...))` from natural attributes
3. Keep human-readable columns (names, codes)
4. `materialized='table'` → physical table in `staging_dev`

### Example — `dim_country`

```sql
select distinct
    to_hex(md5(concat(coalesce(country_code, ''), '|', coalesce(country_name, '')))) as country_key,
    country_code,
    country_name
from {{ ref('stg_silver_star_metrics') }}
where country_code is not null or country_name is not null
```

### Example — `dim_indicator`

```sql
select distinct
    to_hex(md5(coalesce(indicator_name, ''))) as indicator_key,
    indicator_name
from {{ ref('stg_silver_star_metrics') }}
where indicator_name is not null
```

### Example — `dim_geography` (country + admin region)

Key = MD5 of `country_code | country_name | admin_region`.

### Example — `dim_period` (year grain)

Uses `period_year` directly as `period_key` (integer year), plus `start_date` for reporting.

### Full dimension inventory

| dbt model | Key column | Natural attributes hashed / stored |
|-----------|------------|-------------------------------------|
| `dim_country` | `country_key` | `country_code`, `country_name` |
| `dim_period` | `period_key` | `period_year` |
| `dim_geography` | `geography_key` | country + `admin_region` |
| `dim_indicator` | `indicator_key` | `indicator_name` |
| `dim_source` | `source_key` | `source_name` |
| `dim_unit` | `unit_key` | `unit_name` |
| `dim_crop` | `crop_key` | `crop_name` |
| `dim_value_chain` | `value_chain_stage_key` | `value_chain_stage_name` |
| `dim_season` | `season_key` | `season_name` |
| `dim_policy_area` | `policy_area_key` | `policy_area_name` |
| `dim_technology` | `technology_key` | `technology_name` |
| `dim_farm_practice` | `practice_key` | `practice_name` |
| `dim_shock_type` | `shock_type_key` | `shock_type_name` |
| `dim_ipc_phase` | `ipc_phase_key` | `ipc_phase_name`, `ipc_phase_numeric` |

**Tests (in `star_schema.yml`):** each `*_key` has `not_null` and `unique`.

---

## 1.5 Step 3 — Fact tables (10 tables)

**Pattern (same idea for every `fact_*` model):**

1. Read from `stg_silver_star_metrics`
2. **Filter** `where domain_name = '<domain>'` (only rows for that subject area)
3. Create **`fact_<domain>_key`** — MD5 of the grain columns that make a row unique
4. Add **foreign keys** — same MD5 formulas as in the matching `dim_*` models
5. Expose one **measure column** (e.g. `land_use_value`, `climate_value`)

### Example — `fact_land_use`

```sql
select
    to_hex(md5(concat(
        coalesce(source_name, ''), '|',
        coalesce(indicator_name, ''), '|',
        coalesce(country_code, ''), '|',
        coalesce(cast(period_year as string), ''), '|',
        coalesce(crop_name, '')
    ))) as fact_land_use_key,
    to_hex(md5(concat(coalesce(country_code, ''), '|', coalesce(country_name, '')))) as country_key,
    cast(period_year as int64) as period_key,
    … geography_key, indicator_key, source_key, unit_key, crop_key …
    metric_value as land_use_value
from {{ ref('stg_silver_star_metrics') }}
where domain_name = 'land_use'
```

### Example — `fact_production` (extra dimensions)

Grain includes `crop_name` and `season_name`; also outputs `season_key` and `practice_key`.

### Example — `fact_humanitarian` (IPC / shocks)

Uses `shock_type_key`, `ipc_phase_key`; measure = `humanitarian_value`.

### Full fact inventory

| Fact table | `domain_name` filter | Measure column | Typical dimension keys on the fact |
|------------|----------------------|----------------|-------------------------------------|
| `fact_climate` | `climate` | `climate_value` | country, period, geography, indicator, source, unit |
| `fact_land_use` | `land_use` | `land_use_value` | + crop |
| `fact_enterprise_investment` | `enterprise_investment` | `enterprise_investment_value` | + crop |
| `fact_technology` | `technology` | `technology_value` | + crop, technology |
| `fact_market_access` | `market_access` | `market_access_value` | + crop |
| `fact_production` | `production` | `production_value` | + crop, season, practice |
| `fact_policy` | `policy` | `policy_value` | + policy_area |
| `fact_nutrition` | `nutrition` | `nutrition_value` | + crop |
| `fact_value_chain` | `value_chain` | `value_chain_value` | + value_chain_stage |
| `fact_humanitarian` | `humanitarian` | `humanitarian_value` | + shock_type, ipc_phase |

**Important:** Facts store **keys only**, not country names. Names live in dimensions — classic star schema for Power BI.

---

## 1.6 Surrogate keys — rules you used (explain to the team)

| Rule | Detail |
|------|--------|
| Algorithm | `to_hex(md5(concat(...)))` in BigQuery |
| Separator | Pipe `|` between fields in `concat` |
| Nulls | `coalesce(column, '')` so NULLs hash consistently |
| Same key everywhere | Fact `country_key` uses **identical** MD5 expression as `dim_country.country_key` |
| Grain | Fact primary key hashes the columns that define **one fact row** (source + indicator + country + year + …) |

If the same natural combination appears twice in staging, you get the same key — dimensions stay stable across refreshes.

---

## 1.7 How to build star schema tables in BigQuery (dbt)

```bash
cd dbt
set BQ_PROJECT=opentrace-prod-5ga4
set BQ_DATASET_SILVER=staging_dev

# 1) Staging view (required first)
dbt run --target silver --select stg_silver_star_metrics

# 2) All dimensions
dbt run --target silver --select dim_country dim_period dim_geography dim_indicator dim_source dim_unit dim_crop dim_value_chain dim_season dim_policy_area dim_technology dim_farm_practice dim_shock_type dim_ipc_phase

# 3) All facts
dbt run --target silver --select fact_climate fact_land_use fact_enterprise_investment fact_technology fact_market_access fact_production fact_policy fact_nutrition fact_value_chain fact_humanitarian

# Or everything star-related in one go:
dbt run --target silver --select stg_silver_star_metrics dim_* fact_*
```

**Output location:** `opentrace-prod-5ga4.staging_dev.<model_name>` (one table per `dim_*` / `fact_*` model).

**Prerequisite:** Silver source tables referenced in `stg_silver_star_metrics` must exist in BigQuery (FAO, NASA POWER, yield, FEWS NET, etc.). If a source is missing, that domain’s CTE returns no rows until the pipeline loads it.

---

## 1.8 Power BI — how to use facts and dimensions

1. Import fact table(s) you need (e.g. `fact_land_use`, `fact_production`).
2. Import related `dim_*` tables.
3. Create relationships in the model view:
   - `fact_land_use.country_key` → `dim_country.country_key`
   - `fact_land_use.period_key` → `dim_period.period_key`
   - (same pattern for `indicator_key`, `source_key`, etc.)
4. Use dimension fields on **slicers** and axes; use `*_value` columns in **measures** (SUM, AVG).

---

## 1.9 One-minute script — facts & dimensions (for presenting)

> “First I built a staging view, `stg_silver_star_metrics`, that unions all our silver sources — FAO, NASA climate, yield, FEWS NET — into one standard set of columns with a `domain_name` for each subject area. From that single staging layer I created **14 dimension tables** with distinct values and MD5 surrogate keys, and **10 fact tables** filtered by domain. Each fact row has foreign keys that match the dimensions and one numeric measure column. That’s a classic star schema in `staging_dev`, ready for Power BI relationships. Separately, for the soil map, I built `soil_dashboard_base` at lat/lon grid — that’s Part 2 of this doc.”

---

## 1.10 Code files to show when presenting Part 1

| Order | File | What to highlight |
|-------|------|-------------------|
| 1 | `stg_silver_star_metrics.sql` | One CTE per domain + final `union all` |
| 2 | `dim_country.sql` | Simple `distinct` + MD5 key pattern |
| 3 | `fact_land_use.sql` | `where domain_name = 'land_use'` + keys + measure |
| 4 | `star_schema.yml` | Tests and descriptions for all models |

---

# Part 2: Soil dashboard (`soil_dashboard_base` — ISDA + ISRIC)

**Production SQL:** `dbt/models/silver/soil_dashboard_base.sql`  
**BigQuery output:** `opentrace-prod-5ga4.staging_dev.soil_dashboard_base`  
**Deploy:** `dbt run --target silver --select soil_dashboard_base` (with `BQ_DATASET_SILVER=staging_dev`)

---

## 2.1 What problem this solves

The team asked to combine:

| Source | Dataset | Table | Role |
|--------|---------|-------|------|
| ISDA soil property | `landing` | `isda_soil_bulk_master` | Soil properties (carbon, nitrogen, pH, etc.) at many points |
| ISRIC Africa soil | `landing` | `isric_africa_soil_data` | Soil chemistry/texture at many points |

**Goal:** One **dashboard-ready** table with location (`lat`, `lon`) and soil metrics from both sources, written to the **existing** staging layer dataset **`staging_dev`** (team naming convention for “silver”).

**Important:** We do **not** create new BigQuery datasets. We only create **one new table** inside `staging_dev`.

---

## 2.2 End-to-end flow (simple diagram)

```
landing.isda_soil_bulk_master          landing.isric_africa_soil_data
         |                                        |
         |  (1) Convert ISDA x/y meters           |  (3) Already WGS84 lat/lon
         |      → WGS84 lat/lon                   |      round to grid
         |  (2) Aggregate by grid                 |  (4) Aggregate by grid
         v                                        v
              isda_agg (lat_2, lon_2, metrics)     isric_agg (lat_2, lon_2, metrics)
                              \                  /
                               \                /
                                v              v
                         FULL OUTER JOIN on (lat_2, lon_2)
                                        |
                                        v
                    staging_dev.soil_dashboard_base  ← Power BI / maps
```

---

## 2.3 Why a “geocoding” step was needed (ISDA)

### 2.3.1 What we discovered

- **ISRIC** stores real geographic coordinates: latitude/longitude in **degrees** (e.g. -35 to 37.5).
- **ISDA** stores coordinates that look like `latitude` / `longitude` but are actually **projected coordinates in meters** (Web Mercator–style range, roughly ±20,037,508 m).

If you join ISDA and ISRIC **without conversion**, almost **no rows match** on lat/lon.

### 2.3.2 What we do (Web Mercator → WGS84)

For each ISDA row we treat:

- `longitude` column → **x** (meters east)
- `latitude` column → **y** (meters north)

Constants:

- Earth radius **R = 6,378,137** meters (standard for Web Mercator)

Formulas used in SQL:

| Output | Formula (concept) | SQL idea |
|--------|-------------------|----------|
| Longitude (degrees) | `lon = (x / R) * (180 / π)` | `longitude / 6378137.0 * 180.0 / 3.141592653589793` |
| Latitude (degrees) | `lat = atan(sinh(y / R)) * (180 / π)` | `atan(sinh(latitude / 6378137.0)) * 180.0 / 3.141592653589793` |

Then we **round to 2 decimal places** → `lat_2`, `lon_2`.

**Why round?** ISDA has millions of points at nearly identical locations. Rounding builds a **grid** (~1.1 km at the equator) so we can aggregate and join reliably.

### 2.3.3 Safety filter on ISDA

We only keep rows where raw x/y are in the valid Web Mercator meter range:

```sql
cast(latitude as float64) between -20037508.34 and 20037508.34
cast(longitude as float64) between -20037508.34 and 20037508.34
```

This drops invalid or non-projected values before conversion.

---

## 2.4 ISDA transformation (step by step)

### Step A — `isda` CTE (convert + clean)

**Input columns from** `landing.isda_soil_bulk_master`:

| Column | Meaning |
|--------|---------|
| `latitude` | Actually **y** in meters (not degrees) |
| `longitude` | Actually **x** in meters (not degrees) |
| `property` | Soil property name (e.g. carbon, nitrogen, ph) |
| `depth` | Depth label |
| `value` | Numeric measurement |

**Output of this step:** One row per original point, with:

- `lat_2`, `lon_2` — WGS84 degrees on a 2-decimal grid
- `property_name` — lowercased property
- `property_value` — numeric value

### Step B — `isda_agg` CTE (aggregate per grid cell)

Group by `(lat_2, lon_2)` and compute:

| Output column | How it is calculated |
|---------------|----------------------|
| `isda_points` | Count of raw points in that cell |
| `isda_avg_value_all` | Average of all property values |
| `isda_avg_carbon` | Average where `property` contains `'carbon'` |
| `isda_avg_nitrogen` | Average where `property` contains `'nitrogen'` |
| `isda_avg_ph` | Average where property is ph-related |

**Why aggregate?** One grid cell may have thousands of ISDA measurements (different depths/properties). The dashboard needs **one row per map location**, not millions of duplicate coordinates.

---

## 2.5 ISRIC transformation (step by step)

ISRIC does **not** need meter-to-degree conversion.

### Step C — `isric` CTE

**Input:** `landing.isric_africa_soil_data`

- `latitude`, `longitude` — already in degrees
- Depth-specific columns: `soc_0_5cm`, `phh2o_0_5cm`, `clay_0_5cm`, `nitrogen_0_5cm`, `sand_0_5cm`, `silt_0_5cm`

We round to the same grid: `lat_2`, `lon_2` (2 decimals).

### Step D — `isric_agg` CTE

Group by `(lat_2, lon_2)` and average:

| Output column | Source column |
|---------------|---------------|
| `isric_points` | count of rows |
| `isric_avg_soc_0_5cm` | `soc_0_5cm` |
| `isric_avg_ph_0_5cm` | `phh2o_0_5cm` |
| `isric_avg_clay_0_5cm` | `clay_0_5cm` |
| `isric_avg_nitrogen_0_5cm` | `nitrogen_0_5cm` |
| `isric_avg_sand_0_5cm` | `sand_0_5cm` |
| `isric_avg_silt_0_5cm` | `silt_0_5cm` |

---

## 2.6 The join (how ISDA and ISRIC come together)

### Join type: **FULL OUTER JOIN**

```sql
from isda_agg i
full outer join isric_agg r
  on i.lat_2 = r.lat_2
 and i.lon_2 = r.lon_2
```

| Situation | What you get |
|-----------|----------------|
| ISDA + ISRIC same grid cell | Both ISDA and ISRIC metrics populated |
| Only ISDA at that cell | ISRIC columns NULL |
| Only ISRIC at that cell | ISDA columns NULL |

**Why FULL OUTER and not INNER?**  
We keep **all** locations from either source. INNER would drop cells that exist on only one side (we had ~870 cells with both; many more with ISDA-only).

### Final location columns

```sql
coalesce(i.lat_2, r.lat_2) as lat_2
coalesce(i.lon_2, r.lon_2) as lon_2
```

Whichever side has the coordinate wins when the other is missing.

---

## 2.7 Output table schema (what each column means)

| Column | Source | Use in dashboard |
|--------|--------|------------------|
| `lat_2` | Join key | Map latitude |
| `lon_2` | Join key | Map longitude |
| `isda_points` | ISDA | Data density / quality |
| `isda_avg_value_all` | ISDA | General soil value |
| `isda_avg_carbon` | ISDA | Carbon indicator |
| `isda_avg_nitrogen` | ISDA | Nitrogen indicator |
| `isda_avg_ph` | ISDA | pH indicator |
| `isric_points` | ISRIC | Data density |
| `isric_avg_soc_0_5cm` | ISRIC | Soil organic carbon (0–5 cm) |
| `isric_avg_ph_0_5cm` | ISRIC | pH (0–5 cm) |
| `isric_avg_clay_0_5cm` | ISRIC | Clay % |
| `isric_avg_nitrogen_0_5cm` | ISRIC | Nitrogen |
| `isric_avg_sand_0_5cm` | ISRIC | Sand % |
| `isric_avg_silt_0_5cm` | ISRIC | Silt % |

**Approximate size after run:** ~2.5 million rows (mostly ISDA grid cells; ISRIC covers fewer points).

---

## 2.8 How this gets to BigQuery (dbt)

| Setting | Value |
|---------|--------|
| Project | `opentrace-prod-5ga4` |
| Target dataset (env) | `BQ_DATASET_SILVER=staging_dev` |
| dbt target | `silver` (folder `models/silver/`) |
| Materialization | `table` (full table rebuild) |

**Command:**

```bash
cd dbt
export BQ_PROJECT=opentrace-prod-5ga4
export BQ_DATASET_SILVER=staging_dev
dbt run --target silver --select soil_dashboard_base
```

**Result:** Table `staging_dev.soil_dashboard_base` in BigQuery.

---

## 2.9 What we did NOT include in this run (by team request)

| Item | Status |
|------|--------|
| FAO land use / fertilizer / pesticide | **Not in this table** — separate model `land_use_soil_property_join`; wait until FAO sources are ready |
| New BigQuery datasets | **Not created** — only `staging_dev` + `landing` |
| Country-level join | **Not in this model** — geo grid only (lat/lon) |

---

## 2.10 How to verify in BigQuery (demo for the team)

**Row count:**

```sql
SELECT COUNT(*) AS row_count
FROM `opentrace-prod-5ga4.staging_dev.soil_dashboard_base`;
```

**Sample with both sources:**

```sql
SELECT
  lat_2,
  lon_2,
  isda_avg_carbon,
  isric_avg_soc_0_5cm,
  isric_avg_ph_0_5cm
FROM `opentrace-prod-5ga4.staging_dev.soil_dashboard_base`
WHERE isda_avg_carbon IS NOT NULL
  AND isric_avg_soc_0_5cm IS NOT NULL
LIMIT 20;
```

**Overlap count (both ISDA + ISRIC at same grid):**

```sql
SELECT COUNT(*) AS overlap_cells
FROM `opentrace-prod-5ga4.staging_dev.soil_dashboard_base`
WHERE isda_avg_carbon IS NOT NULL
  AND isric_avg_soc_0_5cm IS NOT NULL;
```

---

## 2.11 Power BI connection (soil table)

1. Get data → **Google BigQuery**
2. Project: `opentrace-prod-5ga4`
3. Dataset: **`staging_dev`**
4. Table: **`soil_dashboard_base`**
5. Map visual: **Latitude** = `lat_2`, **Longitude** = `lon_2`, color by `isda_avg_carbon` or `isric_avg_soc_0_5cm`

---

## 2.12 One-minute explanation script — soil (for presenting)

> “We read ISDA and ISRIC from the existing **landing** dataset. ISRIC already has latitude and longitude in degrees. ISDA stores coordinates in projected meters, so we convert them to WGS84 using the standard Web Mercator inverse formulas, then round to a two-decimal grid. We aggregate both sources to one row per grid cell, then **full outer join** on latitude and longitude so we keep all locations from either dataset. The result is materialized with dbt into **staging_dev.soil_dashboard_base** — about 2.5 million rows — ready for Power BI maps and charts. We did not wait for FAO; that is a separate follow-up model.”

---

## 2.13 Code reference map (which part of the SQL file to show)

| When presenting… | Show lines in `soil_dashboard_base.sql` |
|------------------|----------------------------------------|
| Geocoding / ISDA fix | Lines 3–21 (`isda` CTE) |
| ISDA metrics | Lines 23–34 (`isda_agg`) |
| ISRIC prep | Lines 36–64 |
| Join | Lines 66–84 |

---

## 2.14 Related GitHub work

- Branch: `feat/staging-dev-soil-dashboard`
- Pull request: **#13** — “Add soil dashboard models under staging_dev layer”
- Files changed: `soil_dashboard_base.sql` (+ optional FAO scaffold model not used in this run)

---

*Document version: Part 1 = star schema (facts/dims); Part 2 = soil dashboard (ISDA geocoding + ISRIC join).*
