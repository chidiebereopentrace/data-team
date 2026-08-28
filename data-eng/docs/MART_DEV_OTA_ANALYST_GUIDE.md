# Mart Dev Database Guide for OTA Insights Analysts

**Prepared by:** OpenTrace Data Team  
**Date:** August 2026  
**Dataset:** `mart_dev` (BigQuery analytics-ready / gold layer)  
**Companion workbook:** [mart_dev_entity_dictionary.xlsx](./mart_dev_entity_dictionary.xlsx)  
**Classification:** Internal — for analysts authoring OTA insights

---

## 1. Purpose of this guide

This document teaches analysts how to use the OpenTrace **`mart_dev`** warehouse when writing **OTA insights** analytical reports (insight / metric / recommendation narratives).

OTA insights are **authored analytical products**. Metrics and claims should be **grounded in `mart_dev`** (and may later be ingested into the Ask ADZA `OTA_insights` Qdrant corpus). This guide is the database playbook for that work.

Use the Excel entity dictionary for per-table purpose, columns, relationships, and ACF fields. Use this guide for **how to think and query**.

---

## 2. How OTA insights relate to mart_dev

| Layer | Role |
|-------|------|
| **`mart_dev`** | Star-schema facts, dimensions, aggregates — source of truth for numbers |
| **OTA report (authored)** | Narrative insight + metric + recommendation grounded in mart queries |
| **Qdrant `OTA_insights`** | Optional vector corpus so Ask ADZA can retrieve authored OTAs |

**Workflow**

1. Frame the OTA question (sourcing, prices, climate, food security, trade, …).
2. Pick fact or aggregate + filters from the recipes below / workbook.
3. Query BigQuery `mart_dev` with grain and `source_key` discipline.
4. Record metric values with units, place, as-of date, and source.
5. Write insight and recommendation lanes; cite ACF-friendly fields (`tier`, `metric`, `source_id`, `as_of_date`, …).

---

## 3. Star-schema overview

```text
                    dim_geography
                    dim_product / dim_item
                    dim_element / dim_unit
                    dim_source / dim_date / dim_season
                           ▲
                           │ *_key joins
                           │
     fct_*  (measures + ACF columns)  ──►  agg_* (country/month rollups)
                           │
                     bridge_* (AEZ, research)
```

**Inventory (63 live models):** 22 dimensions · 27 facts · 11 aggregates · 3 bridges  

Full list: workbook **Entities** sheet.

**Default query pattern**

- Prefer **`agg_*`** for national/monthly rollups in OTA metrics.
- Drop to **`fct_*`** for market, FNID, point, or long-form element detail.
- Always keep **`source_key`** — never silently blend sources.
- Filter **grain columns** (`production_grain`, `trade_grain`, `climate_grain`, …) before charting.

---

## 4. Core dimensions (when to use which)

| Dimension | Use when |
|-----------|----------|
| `dim_geography` | Every geo-aware fact — join on `geography_key`; filter `geo_level` / `data_level` / `country_iso3` |
| `dim_ref_country` | Africa-scope policy (`in_africa_scope`); not a substitute for fact→geography joins |
| `dim_product` | Crops/commodities on production, prices, food balance, many trade rows |
| `dim_item` | FAOSTAT long-form item codes on emissions, employment, land inputs, investment, etc. |
| `dim_element` | Which measure in long-form facts (area harvested vs production qty vs imports, …) |
| `dim_source` | Lineage + ACF tier — required for trustworthy OTAs |
| `dim_date` | Calendar helpers via `date_key` |
| `dim_season` | Seasonal yield / `agg_production_country_season` — **not** the same as FAOSTAT year |
| `dim_market` | Market-level price facts |
| `dim_classification` | FEWS IPC phase labels |
| `dim_unit` | Units / currencies |
| `dim_household` / `dim_livestock` | ILRI household and animal health |
| `dim_organisation` / `dim_person` / `dim_research_project` | Research / ASTI investment network |

**Rule:** Prefer `dim_product` for commodity OTAs; use `dim_item` when the fact only exposes `item_key`.

---

## 5. Fact catalog by domain

### Production & inputs

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_production` | FAOSTAT country–year long-form; filter `production_grain` | National production / area / value |
| `fct_yield` | FNID–season `yield_raw` | Seasonal yield risk |
| `fct_land_inputs` | Filter `input_grain` | Fertilizer / land use intensity |
| `fct_forestry` | Filter `forestry_grain` | Forestry supply |
| `fct_machinery` | Country–year | Mechanisation context |

### Prices & trade

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_prices` | Market × month | Market detail |
| `fct_trade` | Filter `trade_grain` | FAOSTAT trade vs FEWS borders |

### Food security & balance

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_food_security` | FEWS FNID–month; `measure_type` | Early warning |
| `fct_food_balance` | Long-form; map by `element_code` | Supply utilisation |
| `fct_humanitarian` | Country–year | Aid volumes |
| `fct_food_hazards` | Study-level | Food safety |

### Climate, vegetation, air

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_climate` | Filter `climate_grain` | Stress / rainfall / model indicators |
| `fct_emissions` | Country–year | GHG |
| `fct_vegetation` | Filter `vegetation_grain` | NDVI / carrying capacity |
| `fct_air_quality` | Sensor point | Local air quality |

### Soil & spatial

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_soil_health` | Point × property × depth | Prefer `agg_soil_admin` for summaries |
| `fct_protected_areas` / `fct_germplasm` / `fct_biodiversity` | Feature / point | Conservation / genetic resources |

### Socio-economic & household

| Table | Grain | OTA use |
|-------|-------|---------|
| `fct_economics` / `fct_employment` / `fct_hdi` | Country–year | Macro / labour / HDI |
| `fct_gender_inclusion` | National SDG | Do **not** union with household gender |
| `fct_household` / `fct_animal_health` / `fct_insurance` | Community / household | Livelihood & livestock OTAs |
| `fct_investment` | Country–year + org | Ag R&D funding |

---

## 6. Aggregates — when to prefer them

| Aggregate | Prefer for |
|-----------|------------|
| `agg_production_country_year` | Cross-country production OTAs (physical grain) |
| `agg_production_country_season` | Seasonal country rollups from yield |
| `agg_prices_country_month` | National price trends |
| `agg_food_security_country_month` | National FEWS population-in-phase |
| `agg_food_balance_country_year` | National food/feed/losses sums |
| `agg_economics_country_year` / `agg_employment_country_year` / `agg_emissions_country_year` / `agg_forestry_country_year` | Domain country-year averages |
| `agg_hdi_latest` | Latest HDI snapshot (not a time series) |
| `agg_soil_admin` | Admin soil summaries vs millions of points |

Aggregates **still carry `source_key`** — keep it in GROUP BY / filters.

---

## 7. ACF and citation hygiene

Every trustworthy OTA metric should carry (from the fact or join):

- **What:** `metric`, `value`, `unit`
- **Where:** `country_iso3`, `place_scope` / geography attributes
- **When:** `as_of_date` (+ note `as_of_date_basis` if `loaded_at`)
- **Who/source:** `source_key` / `source_id`, `tier`

See workbook sheet **ACF_Contract** and [CATALOG_TO_MART_MAP.md](./CATALOG_TO_MART_MAP.md).

Do not invent units or fill null `common_unit_price` on WFP or FAOSTAT rows (FEWS-only field).

---

## 8. Hard grain rules

1. **`fct_production` ≠ `fct_yield`** — never average across them; state which grain the OTA uses.
2. **`production_grain`:** use `physical` for area/qty aggregates (`agg_production_country_year`).
3. **Price months** must be calendar 1–12 (or null) — not FAOSTAT label codes.
4. **Always group by `source_key`** — no silent source blend.
5. **Filter grain discriminators** before charts: `trade_grain`, `climate_grain`, `vegetation_grain`, `forestry_grain`, `input_grain`, `measure_type`.
6. **Food balance** food/feed/losses map by **element_code**, not English labels alone.
7. **Gender:** national `fct_gender_inclusion` stays separate from household gender scores.
8. **ILRI dairy cow-day** metrics are **out** of production/yield facts by design.
9. **Partition pruning:** filter `as_of_date` with ranges — do not wrap it in `EXTRACT(year …)` if you can avoid it.

---

## 9. OTA report recipes

### 9.1 Cross-country production / sourcing risk

- **Tables:** `agg_production_country_year` + `dim_geography` + `dim_product` + `dim_source`
- **Filters:** products of interest; year window; Africa ISO3 list; keep `source_key`
- **OTA lanes:** metric (tonnes / ha by country-year) + insight (who is expanding/contracting) + recommendation (sourcing watchlist)

### 9.2 Price volatility & market brief

- **Tables:** `agg_prices_country_month` (national) or `fct_prices` + `dim_market` (markets)
- **Filters:** product, months 1–12, countries; **always** filter `price_source` (`fews` | `wfp` | `faostat`) and keep `source_key`
- **Caveat:** sources differ in grain (FEWS sub-national markets vs WFP/FAOSTAT national); do not compare across sources without stating both

### 9.3 Climate stress vs yield

- **Tables:** `fct_climate` (filter `climate_grain`) + `fct_yield` or `agg_production_country_season`
- **Align on:** `country_iso3` + year/season — do not invent a blended yield from FAOSTAT production

### 9.4 Food security early warning

- **Tables:** `fct_food_security` or `agg_food_security_country_month` + `dim_classification`
- **Filters:** `measure_type`; scenario; month window
- **Caveat:** `pct_phase3/4/5` typically on population product only

### 9.5 Trade / border flows

- **Tables:** `fct_trade` filtered by `trade_grain`
- **FEWS:** use `source_country`, `destination_country`, `border_point`
- **FAOSTAT:** country–year element/item long-form

Full recipe table: workbook **Analytical_Recipes**.

---

## 10. SQL patterns

### National production (physical)

```sql
SELECT
  g.country_iso3,
  p.product_name,
  a.year,
  a.area_harvested,
  a.production_qty,
  a.yield_recomputed,
  a.source_key
FROM `opentrace-prod-5ga4.mart_dev.agg_production_country_year` a
JOIN `opentrace-prod-5ga4.mart_dev.dim_geography` g USING (geography_key)
JOIN `opentrace-prod-5ga4.mart_dev.dim_product` p USING (product_key)
WHERE g.country_iso3 IN ('NGA', 'GHA', 'CIV')
  AND LOWER(p.product_name) LIKE '%maize%'
  AND a.year BETWEEN 2019 AND 2024
ORDER BY a.year, g.country_iso3;
```

### Country-month prices

```sql
SELECT
  g.country_iso3,
  pr.product_name,
  a.year,
  a.month,
  a.price_avg,
  a.common_unit_price_avg,
  a.source_key
FROM `opentrace-prod-5ga4.mart_dev.agg_prices_country_month` a
JOIN `opentrace-prod-5ga4.mart_dev.dim_geography` g USING (geography_key)
JOIN `opentrace-prod-5ga4.mart_dev.dim_product` pr USING (product_key)
WHERE g.country_iso3 IN ('ETH', 'KEN')
  AND a.year = 2024
  AND a.month BETWEEN 1 AND 12;
```

### Fact → geography → source (generic)

```sql
SELECT
  f.*,
  g.geo_level,
  g.country_iso3,
  s.source_natural_key,
  s.tier
FROM `opentrace-prod-5ga4.mart_dev.fct_prices` f
LEFT JOIN `opentrace-prod-5ga4.mart_dev.dim_geography` g
  ON f.geography_key = g.geography_key
LEFT JOIN `opentrace-prod-5ga4.mart_dev.dim_source` s
  ON f.source_key = s.source_key
WHERE f.as_of_date BETWEEN '2023-01-01' AND '2024-12-31'
  AND f.country_iso3 = 'ETH';
```

Replace project id if your environment differs; dataset name remains `mart_dev`.

---

## 11. Known caveats (from MART_QA_NOTES)

- **HDI geo** historically problematic — verify `geography_key` / `country_iso3` before citing.
- **Prices:** `fct_prices` has three feeds (`price_source`: `fews`, `wfp`, `faostat`); filter before comparing. `common_unit_price` null on WFP/FAOSTAT is expected (FEWS-only). Current `mart_dev_sa` build: 0% `geography_key` null on all three (post ISO3 alias fix for WFP legacy names).
- **Soil iSDA:** large volume; weak `as_of_date` historically; prefer `agg_soil_admin`.
- **Point facts** (air, some climate, biodiversity, vegetation): check geo coverage; nearest-city resolution applies.
- **Cluster-only facts** (soil, vegetation, household, animal health, protected, germplasm): partition pruning on `as_of_date` may not apply.
- **FEWS food security:** null phase % on classification rows is expected.

Details: [MART_QA_NOTES.md](./MART_QA_NOTES.md).

---

## 12. Appendix — entity list and regeneration

- **Workbook:** [mart_dev_entity_dictionary.xlsx](./mart_dev_entity_dictionary.xlsx)  
  Sheets: Readme, Entities, Columns, Relationships, Analytical_Recipes, ACF_Contract
- **Seed:** [mart_entity_dictionary_seed.yaml](./mart_entity_dictionary_seed.yaml)

Regenerate workbook:

```powershell
cd data-eng
python scripts/build_mart_entity_dictionary.py
```

Regenerate this guide’s DOCX:

```powershell
cd data-eng
python scripts/build_mart_ota_analyst_guide_docx.py
```

**Contact:** contact@opentrace.africa
