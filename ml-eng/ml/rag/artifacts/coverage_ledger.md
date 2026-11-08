# Coverage ledger summary

Snapshot `0c0b39aca16d827b`, coverage built 2026-11-04T21:31:28.623549+00:00.

| Class | Readiness | Default table | Default readiness | Serving tables |
|---|---|---|---|---|
| AH | fixture_verified | fct_animal_health | fixture_verified | 3 |
| BIO | fixture_verified | fct_biodiversity | fixture_verified | 3 |
| CLIM | fixture_verified | fct_climate | fixture_verified | 1 |
| EL | fixture_verified | fct_employment | fixture_verified | 3 |
| ENV | fixture_verified | fct_emissions | fixture_verified | 3 |
| FS | fixture_verified | fct_food_security | fixture_verified | 6 |
| FVC | fixture_verified | fct_food_balance | fixture_verified | 5 |
| GYI | fixture_verified | fct_gender_inclusion | fixture_verified | 3 |
| HDI | fixture_verified | fct_hdi | fixture_verified | 3 |
| INP | fixture_verified | fct_land_inputs | fixture_verified | 4 |
| PRC | fixture_verified | fct_prices | fixture_verified | 2 |
| PROD | fixture_verified | agg_production_country_year | fixture_verified | 5 |
| RES | fixture_verified | fct_investment | fixture_verified | 3 |
| SOIL | fixture_verified | fct_soil_health | fixture_verified | 4 |
| VEG | fixture_verified | fct_vegetation | fixture_verified | 1 |

## Blocked serving tables

- none

## Unsupported mart objects

- `agg_economics_country_year`: no indicator class lists or labels this table
- `agg_employment_country_year`: no indicator class lists or labels this table
- `agg_food_balance_country_year`: no indicator class lists or labels this table
- `agg_forestry_country_year`: no indicator class lists or labels this table
- `fct_air_quality`: no indicator class lists or labels this table
- `fct_land_use`: no indicator class lists or labels this table
- `fct_market_access`: no indicator class lists or labels this table
- `fct_population_dynamics`: no indicator class lists or labels this table
- `fct_technology`: no indicator class lists or labels this table
- `fct_value_chain`: no indicator class lists or labels this table

## Contract disagreements

- {"kind": "class_membership", "table": "agg_emissions_country_year", "table_yaml": ["ENV"], "taxonomy": []}
- {"kind": "class_membership", "table": "agg_food_security_country_month", "table_yaml": ["FS"], "taxonomy": []}
- {"kind": "class_membership", "table": "agg_production_country_season", "table_yaml": [], "taxonomy": ["PROD"]}
- {"kind": "class_membership", "table": "fct_biodiversity", "table_yaml": [], "taxonomy": ["BIO"]}
- {"kind": "class_membership", "table": "fct_fertilizer", "table_yaml": [], "taxonomy": ["INP"]}
- {"kind": "class_membership", "table": "fct_forestry", "table_yaml": ["ENV"], "taxonomy": ["FVC"]}
- {"kind": "class_membership", "table": "fct_gender_inclusion", "table_yaml": [], "taxonomy": ["GYI"]}
- {"kind": "class_membership", "table": "fct_germplasm", "table_yaml": [], "taxonomy": ["BIO"]}
- {"kind": "class_membership", "table": "fct_humanitarian", "table_yaml": [], "taxonomy": ["FS"]}
- {"kind": "class_membership", "table": "fct_investment", "table_yaml": [], "taxonomy": ["RES"]}
- {"kind": "class_membership", "table": "fct_land_degradation", "table_yaml": [], "taxonomy": ["SOIL"]}
- {"kind": "class_membership", "table": "fct_pesticide", "table_yaml": [], "taxonomy": ["INP"]}
- {"kind": "class_membership", "table": "fct_protected_areas", "table_yaml": [], "taxonomy": ["BIO"]}
- {"kind": "class_membership", "table": "fct_research_expenditure", "table_yaml": [], "taxonomy": ["RES"]}
- {"kind": "class_membership", "table": "fct_researchers", "table_yaml": [], "taxonomy": ["RES"]}
- {"kind": "class_membership", "table": "fct_vegetation", "table_yaml": [], "taxonomy": ["VEG"]}
