# Coverage ledger summary

Snapshot `be0c5bbf9de9f880`, coverage built 2026-11-12T23:28:45.519083+00:00.

| Class | Readiness | Default table | Default readiness | Serving tables |
|---|---|---|---|---|
| AH | integration_verified | fct_animal_health | integration_verified | 3 |
| BIO | integration_verified | fct_biodiversity | integration_verified | 3 |
| CLIM | integration_verified | fct_climate | integration_verified | 1 |
| EL | integration_verified | fct_employment | integration_verified | 6 |
| ENV | integration_verified | fct_emissions | integration_verified | 5 |
| FS | integration_verified | fct_food_security | integration_verified | 6 |
| FVC | integration_verified | fct_food_balance | integration_verified | 9 |
| GYI | integration_verified | fct_gender_inclusion | integration_verified | 3 |
| HDI | integration_verified | fct_hdi | integration_verified | 3 |
| INP | integration_verified | fct_land_inputs | integration_verified | 4 |
| PRC | integration_verified | fct_prices | integration_verified | 2 |
| PROD | integration_verified | agg_production_country_year | integration_verified | 5 |
| RES | integration_verified | fct_investment | integration_verified | 4 |
| SOIL | integration_verified | fct_soil_health | integration_verified | 4 |
| VEG | integration_verified | fct_vegetation | integration_verified | 1 |

## Blocked serving tables

- none

## Unsupported mart objects

- none

## Contract disagreements

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
