---
status: accepted
---

# The ingestion engine trusts the manifest, not a probe

Curated ingestion loads one coded source at a time into `landing_dev`. The OTA framework collects publisher URLs and will land an unlabeled grid. An earlier decision map (`manifest-ingestion`) let a probe override the declared contract and put the package at `data-eng/manifest-ingestion/`. We rejected both.

The engine is a separate package at `data-eng/ingestion_engine`. It coexists with curated ingestion and does not import that package at runtime. One manifest row is one landing table. The row declares the contract (`json_api`, `csv`, `xlsx`, or `html_table`), the landing table name, the indicator class, the geo grain, the temporal grain, and whether the endpoint is gated. A gated row names a secret (`auth_required`, `auth_type`, `auth_ref`). The secret value stays in the job environment.

The engine fetches only that contract. A header, country, date, or class-column mismatch quarantines the row or fails the ref. It does not guess a layout, and it does not save `column_1` grids as success. A hand-written adapter exists only when those four contracts cannot express the publisher. Load behavior follows the row: `append_snapshot`, `replace_partition`, or `replace_table`. Landing stores the reported extract. A row moves to the paired raw dataset only after its subject country is one of the 54 African countries and its date, value, unit, and reported class columns check out. Other country columns declared in `geo_country_columns` stay on the row. A resolved subject country outside those 54 is left out of landing and raw. `landing_dev` pairs with `raw_dev`, and `landing` pairs with `raw`.
