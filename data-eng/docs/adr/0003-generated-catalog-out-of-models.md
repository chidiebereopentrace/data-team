---
status: accepted
---

# Generated BigQuery catalog lives under dbt/catalog/

Hand-maintaining a multi‑10MB `models/sources.yml` made the dbt models tree unusable in editors and agents. Generated source YAML now lands at `dbt/catalog/sources.generated.yml`, included via `model-paths`. Hand-written sources (`sources_ingestion.yml`, `sources_geoboundaries.yml`, etc.) stay under `models/`. Regenerate with `python data/local/scripts/generate_dbt_sources.py --refresh`.

The generated file is committed (it was previously gitignored): without it, 203 of 204 `source()` refs are unresolved and `dbt parse` fails on a clean clone and in CI, which has no BigQuery access. Regenerate and commit it when BigQuery schemas change.
