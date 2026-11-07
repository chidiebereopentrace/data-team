# Data

Local partition DB and validation assets. Aligns with **transform-v2**:

**Ingest writes** `landing_dev` + `raw_dev` → **dbt** `raw_dev` → `staging_dev` → `intermediate_dev` → `mart_dev`

## Layout

- **`local/`** — Local Postgres partition: schema, sync scripts, `.env.example`. (Physical rename to `data-eng/local/` deferred — many scripts and compose paths still use `data/local/`.)
- **`validation/`** — Data quality checks.
- Legacy SQL under `data/sql/` moved to [`../archive/data-sql/`](../archive/data-sql/). Prefer dbt models.

Runtime ingest: **`data-eng/ingestion/`**.

## BigQuery schema catalog

```bash
cd data-eng
python data/local/scripts/bq_schema_catalog.py
python data/local/scripts/generate_dbt_sources.py
# writes dbt/catalog/sources.generated.yml (committed; commit the refresh)
```

## Flow

1. Ingest → **`landing_dev`** / **`raw_dev`** (`data-eng/ingestion/`).
2. dbt reads primordial **`raw_dev`**, writes staging → intermediate → mart.
3. Sync a partition into local Postgres (`data/local/`) for IDE work.
4. Validation in CI or Composer after pipeline steps.
