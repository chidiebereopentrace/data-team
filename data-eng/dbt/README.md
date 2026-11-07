# dbt — OpenTrace (transform-v2)

Pipeline version **2.x** (`version` in `dbt_project.yml`; git tags `transform-v2.*`). Baseline before the contract cut: tag `transform-v1.0.0`.

**Contract:** Ingestion writes `landing_dev` and `raw_dev`. dbt treats **`raw_dev` as the primordial source** through `staging_dev` → `intermediate_dev` → `mart_dev`. Soft cutover: existing model SQL (including transitional `models/landing` and `models/raw_dev` runners) is unchanged at 2.0.0.

See [`../README.md`](../README.md), [`../CONTEXT.md`](../CONTEXT.md), [`../docs/adr/`](../docs/adr/).

Env vars `BQ_DATASET_BRONZE` / `SILVER` / `GOLD` are legacy *names*; values should be `raw_dev` / `staging_dev` / `mart_dev`.

## Prerequisites

- BigQuery datasets: `landing_dev`, `raw_dev`, `staging_dev`, `intermediate_dev`, `mart_dev` (and any legacy `landing` still referenced by models).
- Credentials: OAuth (`gcloud auth application-default login`) or service account for `*_sa` targets.
- The generated source catalog is committed. After BigQuery schema changes, regenerate and commit it:

```bash
cd data-eng
python data/local/scripts/generate_dbt_sources.py --refresh
```

## Running dbt

```bash
cd data-eng/dbt
dbt deps
dbt run --target raw_dev
dbt run --target staging_dev
dbt run --target intermediate_dev
dbt test --target staging_dev
dbt run --target mart_dev
```

## Project layout

```
dbt/
├── dbt_project.yml          # version 2.0.0+
├── profiles.yml
├── catalog/
│   └── sources.generated.yml  # generated, committed
└── models/
    ├── sources_ingestion.yml  # hand-written
    ├── sources_geoboundaries.yml
    ├── landing/               # transitional runners (soft cutover)
    ├── raw_dev/
    ├── staging_dev/
    ├── intermediate_dev/
    └── mart_dev/
```

## Versioning

| Tag | Meaning |
| --- | ------- |
| `transform-v1.0.0` | Frozen baseline before ownership contract |
| `transform-v2.Y.Z` | Current contract; bump with `dbt_project.yml` `version` |

Do not create a parallel `dbt-v2/` tree.
