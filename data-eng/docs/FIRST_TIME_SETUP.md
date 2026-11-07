# First-time setup guide

For team members who clone the repo for the first time. This guide gets Docker and Postgres up, syncs dbt sources from BigQuery using your OAuth access, then walks through using dbt for transformation pipelines and orchestration.

## Prerequisites

- **Docker** (and Docker Compose) — for Postgres and optional dbt-in-Docker.
- **Python 3** (e.g. 3.10+) — for catalog and dbt-source scripts; use a venv if you like.
- **gcloud CLI** — for OAuth to BigQuery (`gcloud auth application-default login`). No service-account key required for the initial dbt setup.

## 1. Run the first-time setup script

From the repo root:

```bash
git clone <repo-url> && cd data-team
bash scripts/first_time_setup.sh
```

The script:

1. **Builds the Docker image** and starts Postgres (`docker compose up -d`). The `datateam_local` database is created on first run.
2. **Ensures `data/local/.env` exists** — copies from `.env.example` if missing. You should edit it and set **`BQ_PROJECT`** (and optionally `BQ_DATASET_LANDING`, `BQ_DATASET_BRONZE`, etc.).
3. **Reminds you to authenticate** — for OAuth (no key file), run once:  
   `gcloud auth application-default login`
4. **Populates dbt sources from BigQuery** — runs `python data/local/scripts/generate_dbt_sources.py --refresh`, which fetches the BQ schema and rewrites the committed **`dbt/catalog/sources.generated.yml`** so dbt stays in sync.

If `BQ_PROJECT` is not set when you run the script, it will skip step 4 and tell you to set it and re-run (or run the generate script manually).

Optional: to also **populate the local Postgres** with schema and data from BigQuery, you need a service-account key in `data/local/keys/` and `GOOGLE_APPLICATION_CREDENTIALS` in `.env`, then run `docker compose --profile setup up`. See [data/local/README.md](../data/local/README.md).

---

## 2. Using dbt for transformation pipelines

### 2.1 Load env and run dbt

From repo root, load `.env` then run dbt:

```bash
set -a && source data/local/.env && set +a
cd dbt
dbt deps
dbt run --target raw_dev
```

- **Targets:** `raw_dev`, `staging_dev`, `mart_dev` (OAuth) or `raw_dev_sa`, `staging_dev_sa`, `mart_dev_sa` (service-account key). Each target writes to the corresponding BigQuery dataset.
- **One `dbt run` runs all models** (raw_dev, staging_dev, mart_dev) and writes to all three datasets, because each model folder has its own `+schema` in `dbt_project.yml`. To run only one layer:
  - raw_dev only: `dbt run --target raw_dev --select raw_dev.*`
  - staging_dev only: `dbt run --target staging_dev --select staging_dev.*`
  - mart_dev only: `dbt run --target mart_dev --select mart_dev.*`

### 2.2 Test and docs

```bash
dbt test
dbt docs generate
dbt docs serve
```

### 2.3 Add or change models

- **raw_dev:** Add SQL under `dbt/models/raw_dev/`. Read from `{{ source('landing', 'table_name') }}` or `{{ source('raw_dev', 'table_name') }}`. These models write to the raw_dev dataset (via `BQ_DATASET_BRONZE`).
- **staging_dev:** Add under `dbt/models/staging_dev/`. Use `{{ ref('...') }}` and optionally `{{ source('staging_dev', '...') }}`. Writes to the staging_dev dataset.
- **mart_dev:** Add under `dbt/models/mart_dev/`. Use `{{ ref('...') }}` and sources. Writes to the mart_dev dataset.

Full reference: [dbt/README.md](../dbt/README.md).

### 2.4 Run dbt in Docker (optional)

If you prefer not to install dbt locally, use the `dbt` service with the **`_sa`** targets (service-account key required):

```bash
# Ensure data/local/.env has GOOGLE_APPLICATION_CREDENTIALS and data/local/keys/<key>.json exists
docker compose --profile dbt run --rm dbt sh -c "dbt deps && dbt run --target raw_dev_sa"
docker compose --profile dbt run --rm dbt dbt run --target staging_dev_sa
docker compose --profile dbt run --rm dbt dbt run --target mart_dev_sa
```

---

## 3. Orchestration

Scheduled pipelines (raw_dev → staging_dev → mart_dev, ingestion, etc.) run in **GCP Composer** (Airflow). DAGs live under **`data-eng/airflow/dags/`** (historical `infra/composer/` points here).

- **Development:** Run dbt locally or in Docker as above; push changes to the repo.
- **Production:** Deploy DAGs from `data-eng/airflow/` to the Composer environment (e.g. via Cloud Build or manual upload). They trigger ingestion, BigQuery SQL, and optionally dbt or Vertex AI jobs.

See [infra/README.md](../infra/README.md) (symlink to `data-eng/infra`) and [docs/ARCHITECTURE.md](ARCHITECTURE.md) for how orchestration fits the rest of the stack.

---

## Quick reference

| Goal | Command |
|------|--------|
| First-time setup | `bash scripts/first_time_setup.sh` |
| Refresh dbt sources from BQ | `python data/local/scripts/generate_dbt_sources.py --refresh` |
| Run all dbt models | `cd dbt && dbt run --target raw_dev` (or staging_dev/mart_dev) |
| Run one layer only | `dbt run --target raw_dev --select raw_dev.*` |
| Run dbt in Docker | `docker compose --profile dbt run --rm dbt dbt run --target raw_dev_sa` |
