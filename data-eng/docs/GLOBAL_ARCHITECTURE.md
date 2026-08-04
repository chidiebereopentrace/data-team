# OpenTrace Data Engineering — Global Architecture

**Audience:** Data engineers, analysts, ML/platform engineers, and technical stakeholders integrating with OpenTrace datasets.  
**Package:** `data-eng/`  
**Last updated:** 2026-06-27  

Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) (overview), [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md) (onboarding), [LOCAL_STACK.md](LOCAL_STACK.md) (local Airbyte + Airflow), [BIGQUERY_SCHEMA.md](BIGQUERY_SCHEMA.md) (live schema catalog), [SOIL_DASHBOARD_TRANSFORMATION.md](SOIL_DASHBOARD_TRANSFORMATION.md) (star schema + soil dashboard).

---

## 1. Purpose

The **data-eng** tree owns the full analytics engineering stack for OpenTrace: **ingestion → orchestration → medallion transforms → warehouse governance → local dev tooling**. It delivers curated African agriculture, food security, climate, and socioeconomic datasets in BigQuery for BI dashboards, research, and downstream ML (notably the Ask ADZA RAG system in `ml-eng/`).

| Layer | Technology | Role |
|-------|------------|------|
| Ingestion | Airbyte | Load external sources into BigQuery **`landing`** only |
| Raw / bronze | BigQuery **`raw_dev`** (dev) / **`raw_prod`** (prod) | Immutable, source-aligned tables |
| Staging / silver | BigQuery **`staging_dev`** / **`staging_prod`** | Cleaned entities, star-schema dims & facts |
| Marts / gold | BigQuery **`mart_dev`** / **`mart_prod`** | Aggregated, analytics- and ML-ready outputs |
| Transforms | dbt (`data-eng/dbt/`) | Versioned SQL; folder names match BQ dataset IDs |
| Orchestration | Apache Airflow / Cloud Composer | Trigger syncs, run dbt, quality checks |
| Infrastructure | Terraform (`data-eng/infra/`) | Datasets, IAM, Airbyte–Composer wiring |
| Local dev | PostgreSQL partition DB (`data/local/`) | Fast offline ETL prototyping against real schema |
| Schema sync | Python scripts (`data/local/scripts/`) | BQ catalog → dbt `sources.yml` → local DDL |

### 1.1 Design principles

- **Prototype in the repo** — Notebooks and SQL here are the source of truth for logic; production runs in GCP.
- **Medallion separation** — Each layer has a dedicated BigQuery dataset; cross-layer reads follow landing → raw → staging → mart.
- **Ingestion vs transform vs orchestration vs infra** — Airbyte loads only; dbt transforms only; Airflow orchestrates only; Terraform provisions only.
- **Notebooks never production** — `data-pipelines/` and `notebooks/` are for design; promoted logic lives in dbt models or `data/sql/`.
- **Bronze is the ML contract** — Downstream RAG NL→SQL queries **`raw_dev`** (env: `BQ_DATASET_BRONZE`); silver/gold serve analytics and dashboards.
- **Schema as code** — Regenerate `sources.yml` and local DDL from live BigQuery rather than hand-maintaining hundreds of columns.
- **One database per developer locally** — Shared Postgres server uses separate DBs/schemas to avoid write conflicts.
- **Secrets never committed** — OAuth locally; service accounts and API tokens via Secret Manager in prod.

### 1.2 System boundaries

```mermaid
flowchart LR
  subgraph in_scope [In scope data-eng]
    AB[Airbyte ingestion]
    DBT[dbt transforms]
    AF[Airflow DAGs]
    TF[Terraform]
    LOCAL[Local Postgres sync]
  end

  subgraph external_src [External data sources]
    FAO[FAO / FAOSTAT]
    FEWS[FEWS NET]
    NASA[NASA POWER]
    OAI[OpenAIRE]
    GBIF[GBIF]
    WFP[WFP VAMPIRE]
    ISDA[iSDA / ISRIC soil]
  end

  subgraph gcp [GCP]
    BQ[(BigQuery medallion)]
    COMP[Cloud Composer]
    SM[Secret Manager]
  end

  subgraph downstream [Downstream consumers]
    RAG[ml-eng RAG NL-to-SQL]
    PBI[Power BI / Looker]
    FEAT[ML feature store]
  end

  external_src --> AB
  AB --> BQ
  DBT --> BQ
  COMP --> AF
  AF --> AB
  AF --> DBT
  TF --> BQ
  TF --> SM
  LOCAL -.->|partition sync| BQ
  BQ --> RAG
  BQ --> PBI
  BQ --> FEAT
```

**Out of scope for `data-eng/`:** Model training, RAG chunking/embeddings, vector stores, and serving APIs — those live in repo-root **`ml/`** / **`ml-eng/`**.

---

## 2. Repository layout

From the repository root, **`data`**, **`dbt`**, and **`infra`** are symlinks into `data-eng/` so existing paths like `cd dbt` and `data/local/.env` keep working.

| Path | Role |
|------|------|
| **`infra/`** | Terraform — BigQuery datasets, Composer/Airbyte IAM, storage stubs |
| **`airflow/`** | DAGs: Airbyte sync, dbt pipeline, data quality placeholders |
| **`dbt/`** | BigQuery transformations (`landing`, `raw_dev`, `staging_dev`, `mart_dev`) |
| **`airbyte/`** | Connection registry, destination docs, trigger scripts |
| **`data/`** | Ingestion configs, hand-written SQL, validation, **local dev DB** |
| **`data-pipelines/`** | Notebook-first ETL design (non-prod exploration) |
| **`libs/python/`** | Shared pipeline libraries (BQ helpers, logging) — no `ml/` imports |
| **`notebooks/`** | Scratch exploration |
| **`docker/`** | Local compose notes; canonical compose at `data-eng/docker-compose.yml` |
| **`config/`** | Non-secret templates; secrets via Secret Manager in prod |
| **`docs/`** | Architecture, setup guides, schema catalog |
| **`ci/github_actions/`** | Documents mapping to root `.github/workflows/data-eng-*.yml` |

### 2.1 Governance matrix

| Concern | Owner | Where logic lives | Must not |
|---------|-------|-------------------|----------|
| Load from source | Airbyte | UI + `airbyte/connections/` registry | Run transforms |
| Raw → staging → mart | dbt | `dbt/models/{landing,raw_dev,staging_dev,mart_dev}/` | Schedule jobs |
| Schedule & glue | Airflow | `airflow/dags/` | Contain business SQL |
| GCP resources | Terraform | `infra/modules/`, `infra/environments/` | Store secrets in git |
| ETL design | Engineers | `data-pipelines/`, `data/sql/` | Deploy notebooks to prod |

---

## 3. Medallion architecture

OpenTrace uses a **four-dataset medallion** on BigQuery. Naming aligns with industry bronze/silver/gold terminology while matching physical dataset IDs in GCP.

```mermaid
flowchart TB
  subgraph sources [External sources]
    APIs[REST / bulk APIs]
    GCS[GCS / file drops]
    Portals[Geoportals / ArcGIS]
  end

  subgraph landing_ds [landing]
    LAND[Airbyte-loaded tables<br/>33+ tables]
  end

  subgraph raw_ds [raw_dev / raw_prod]
    RAW[Source-aligned bronze views/tables<br/>63+ dbt models]
  end

  subgraph staging_ds [staging_dev / staging_prod]
    STAR[Star schema: 14 dims + 10 facts]
    SOIL[soil_dashboard_base]
    JOIN[land_use_soil_property_join]
  end

  subgraph mart_ds [mart_dev / mart_prod]
    GOLD[Analytics / ML-ready marts]
  end

  sources -->|Airbyte| landing_ds
  landing_ds -->|dbt raw_dev models| raw_ds
  raw_ds -->|dbt staging_dev models| staging_ds
  staging_ds -->|dbt mart_dev models| mart_ds
```

### 3.1 Dataset naming and env vars

| Medallion term | Dev dataset ID | Prod dataset ID | Env var (back-compat) |
|----------------|----------------|-------------------|------------------------|
| Landing | `landing` | `landing` | `BQ_DATASET_LANDING` |
| Bronze / raw | `raw_dev` | `raw_prod` | `BQ_DATASET_BRONZE` |
| Silver / staging | `staging_dev` | `staging_prod` | `BQ_DATASET_SILVER` |
| Gold / mart | `mart_dev` | `mart_prod` | `BQ_DATASET_GOLD` |

**Project:** `opentrace-prod-5ga4` (catalog default). Override with `BQ_PROJECT`.  
**Region:** `europe-west3` (`DBT_BIGQUERY_LOCATION`).

### 3.2 Layer responsibilities

| Layer | Mutability | Grain | Typical contents |
|-------|------------|-------|------------------|
| **landing** | Append/replace by Airbyte | As delivered by source | FEWS NET time series, FAOSTAT bulk, NASA POWER raw, OpenAIRE exports, iSDA soil bulk |
| **raw_dev** | dbt views/tables over landing + passthrough | Source schema preserved | `fao_*_bronze`, `FEWS_NET_*`, `africa_nasa_power_*`, `openaire_*`, `gbif_*` |
| **staging_dev** | dbt tables | Business entities | `dim_country`, `dim_crop`, `fact_production`, `fact_climate`, `fact_humanitarian`, … |
| **mart_dev** | dbt tables | Aggregated / dashboard-ready | Indicator marts (early stage; `gold_example` placeholder) |

### 3.3 dbt targets

Run from `data-eng/dbt/` with `DBT_PROFILES_DIR=.`:

```bash
dbt run --target raw_dev      # OAuth — local dev
dbt run --target staging_dev
dbt run --target mart_dev
dbt run --target raw_dev_sa   # Service account — Docker/CI
```

Each model folder sets `+schema` in `dbt_project.yml` so a single `dbt run --target raw_dev` can build all layers when selectors permit. To run one layer only:

```bash
dbt run --target staging_dev --select staging_dev.*
```

**Macro:** `generate_schema_name` ensures custom schema names (dataset IDs) are used verbatim rather than prefixed with the target default.

---

## 4. Data sources catalog

OpenTrace aggregates **African agriculture, food security, climate, biodiversity, and research** datasets. Sources enter via Airbyte into **`landing`**, then flow through dbt.

### 4.1 By domain

| Domain | Sources | Representative landing / raw tables |
|--------|---------|-------------------------------------|
| **Food security & markets** | FEWS NET, WFP VAMPIRE | `FEWS_NET_*`, `WFP_VAMPIRE_Tool_global_food_prices` |
| **Production & trade** | FAOSTAT, FAO QCL/RL/RP/TI/TCL, OECD Food | `FAOstat_africa_*`, `fao_trade_*`, `fao_land_use_bronze`, `OECD_Food_data_Africa_NEW` |
| **Climate & environment** | NASA POWER, Copernicus ERA5, Climate Watch, CEDA | `africa_nasa_power_raw`, `climatewatch_*`, `ceda_climate_data`, `copernicus_climate_raw_*` |
| **Soil & land** | iSDA Africa, ISRIC, cropland summaries | `isda_*`, `isric_africa_soil_data`, `africa_crop_production_full_melted`, `cropland_area_summary_2019_africa` |
| **Socioeconomic** | World Bank GDP/HDI, IFPRI | `world_gdp_ppp`, `world_Human_development_index`, `ifpri_africa_bronze`, `africa_gdp_ppp` |
| **Research & metadata** | OpenAIRE | `openaire_*_raw`, `openaire_*_bronze` |
| **Biodiversity & GIS** | GBIF, ArcGIS layers, ILRI surveys | `gbif_occurrence_search`, `arcgis_*`, `ilri_*`, `cifor_icraf_raw` |
| **Air quality (local)** | Nakuru archive | `nakuru_air_quality_archive` |

### 4.2 Ingestion pattern

1. **Configure connector** in Airbyte (local Docker or GCE VM in phase 2).
2. **Destination:** BigQuery dataset **`landing`** only — service account `sa-airbyte-bq-writer` has `dataEditor` on `landing` and `jobUser` at project scope.
3. **Register connection** in `airbyte/connections/registry.yaml` (UUID, owner, cadence).
4. **Airflow `airbyte_sync` DAG** triggers sync → polls job → triggers `dbt_pipeline`.
5. **Refresh dbt sources:** `python data/local/scripts/generate_dbt_sources.py --refresh`.
6. **dbt raw_dev models** read `{{ source('landing', 'table') }}` and materialize bronze-aligned tables/views in `raw_dev`.

### 4.3 FEWS NET (example depth)

FEWS NET is a core food-security source with multiple related tables:

- **Data series metadata:** classifications, market prices, cross-border trade, food-insecure population estimates.
- **Time series facts:** paired `*_time_series_data` tables with geographic admin levels (`admin_0`–`admin_4`), IPC phases, scenario names, and indicator values.

Bronze models in `dbt/models/raw_dev/FEWS_NET_*` mirror landing structure; silver **`fact_humanitarian`** and related dims consume normalized FEWS metrics via the star-schema staging view.

---

## 5. Transformations (dbt)

### 5.1 Project structure

```
data-eng/dbt/
├── dbt_project.yml       # Dataset vars, model folder → BQ schema mapping
├── profiles.yml          # raw_dev | staging_dev | mart_dev (+ _sa)
├── macros/
│   └── generate_schema_name.sql
├── models/
│   ├── landing/          # 33 models — passthrough / light typing from landing BQ tables
│   ├── raw_dev/          # 63 models — bronze layer
│   ├── staging_dev/      # 26 models — silver / star schema
│   └── mart_dev/         # Marts (early stage)
└── models/sources.yml    # Generated — do not hand-edit at scale
```

### 5.2 Model patterns

**Landing → raw (bronze):**

```sql
-- Typical raw_dev model reads landing source
select * from {{ source('landing', 'fao_trade_crops_livestock') }}
```

Many raw models are **`materialized='view'`** with `enabled=false` when the table is already populated directly in `raw_dev` and only needs catalog presence.

**Staging star schema:**

The dimensional model follows a strict build order documented in [SOIL_DASHBOARD_TRANSFORMATION.md](SOIL_DASHBOARD_TRANSFORMATION.md):

```
Silver sources (FAO, NASA POWER, yield, FEWS NET, …)
        ↓
stg_silver_star_metrics   ← VIEW: canonical column layout (domain_name, metric_value, …)
        ↓
   dim_* (14)  +  fact_* (10, filtered by domain_name)
        ↓
   Power BI / Looker / dashboards
```

**Fact tables** (`fact_production`, `fact_climate`, `fact_land_use`, `fact_nutrition`, `fact_humanitarian`, `fact_market_access`, `fact_policy`, `fact_technology`, `fact_value_chain`, `fact_enterprise_investment`) use **MD5 surrogate keys** (`country_key`, `period_key`, `indicator_key`, …) for consistent joins across domains.

**Soil dashboard (separate grain):**

`soil_dashboard_base` joins **iSDA** (Web Mercator projected coords) with **ISRIC** (WGS84) at rounded lat/lon for map-based Power BI dashboards — distinct from the country/year star schema.

### 5.3 Sources sync pipeline

```mermaid
sequenceDiagram
  participant BQ as BigQuery
  participant CAT as bq_schema_catalog.py
  participant GEN as generate_dbt_sources.py
  participant DBT as dbt sources.yml
  participant PG as bq_schema_to_local_pg.py

  BQ->>CAT: List datasets + table schemas
  CAT->>CAT: docs/bq_schema_catalog.json
  CAT->>CAT: docs/BIGQUERY_SCHEMA.md
  GEN->>DBT: Regenerate sources.yml
  PG->>PG: schema/{bronze,silver,gold}/*.sql
  PG->>PG: CREATE TABLE on LOCAL_DB_URL
```

Commands (from repo root, with `data/local/.env` loaded):

```bash
python data/local/scripts/bq_schema_catalog.py
python data/local/scripts/generate_dbt_sources.py --refresh
python data/local/scripts/bq_schema_to_local_pg.py
```

---

## 6. Orchestration (Airflow / Composer)

DAGs live in **`data-eng/airflow/dags/`** — not in Terraform. Composer in GCP deploys from this folder.

### 6.1 DAG inventory

| DAG ID | Schedule | Path | Purpose |
|--------|----------|------|---------|
| `airbyte_sync` | `@daily` | `dags/ingestion/airbyte_sync_dag.py` | Trigger Airbyte syncs → wait → trigger dbt |
| `dbt_pipeline` | `@daily` | `dags/transformations/dbt_pipeline_dag.py` | Run dbt + tests (placeholder operators) |
| `data_quality` | `@daily` | `dags/monitoring/data_quality_dag.py` | Quality checks placeholder |

### 6.2 Ingestion → transform flow

```mermaid
sequenceDiagram
  participant SCH as Airflow Scheduler
  participant AB as Airbyte API
  participant BQ as BigQuery landing
  participant DBT as dbt_pipeline DAG

  SCH->>AB: POST /connections/sync
  AB->>BQ: Load tables
  SCH->>AB: Poll job until succeeded
  SCH->>DBT: TriggerDagRunOperator
  Note over DBT: dbt run + dbt test (to be wired)
```

**Configuration:**

| Variable / env | Purpose |
|----------------|---------|
| `AIRBYTE_URL` | API base (local: `http://host.docker.internal:8000`; prod: private GCE IP) |
| `AIRBYTE_CONNECTION_ID` | Single connection UUID |
| `AIRBYTE_SYNC_CONNECTIONS` | JSON list of UUIDs |
| `AIRBYTE_CLIENT_TOKEN` | Optional Bearer auth |
| `AIRBYTE_JOB_TIMEOUT_SEC` | Poll timeout (default 7200s) |

**Client:** `airflow/dags/common/airbyte_client.py` — urllib-only, no extra dependencies.

### 6.3 Production deployment notes

- **`dbt_pipeline` DAG** currently uses `EmptyOperator` placeholders — wire `BashOperator` or `KubernetesPodOperator` to run `dbt run` / `dbt test` in the Composer environment with the dbt project mounted or synced from GCS.
- **Secrets:** Store `AIRBYTE_CLIENT_TOKEN` in Secret Manager; grant Composer SA accessor (see phase 2 module).
- **DAG sync:** Cloud Build, `gcloud composer environments storage`, or CI deploy step (not yet automated in repo).

---

## 7. Infrastructure (Terraform)

### 7.1 Module map

| Module | Purpose |
|--------|---------|
| **`modules/bigquery/`** | Creates datasets with `prevent_destroy = true` |
| **`modules/phase2_airbyte_composer/`** | Airbyte BQ writer SA, landing IAM, API token secret, Composer→Airbyte firewall |
| **`modules/airflow_composer/`** | Composer environment stub |
| **`modules/iam/`**, **`modules/storage/`**, **`modules/airbyte/`** | Legacy / placeholder stubs |

### 7.2 Environments

| Environment | Path | Datasets provisioned |
|-------------|------|----------------------|
| **dev** | `infra/environments/dev/` | `landing`, `raw_dev`, `staging_dev`, `mart_dev` |
| **prod** | `infra/environments/prod/` | `raw_prod`, `staging_prod`, `mart_prod` |

Prod Terraform does **not** recreate `landing` in the snippet shown — landing is shared ingestion surface; confirm env-specific tfvars for full prod layout.

### 7.3 Phase 2: Airbyte on GCE + Composer

The **`phase2_airbyte_composer`** module wires:

1. **`sa-airbyte-bq-writer`** — `roles/bigquery.jobUser` + `roles/bigquery.dataEditor` on **`landing`** only.
2. **Secret Manager** — `AIRBYTE_API_TOKEN` placeholder; Composer SA gets `secretAccessor`.
3. **Firewall** — Composer subnet CIDR → Airbyte VM tag `airbyte-server` on port 8000.

This keeps Airbyte off the public internet while allowing scheduled syncs from Composer.

### 7.4 State and secrets

- Use **remote state** (GCS backend in `environments/*/backend.tf`).
- Never commit `.tfstate` or credentials.
- Run `terraform fmt -check -recursive data-eng/infra` in CI.

---

## 8. Local development

### 8.1 PostgreSQL partition database

Full BigQuery data stays in GCP; developers sync a **partition** into local Postgres for fast iteration.

```mermaid
flowchart LR
  BQ[(BigQuery raw_dev<br/>staging_dev<br/>mart_dev)]
  SCR[sync_all_tables.py]
  PG[(Postgres datateam_local<br/>schemas: raw_dev.*<br/>staging_dev.*<br/>mart_dev.*)]
  NB[data-pipelines notebooks]
  SQL[data/sql/]

  BQ --> SCR --> PG
  PG --> NB
  NB --> SQL
  SQL -->|promote| BQ
```

**Start Postgres:**

```bash
cd data-eng
docker compose up -d postgres
docker compose --profile setup up --build setup   # optional: full BQ sync
```

**Connect:** `postgresql://postgres:postgres@localhost:5432/datateam_local`

**Power BI:** Host `localhost:5432`, database `datateam_local`, user/pass `postgres`/`postgres`.

### 8.2 Local full stack (phase 1)

Documented in [LOCAL_STACK.md](LOCAL_STACK.md):

1. Airbyte (official Docker install) → BigQuery **`landing`**
2. `docker compose -f data-eng/docker-compose.local.yml up -d` → Airflow on `:8080`
3. Set Airflow Variable `AIRBYTE_CONNECTION_ID`
4. Trigger `airbyte_sync` DAG
5. Refresh dbt sources and run models

### 8.3 Key scripts

| Script | Purpose |
|--------|---------|
| `bq_schema_catalog.py` | Dump all BQ table schemas to JSON + markdown |
| `generate_dbt_sources.py` | Build `dbt/models/sources.yml` from catalog |
| `bq_schema_to_local_pg.py` | Generate + apply Postgres DDL from BQ |
| `sync_all_tables.py` | Copy BQ partition → local Postgres |
| `populate_local_db.sh` | One-shot schema + sync + GIS load |
| `engine_connector.py` | Unified DB connection helper |
| `generate_dbt_models_from_catalog.py` | Scaffold dbt models from catalog |

### 8.4 Credentials model

| Context | Auth method |
|---------|-------------|
| Local dbt (recommended) | `gcloud auth application-default login` + OAuth targets |
| Docker dbt / BQ sync | Service account JSON → `GOOGLE_APPLICATION_CREDENTIALS`, `*_sa` targets |
| Airbyte → BQ | Dedicated SA with landing-only write |

---

## 9. Hand-written SQL and notebooks

### 9.1 `data/sql/`

Production-oriented SQL outside dbt (legacy or dashboard-specific):

| Path | Purpose |
|------|---------|
| `bronze_to_silver/` | Transforms promoted from notebooks |
| `silver_to_gold/` | Mart logic before dbt migration |
| `Indicator_Dashboard_for_looker/` | SQLbooks for Looker dashboards (climate, trade, nutrition, land use, economic metrics, yield) |

### 9.2 `data-pipelines/`

Notebook-first prototyping:

| Notebook | Topic |
|----------|-------|
| `bronze/fews_net_bronze_to_silver_transformation.ipynb` | FEWS NET bronze → silver |
| `bronze/isda_GIS_reverse_projection.ipynb` | iSDA coordinate reprojection |

**Convention:** One notebook per major flow; document partition keys and incremental strategy; export to dbt or `data/sql/` when ready.

---

## 10. Downstream consumers

### 10.1 ml-eng / Ask ADZA RAG

The RAG system (`ml-eng/ml/rag/`) queries **bronze only** at runtime:

| Setting | Typical value | Role |
|---------|---------------|------|
| `BQ_PROJECT` | `opentrace-prod-5ga4` | GCP project |
| `BQ_DATASET_BRONZE` | `raw_dev` | NL→SQL target dataset |

**Contract:**

- Table metadata YAML in `ml-eng/ml/rag/bq_tables_yaml_files/` references `raw_dev.*` tables (e.g. `ilri_vegetation_survey_v1`, `yield_raw_data`, `fews_net_food_security_master`).
- Vector chunks may *describe* silver/gold tables, but live SQL execution is allowlisted to bronze.
- Schema changes in `raw_dev` require regenerating catalog/sources and updating RAG table YAML + Qdrant `bq_table_description` corpus.

```mermaid
flowchart LR
  DE[data-eng dbt<br/>raw_dev tables]
  BQ[(BigQuery raw_dev)]
  RAG[ml-eng RAG<br/>NL-to-SQL]
  QD[(Qdrant table<br/>descriptions)]

  DE --> BQ
  BQ --> RAG
  QD --> RAG
```

### 10.2 BI and analytics

| Consumer | Primary datasets | Notes |
|----------|------------------|-------|
| **Power BI** | `staging_dev` (star schema + `soil_dashboard_base`) | Local Postgres mirror for dev |
| **Looker** | `mart_dev`, indicator SQLbooks | `data/sql/Indicator_Dashboard_for_looker/` |
| **Analyst EDA** | All layers via BQ console or synced partition | Reports drive pipeline priorities |

### 10.3 ML feature store (planned)

Gold/mart datasets feed **`ml/features/`** (repo root). Feature definitions prototype in notebooks; production materializes to Vertex AI Feature Store or dedicated BQ dataset. Data-eng owns upstream gold tables; ML owns feature definitions.

---

## 11. CI/CD

Path-triggered workflows in `.github/workflows/`:

| Workflow | Trigger path | Check |
|----------|--------------|-------|
| `data-eng-dbt.yml` | `data-eng/dbt/**` | All YAML parses |
| `data-eng-airflow.yml` | `data-eng/airflow/**` | `python -m compileall` on DAGs |
| `data-eng-terraform.yml` | `data-eng/infra/**` | `terraform fmt -check` |
| `data-eng-airbyte.yml` | `data-eng/airbyte/**` | `py_compile` on trigger script |

**Also at repo root:**

- `ci.yml` — Ruff, Black, notebook checks
- `sql-lint.yml` — SQL in `data/sql`, `data-eng/dbt/models`
- `ml-tests.yml` — ML code when `ml/` changes

**Not yet automated:** Terraform apply, Composer DAG deploy, dbt run against live BQ in CI (requires GCP credentials in Actions).

---

## 12. Environment variable reference

Set in **`data/local/.env`** (template: `data/local/.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BQ_PROJECT` | Yes | — | GCP project ID |
| `BQ_DATASET_LANDING` | No | `landing` | Airbyte destination |
| `BQ_DATASET_BRONZE` | No | `raw_dev` | Bronze / raw layer |
| `BQ_DATASET_SILVER` | No | `staging_dev` | Silver / staging layer |
| `BQ_DATASET_GOLD` | No | `mart_dev` | Gold / mart layer |
| `DBT_TARGET` | No | `raw_dev` | Default dbt target |
| `DBT_BIGQUERY_LOCATION` | No | `europe-west3` | BQ location |
| `GOOGLE_APPLICATION_CREDENTIALS` | Docker/CI | — | Path to SA JSON |
| `LOCAL_DB_URL` | Local PG | — | Postgres connection string |
| `BQ_PARTITION_LIMIT` | No | — | Row cap per table on sync |
| `AIRBYTE_URL` | Airflow | `http://localhost:8000` | Airbyte API |
| `AIRBYTE_CONNECTION_ID` | Airflow | — | Connection UUID |

---

## 13. Deployment topology

```mermaid
flowchart TB
  subgraph dev_local [Developer machine]
    DOCKER[docker-compose.yml<br/>Postgres + optional setup]
    AFLOC[docker-compose.local.yml<br/>Airflow standalone]
    ABL[Airbyte local<br/>localhost:8000]
    DBTL[dbt CLI OAuth]
  end

  subgraph gcp_prod [GCP production]
    ABVM[Airbyte GCE VM<br/>phase 2 private IP]
    COMP[Cloud Composer]
    BQ[(BigQuery<br/>landing + medallion)]
    SM[Secret Manager]
  end

  subgraph saas [External]
    SRC[Data source APIs]
  end

  SRC --> ABL
  SRC --> ABVM
  ABL --> BQ
  ABVM --> BQ
  AFLOC --> ABL
  COMP --> ABVM
  COMP --> BQ
  DBTL --> BQ
  SM --> COMP
```

| Stage | Ingestion | Orchestration | Transforms |
|-------|-----------|---------------|------------|
| **Local phase 1** | Airbyte Docker on host | Airflow Docker | dbt CLI → BQ |
| **Phase 2 prod** | Airbyte on GCE (VPC) | Cloud Composer | dbt in Composer pod / Cloud Build |

---

## 14. Operational runbooks

### 14.1 Add a new data source

1. Create Airbyte source + connection → destination **`landing`**.
2. Run sync; verify table in BigQuery console.
3. `python data/local/scripts/generate_dbt_sources.py --refresh`
4. Add `dbt/models/raw_dev/<source>.sql` reading `{{ source('landing', '...') }}`.
5. Extend `stg_silver_star_metrics` if the source feeds the star schema (new `domain_name` CTE).
6. Add dim/fact models or extend existing facts.
7. `dbt run --select raw_dev.<model>+` to build downstream.
8. Update `airbyte/connections/registry.yaml`.
9. If RAG-relevant: add table YAML under `ml-eng/ml/rag/bq_tables_yaml_files/` and re-ingest Qdrant descriptions.

### 14.2 Schema change in BigQuery

1. Apply change in BQ (or via Airbyte schema evolution).
2. `python data/local/scripts/bq_schema_catalog.py`
3. `python data/local/scripts/generate_dbt_sources.py --refresh`
4. Update affected dbt models.
5. Optionally refresh local Postgres: `bq_schema_to_local_pg.py` + `sync_all_tables.py`.

### 14.3 Onboard a new team member

Run `bash scripts/first_time_setup.sh` — see [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md).

### 14.4 Debug failed Airbyte sync

1. Check Airbyte UI job logs.
2. Verify SA has `dataEditor` on `landing` and `jobUser` on project.
3. Confirm table does not conflict with concurrent schema experiments on shared landing.
4. Re-trigger: `python data-eng/airbyte/scripts/trigger_sync.py <uuid>`.

---

## 15. Current maturity and roadmap

| Area | Status | Next steps |
|------|--------|------------|
| **landing ingestion** | Active — 33+ tables | Expand connectors; formalize registry |
| **raw_dev dbt** | Active — 63 models | Enable disabled views; standardize naming |
| **staging_dev star schema** | Active — 14 dims, 10 facts | Complete `stg_silver_star_metrics` source coverage |
| **mart_dev** | Early — placeholder | Promote Looker SQLbooks to dbt marts |
| **Airflow DAGs** | Scaffold | Wire real dbt/bash operators in Composer |
| **Terraform prod** | Partial | Align prod landing + phase 2 module in env tfvars |
| **CI** | Lint/compile | Add dbt parse + optional BQ integration tests |
| **Feature store** | Planned | Gold → `ml/features/` pipeline |

---

## 16. Quick reference

| Goal | Command / location |
|------|-------------------|
| First-time setup | `bash scripts/first_time_setup.sh` |
| Refresh dbt sources | `python data/local/scripts/generate_dbt_sources.py --refresh` |
| Run all dbt layers | `cd dbt && DBT_PROFILES_DIR=. dbt run --target raw_dev` |
| Run staging only | `dbt run --target staging_dev --select staging_dev.*` |
| Start local Postgres | `cd data-eng && docker compose up -d postgres` |
| Start local Airflow | `docker compose -f data-eng/docker-compose.local.yml up -d` |
| Trigger Airbyte sync | `python data-eng/airbyte/scripts/trigger_sync.py <uuid>` |
| Schema catalog | `python data/local/scripts/bq_schema_catalog.py` |
| Star schema guide | [SOIL_DASHBOARD_TRANSFORMATION.md](SOIL_DASHBOARD_TRANSFORMATION.md) |
| RAG integration | `ml-eng/ml/rag/docs/GLOBAL_ARCHITECTURE.md` |

---

## 17. Related documentation index

| Document | Path |
|----------|------|
| Data-eng overview | [ARCHITECTURE.md](ARCHITECTURE.md) |
| First-time setup | [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md) |
| Local Airbyte + Airflow | [LOCAL_STACK.md](LOCAL_STACK.md) |
| BigQuery schema catalog | [BIGQUERY_SCHEMA.md](BIGQUERY_SCHEMA.md) |
| Star schema + soil dashboard | [SOIL_DASHBOARD_TRANSFORMATION.md](SOIL_DASHBOARD_TRANSFORMATION.md) |
| dbt project | [../dbt/README.md](../dbt/README.md) |
| Local Postgres | [../data/local/README.md](../data/local/README.md) |
| Terraform | [../infra/README.md](../infra/README.md) |
| Airbyte | [../airbyte/README.md](../airbyte/README.md) |
| RAG global architecture | [../../ml-eng/ml/rag/docs/GLOBAL_ARCHITECTURE.md](../../ml-eng/ml/rag/docs/GLOBAL_ARCHITECTURE.md) |
