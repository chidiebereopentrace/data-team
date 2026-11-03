# Data engineering (OpenTrace)

Analytics engineering context for loading external agricultural and food-security data into BigQuery and transforming it for BI and downstream ML.

## Language

**LandingDev** (`landing_dev`):
Ingest-owned BigQuery dataset where load jobs write first (sandbox; partitions may expire).
_Avoid_: bronze landing, Airbyte landing, durable landing-as-primary-target

**RawDev** (`raw_dev`):
Ingest-owned, source-aligned BigQuery dataset. Ingestion writes it. Transformation treats it as the primordial source.
_Avoid_: bronze (as team language), “dbt-owned raw”

**StagingDev** (`staging_dev`):
Transform-owned cleaned entities and conformed grain after RawDev.
_Avoid_: silver (as team language)

**IntermediateDev** (`intermediate_dev`):
Transform-owned enrichment and business logic between staging and marts.

**MartDev** (`mart_dev`):
Transform-owned analytics-ready outputs for BI and consumers.
_Avoid_: gold (as team language)

**Ingestion**:
The module and jobs that write LandingDev and RawDev from external sources.
_Avoid_: Airbyte (retired from the active path)

**IngestNotebook**:
A notebook under `ingestion/notebooks/` used to try or demonstrate a load.
_Avoid_: data-pipelines (retired), top-level `notebooks/` for ingest, treating notebooks as production transforms

**IngestScript**:
A reusable runner under `ingestion/scripts/` (or Make) for repeatable loads.
_Avoid_: copying package internals into ad-hoc one-offs without a script

**IngestPackage**:
The installable CLI and source modules under `ingestion/package/` (`python -m ingestion`).
_Avoid_: presenting the package tree as the primary engineer entry point

**Transformation**:
The dbt pipeline from RawDev through MartDev (transform-v2 contract).
_Avoid_: medallion notebooks as production, bronze→silver→gold naming

**TransformV1**:
Baseline dbt pipeline frozen at git tag `transform-v1.0.0` before the ownership contract cut.

**TransformV2**:
Current dbt pipeline version (`dbt_project` 2.x, tag `transform-v2.*`) under the ingest-writes-raw / primordial-raw contract. Model SQL may match v1 at cutover.

**Ingestion engine**:
The weekly job that reads manifests and writes prepared rows to LandingDev. It coexists with the curated IngestPackage.
_Avoid_: OTA ingestion framework, Airbyte

**Manifest**:
A CSV of source declarations the ingestion engine reads. A directory of these files is one weekly run.
_Avoid_: testing rules, source.yaml, a tracker spreadsheet of URLs

**Manifest row**:
One landing table from one endpoint, file, or HTML table.
_Avoid_: source, when one publisher has several tables

**Ingestion contract**:
How a manifest row is fetched: a JSON API, a CSV file, an Excel workbook, or an HTML table.
_Avoid_: probe

**Gated endpoint**:
A manifest row that names a secret. The secret value is not part of the manifest.
_Avoid_: a public URL, a key pasted into the CSV

**Indicator class**:
The class code a manifest row declares for its measure. The allowed codes are EL, GYI, FVC, FS, PROD, PRC, CLIM, SOIL, AH, VEG, ENV, INP, HDI, BIO, and RES.
_Avoid_: a review label applied after collection

**Geo grain**:
The place level a manifest row promises: country, first admin area, second admin area, market, or point.
_Avoid_: an unlabeled text column that might be a place

**Temporal grain**:
The period a manifest row promises: day, month, quarter, season, or year.
_Avoid_: a date string left unparsed

**Prepared row**:
A landing row with a resolved country, a parsed observation date, and its indicator class.
_Avoid_: unclassified grid

**Quarantine**:
A row that broke its geo, time, or class promise and did not enter LandingDev.
_Avoid_: a partial export that still lands

**Coverage report**:
The count of prepared rows that landed, by indicator class, country, and year, for one weekly run.
_Avoid_: a PDF exported from a laptop

**Failed ref**:
A manifest row that did not fully load. It is part of the weekly report.
_Avoid_: a log line that is not in the report database
