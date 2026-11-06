# Ingestion engine

Weekly job that reads a directory of manifest CSVs. One row is one landing table. Field-by-field instructions for writing a row are in [MANIFEST.md](MANIFEST.md). A new manifest CSV is ingested on the next weekly run when its source id is not already in the ingest report. Sources already recorded as loaded, partial, or failed are not downloaded again.

A row declares the contract (`json_api`, `csv`, `xlsx`, `xls`, or `html_table`), the landing table, the indicator class, the geo grain, the temporal grain, and whether the endpoint is gated. Gated rows name a secret with `auth_ref`. The secret value comes from the environment.

A `csv`, `xlsx`, or `xls` row may set `file_name` to one file name. A Dataverse or Zenodo file list uses that name, and the file download stays on the same host as the endpoint. A ZIP uses that name as the member to read. A ZIP with one CSV or workbook can leave `file_name` blank. A blank `file_name` means the endpoint body is already the table. `row_filter` keeps rows whose cells match `Column=a|b&Other=c`. A blank filter keeps every row. Rows that miss the filter are not loaded and are not quarantined.

Landing keeps the reported columns. A row moves to the paired raw dataset only when its subject country is one of the 54 African countries, its date parses, its value is numeric, its unit is present, and its reported class columns are filled. A resolved subject country outside that list is left out of landing and raw and does not fail the source. Prices also need a commodity and a currency. Production also needs a commodity and a measure kind of quantity, area, or yield.

`geo_country_column` is the country the row is about. `geo_country_columns` lists every other country column the publisher sends, such as importer, exporter, partner, donor, or recipient. Each listed name must also be in `column_names`. Those columns are kept with the publisher's value. A non-African partner does not remove the column and does not reject the row.

Local files are written under `ingestion_engine/out`:

- `landing/` — the reported extract
- `prepared/` — the rows that passed and are loaded to raw
- `quarantine/` — rows that stayed in landing and did not move to raw
- `reports/ingest_runs.csv` — one row for the job, including sources ingested and rows added
- `reports/ingest_coverage.csv` — prepared rows by indicator class, country, and year
- `reports/ingest_sources.csv` — one row per source, including the owner, rows outside Africa, and quarantine counts by reason
- `reports/ingest_failures.csv` — refs that failed, were partial, or were skipped
- `reports/ingest_report.html` — the same report as a page, with rows added by run, indicator, and country
- `reports/ingest_report.pdf` — the same page as a PDF
- `reports/ingest_jobs.html` — one card per source, with filters for status, contract, and error category

The process exits non-zero when any enabled row failed or was partial. Disabled rows are skipped and do not fail the job.

Install once from `data-eng`:

```bash
pip install -e ./ingestion_engine
ingestion-engine --execute
```

Copy `.env.example` to `.env` and set `BQ_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS`. `--manifests` and `--out` override `INGEST_MANIFESTS` and `INGEST_OUT`. The default output directory is this folder's `out/`.

`--execute` loads `landing_dev` and `raw_dev`, and appends the report tables to the landing dataset. After a successful upload the landing, prepared, quarantine, and raw files are deleted. The reports stay. `landing` and `raw` are refused unless `INGEST_ALLOW_PROD=1`. On Railway, set `BQ_SERVICE_ACCOUNT_JSON` instead of a key file. The cron is Sunday 02:00 UTC.

When `INGEST_REPORT_EMAILS` is set, the job emails a short weekly note after the upload: the period, sources in the run, sources loaded, rows added, the success rate, every source, and the failed or partial sources. The PDF chart summary, `ingest_jobs.html`, and the three CSVs are attached. The SMTP host, port, user, password, and from-address use `INGEST_SMTP_HOST`, `INGEST_SMTP_PORT`, `INGEST_SMTP_USER`, `INGEST_SMTP_PASSWORD`, and `INGEST_SMTP_FROM`.

`append_snapshot` adds the run. `replace_partition` replaces the `_ingest_date` partition. `replace_table` replaces the whole table.

Country names ship in `src/ingestion_engine/country_names.yaml`. The 54 African ISO3 codes ship in `src/ingestion_engine/africa_iso3.txt`.
