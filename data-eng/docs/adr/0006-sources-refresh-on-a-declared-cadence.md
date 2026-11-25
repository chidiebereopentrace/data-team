---
status: accepted
---

# Sources refresh on a declared cadence and failed sources are retried

The ingestion engine first fetched each `source_id` once. Any recorded attempt, including a failure, kept that ID from being fetched again. After the first upload the weekly run fetched nothing. Price feeds went stale without any signal, `load_mode` and `freshness_days` only mattered on runs that never came, and one transient failure, such as an HTTP 503, a BigQuery error, or a container killed mid-source, retired a source until someone renamed its ID.

Each uploaded run now reads `ingest_sources` and fetches a source when no attempt has reached raw (`LOADED` or `PARTIAL`), when its last attempt failed, or when the manifest row sets `refresh_days` and that many calendar days have passed since the run that last loaded it. Everything else is `CURRENT`: it is not fetched, and it is reported with the date it last loaded. A blank `refresh_days` means the file does not change, such as a versioned release, and it loads once.

The time of a load is read from the run id, which starts with the run's UTC start. `ingest_sources` needs no new column for the cursor, so rows written before this change still count.

A source whose latest attempt was left `STARTED` was being fetched when its run died. The next run records it as `FAILED` without fetching it, and the run after that fetches it again. A source that crashes the service costs every other week, not every run.

A run without `--execute` does not read the cursor, so a local test fetches every row in its folder.
