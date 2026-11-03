# Decision map: manifest-ingestion

Superseded for the product contract by `data-eng/docs/adr/0004-manifest-contract-is-declared.md` and `data-eng/docs/adr/0005-weekly-coverage-and-failure-report.md`, and by the language in `data-eng/CONTEXT.md`. The package is `data-eng/ingestion_engine`. The manifest wins over a probe. The weekly record is the BigQuery coverage and failure report, not email. Tickets below are the earlier grill, kept so the rejected alternatives stay visible.

Parallel self-deployable ingestion engine beside curated `data-eng/ingestion/`. Same GCP env; weekly worker; repo-checked manifests.

Grilling (this conversation) resolved the product contract. Open tickets are remaining fog before `/system-architect` target design and `/to-prd`.

Related plan: Cursor plan `manifest_ingestion_next`. Next after map: `/system-architect` (target_design).

---

## #1: Parallel vs evolve curated ingestion?

Blocked by: —
Type: Grilling

### Question

Is this a parallel product coexisting with today’s per-source packages, or should the current engine evolve into this?

### Answer

Parallel product. Separate package; coexists. Same GCP env (project, credentials, `landing_dev` / `raw_dev`). Curated `ingestion/` stays for hand-authored sources.

---

## #2: Package location and deploy?

Blocked by: #1
Type: Grilling

### Question

Where does it live and run?

### Answer

New package `data-eng/manifest-ingestion/`. Deploy on Railway. Schedule ~weekly. Manifest is a file checked into the repo for v1 (GCS/Railway-volume manifest later = same read seam).

---

## #3: Source format scope?

Blocked by: #1
Type: Grilling

### Question

Which source kinds in product scope?

### Answer

All: tabular downloads (CSV/Excel), APIs (JSON/XML), HTML, text. Architecture must fit every kind; extractors can ship in sequenced issues.

---

## #4: Classify: declare vs probe?

Blocked by: #3
Type: Grilling

### Question

Does the manifest declare contract, or does the engine probe? On conflict, who wins?

### Answer

Both. Manifest contract is a hint; probe can fill or disagree. **Probe wins.** Log declare/probe mismatch.

---

## #5: Prepare depth and transform authorship?

Blocked by: #1
Type: Grilling

### Question

How far does prepare go, and where does transform logic live?

### Answer

Source-specific business transforms (like today’s `transform.py`), plus iteration from extract metadata/profiles. **Hybrid:** declarative default (naming, types, maps, ingest metadata, profile checks) + per-`source_id` code escape hatch. Keep robust shared prepare interface.

---

## #6: What lands in BigQuery vs side files?

Blocked by: #5
Type: Grilling

### Question

Materialize as-extracted + prepared, or only prepared?

### Answer

Only **prepared** table to warehouse. Extract payloads + data profiles = logs/side files on a **Railway volume**. Descriptions stay tied to extract, not overwritten by prepare.

---

## #7: Run notifications?

Blocked by: #2
Type: Grilling

### Question

When/how to notify humans?

### Answer

Every weekly run emails success summary + failures to a configured recipient list. Provider: **Resend** (API key in Railway env), behind a small send-run-report seam.

---

## #8: Partial failure and load modes?

Blocked by: #6
Type: Grilling

### Question

One row fails — stop or continue? Weekly re-load semantics?

### Answer

**Partial OK:** continue remaining rows; load successes; report failures in email. Load: **replace default** (align with curated `replace_partition`); **merge opt-in** per manifest row. No append-first design.

---

## #9: Auth for protected sources?

Blocked by: #3
Type: Grilling

### Question

How do protected APIs/files authenticate?

### Answer

Manifest references secret **names** + auth scheme (`none` | `bearer` | `header` | `basic` | `query`). Values only in Railway/local env — never in the checked-in manifest. Missing secret → that row fails (partial OK).

---

## #10: landing_dev vs raw_dev?

Blocked by: #6
Type: Grilling

### Question

Does the engine write only landing, or also raw?

### Answer

Always write prepared data to **`landing_dev`**. Promote to **`raw_dev` only when the manifest row opts in** (`promote_to_raw` / approved). Fits LandingDev sandbox vs RawDev primordial contract in `data-eng/CONTEXT.md`.

---

## #11: Table naming?

Blocked by: #10
Type: Grilling

### Question

How are BigQuery table ids chosen?

### Answer

Default: `{source_id}_{indicator_class}_manifest`. Optional `target_table` override. `indicator_class` from a controlled vocabulary. `_manifest` suffix avoids collisions with curated tables in the same datasets.

---

## #12: Railway cron + volume layout?

Blocked by: #2, #6, #7
Type: Research

### Question

Exact Railway service shape: cron expression, volume mount path, process entrypoint, and where side files live under the volume?

### Answer

One Railway service (no always-on web in v1). Cron entrypoint runs the weekly job (e.g. `0 2 * * 0` UTC). Volume mounted at `/data/manifest-ingestion`; side files at `{run_id}/{table_stem}/…` under that mount. Env: GCP credentials, Resend, recipients, and other secrets.

---

## #13: Share or copy curated load/BQ helpers?

Blocked by: #8, #10
Type: Grilling

### Question

Should `manifest-ingestion` import `ingestion.common.load` / `bq`, or copy/adapt replace+merge patterns into the new package to keep engines decoupled?

### Answer

**Copy/adapt (B).** `manifest-ingestion` owns its own load/BQ helpers, started from curated `replace_partition` / `merge` + metadata patterns, then free to diverge. No runtime import of `ingestion.common`. Engines stay decoupled; same GCP env only.

---

## #14: indicator_class vocabulary source?

Blocked by: #11
Type: Research

### Question

Where is the authoritative list of allowed `indicator_class` values (ontology seed, mart cards, new manifest-local enum), and how does validation fail a row?

### Answer

Authoritative codes from [`ml-eng/ml/rag/helpers/mart_indicator_classes.yaml`](../../../ml-eng/ml/rag/helpers/mart_indicator_classes.yaml) (AH, BIO, CLIM, EL, ENV, FS, FVC, GYI, HDI, INP, PRC, PROD, RES, SOIL, VEG). Manifest rows must use these **codes**. Unknown → that row fails (partial OK). No separate manifest-local taxonomy in v1. Package may load/copy the code list at build/runtime without importing the RAG chatbot package if deploy boundaries require it — semantics stay the YAML codes.

---

## #15: HTTP retry / timeout policy?

Blocked by: #3, #9
Type: Grilling

### Question

Retries, timeouts, and which errors are retryable for extract/probe?

### Answer

Up to **3 attempts** total; exponential backoff with jitter (e.g. 1s → 2s → 4s). Retry on network errors, timeouts, HTTP **5xx** and **429** (honor `Retry-After` when present). Fail fast on other **4xx** (auth/config). Default request timeout **30s** for probe/small APIs; **120s** (or per-row override) for file downloads.

---

## #16: Resend from-domain and env contract?

Blocked by: #7
Type: Research

### Question

Verified sending domain, `from` address, recipient env var shape, and failure behavior if Resend is misconfigured (fail run vs log-only)?

### Answer

Env: `RESEND_API_KEY`, `MANIFEST_INGEST_MAIL_FROM` (verified domain address), `MANIFEST_INGEST_MAIL_TO` (comma-separated). Domain verification in Resend dashboard (not app code). If notify fails after loads: log + mark notify failed; **do not roll back BQ loads** (email best-effort).

---

## Frontier

All tickets **#1–#16** answered. No remaining fog for product/ops contract. **Map done.** Next: `/system-architect` (target_design) → `/to-prd` → `/to-issues`.
