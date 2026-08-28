# Production environment variables (RAG API)

Canonical checklist for **Railway**, **GCE / Cloud Run**, or any 12-factor deploy of `ml.rag.app.api`. Apply vars in your platform dashboard; this file is not loaded at runtime.

For Streamlit QA (often a **separate Railway environment**), mirror pipeline vars and add QA flags — see the Streamlit section below and [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md). Do **not** copy private Redis URLs across environments.

Full reference: [config/.env.example](../config/.env.example).

---

## Remove (obsolete or harmful)

| Variable | Why |
|----------|-----|
| `QDRANT_COLLECTION_DATA_DESCRIPTIONS` | Removed; BQ table selection uses staging YAML only |
| `NEWS_PUBLIC_REPORTS` or similar legacy alias | Use `QDRANT_COLLECTION_PUBLIC_REPORTS=public_reports` |
| `GOOGLE_APPLICATION_CREDENTIALS=config/keys/...` | Invalid on Railway; use `GOOGLE_APPLICATION_CREDENTIALS_BASE64` |
| `GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json` | Stale HF/serving path; entrypoint uses `/tmp/gcp-sa-key.json` only |
| `GCP_SA_JSON_B64` / `GCP_SA_JSON` | Legacy HF/serving names; copy into `GOOGLE_APPLICATION_CREDENTIALS_BASE64` then delete |
| Dashboard start that writes `/tmp/gcp-sa.json` | Replaced by [`railway.toml`](../railway.toml) `startCommand` (uvicorn only) |
| `RAG_LLM_BASE_URL` pointing at LAN / LM Studio | Production must use a public LLM endpoint (e.g. OpenRouter) |
| Per-plan **70B** overrides | e.g. `RAG_LLM_MODEL_GOVERNMENT=...70b...`, `RAG_BQ_REASONER_MODEL_ID=...70b...` |
| `RAG_LLM_RERANK=off` as primary rerank guidance | Use `RAG_RERANKER_MODE=cross_encoder` instead |
| `COHERE_API_KEY` / `RAG_RERANKER_COHERE_API_KEY` | Not used; production uses local cross-encoder via fastembed |
| `RAG_RERANKER_MODE=cohere` or `openrouter` | Replaced by explicit `cross_encoder` (OpenRouter LLM auto-promotes to OpenRouter rerank if unset) |
| `RAG_RERANK_MODEL_ID=cohere/rerank-4-pro` | OpenRouter rerank slug; not used with cross_encoder |

Leave `RAG_USE_LEGACY_RESEARCH_COLLECTION` **unset/off** unless you still query the old mixed `research_other_papers` collection.

---

## Set — required

### Qdrant (six retrieve paths)

Names must match **actual** Qdrant Cloud collections (after migration/ingestion).

```bash
QDRANT_URL=https://...
QDRANT_API_KEY=...

QDRANT_COLLECTION_NEWS=news_data
QDRANT_COLLECTION_ACADEMIC_PAPERS=academic_papers
QDRANT_COLLECTION_POLICIES=policies
QDRANT_COLLECTION_PUBLIC_REPORTS=public_reports
QDRANT_COLLECTION_FORMATION=formation
QDRANT_COLLECTION_OTA_INSIGHTS=OTA_insights
```

### LLM (OpenRouter — chat Qwen3-30B-A3B; BQ planner/NL2SQL on DeepSeek)

```bash
RAG_LLM_BASE_URL=https://openrouter.ai/api/v1
RAG_LLM_API_KEY=<OPENROUTER_API_KEY>
RAG_LLM_MODEL_ID=qwen/qwen3-30b-a3b-instruct-2507
RAG_SUMMARY_MODEL_ID=qwen/qwen3-30b-a3b-instruct-2507
RAG_LLM_TIMEOUT_S=300
# BQ table planner + SQL generation (dedicated; keep chat on Qwen)
RAG_BQ_REASONER_MODEL_ID=deepseek/deepseek-v4-flash-0731
RAG_BQ_NL2SQL_MODEL_ID=deepseek/deepseek-v4-flash-0731
```

Unset or align per-plan vars so they do not override chat to a larger model:

- `RAG_LLM_MODEL_FREE`, `RAG_LLM_MODEL_FARMERS`, `RAG_LLM_MODEL_GOVERNMENT`, etc.

### BigQuery (staging YAML reasoner + NL2SQL)

```bash
BQ_PROJECT=opentrace-prod-5ga4
BQ_DATASET_SILVER=staging_dev
GOOGLE_APPLICATION_CREDENTIALS_BASE64=<base64 of GCP service account JSON>
```

RAG NL-to-SQL queries **`staging_dev`** only (`BQ_DATASET_SILVER`). `BQ_DATASET_BRONZE` is for data-eng tooling, not the live RAG path.

Encode key: [scripts/encode-gcp-key.sh](../scripts/encode-gcp-key.sh).

**Railway bootstrap:** `Dockerfile.railway` / `Dockerfile.railway.streamlit` run [scripts/railway-entrypoint.sh](../scripts/railway-entrypoint.sh), which:

1. Base64-decodes `GOOGLE_APPLICATION_CREDENTIALS_BASE64` to **`/tmp/gcp-sa-key.json`**
2. Validates with `json.load` (container **exits non-zero** if decode/parse fails)
3. Forces `GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa-key.json` and removes a stale `/tmp/gcp-sa.json` if present

Do **not** set `GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json` in the Railway dashboard (that path caused production “not a valid json file” 500s). Prefer leaving `GOOGLE_APPLICATION_CREDENTIALS` unset and relying on BASE64 + entrypoint.

**Python bootstrap (import time):** [`ml/rag/gcp_credentials.py`](../ml/rag/gcp_credentials.py) re-decodes `GOOGLE_APPLICATION_CREDENTIALS_BASE64` on every process start so Railway platform env injection cannot leave ADC pointing at stale `/tmp/gcp-sa.json` after the shell entrypoint.

`GET /ready` nests a `bq` object: when `BQ_PROJECT` is set, invalid/missing SA JSON or a failed BigQuery probe → `status: not_ready`.

### Troubleshooting — `File /tmp/gcp-sa.json is not a valid json file`

If `/query` returns:

```json
{"detail": "('File /tmp/gcp-sa.json is not a valid json file.', JSONDecodeError(...))"}
```

This was caused by a Railway **Custom Start Command** that decoded `GCP_SA_JSON_B64` into `/tmp/gcp-sa.json` (empty file if that var is unset). [`railway.toml`](../railway.toml) now pins `startCommand` to `sh -c 'uvicorn ml.rag.api:app --host 0.0.0.0 --port ${PORT:-7860}'`; the entrypoint writes `/tmp/gcp-sa-key.json`.

1. **Clear** the dashboard Custom Start Command (after deploy it should show “set in railway.toml”).
2. **Set** `GOOGLE_APPLICATION_CREDENTIALS_BASE64` — copy from `GCP_SA_JSON_B64` if that is where the key lives, or re-encode with `ml-eng/scripts/encode-gcp-key.sh` (single line, no quotes).
3. **Delete** `GCP_SA_JSON_B64`, `GCP_SA_JSON`, and `GOOGLE_APPLICATION_CREDENTIALS` (especially `/tmp/gcp-sa.json`).
4. Confirm `BQ_PROJECT` and `BQ_DATASET_SILVER=staging_dev`.
5. **Redeploy** (`Dockerfile.railway`).
6. **Verify** `GET /ready` → `bq.gcp.path` is `/tmp/gcp-sa-key.json`, `bq.gcp.json_ok: true`, `status: ready`.

If the container fails to start after deploy, check deploy logs for entrypoint `✗` lines (invalid BASE64 or missing creds when `BQ_PROJECT` is set).

### Reranker (local cross-encoder via fastembed — no torch)

**Required on Railway** because OpenRouter LLM config auto-promotes to OpenRouter `/rerank` when `RAG_RERANKER_MODE` is unset.

```bash
RAG_RERANKER_MODE=cross_encoder
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
```

The rerank model is **baked into** `Dockerfile.railway` at build time (~280 MB ONNX). Plan for ~300 MB extra RAM at runtime.

**Alternatives (not primary Railway guidance):** Cohere SDK (`RAG_RERANKER_MODE=cohere` + `COHERE_API_KEY`) or OpenRouter `/rerank` (`RAG_RERANKER_MODE=openrouter` + `RAG_RERANK_MODEL_ID=cohere/rerank-4-pro`).

---

## Set — recommended

### BQ YAML schema packs (baked into image)

```bash
RAG_BQ_SKIP_LIVE_SCHEMA=on
RAG_BQ_NL2SQL_MODE=per_hint
RAG_BQ_MAX_TABLES=6
RAG_BQ_MAX_SQL_QUERIES=10
RAG_BQ_NL2SQL_PARALLEL=on
RAG_BQ_NL2SQL_PARALLEL_WORKERS=4
RAG_BQ_HINT_MAX_BYTES=8000
RAG_BQ_REASONER_INDEX_MAX_BYTES=12000
RAG_BQ_CONTEXT_MAX_BYTES=6000
```

### Vector retrieval (corpus router + constraints)

```bash
RAG_CORPUS_ROUTER=on
RAG_NEWS_DOMAIN_FILTER=on
# For strict country/year QA, disable geo fallbacks (empty > wrong geography):
# RAG_NEWS_GEO_FALLBACK=off
# RAG_RESEARCH_GEO_FALLBACK=off
```

### OpenRouter (optional)

```bash
OPENROUTER_HTTP_REFERER=https://opentrace.africa
OPENROUTER_APP_TITLE=Ask ADZA
# RAG_OPENROUTER_SESSION_ID=on   # default; bundles LLM calls per Langfuse trace
```

### Scaling / Redis (API + Redis same Railway environment)

API and Redis must share the **same** Railway environment so private networking works. Prefer the private hostname (or Railway’s `${{Redis.REDIS_URL}}` reference). Do **not** use `REDIS_PUBLIC_URL` / `*.proxy.rlwy.net` for API↔Redis.

```bash
# Private network (example — use your Redis service name):
RAG_REDIS_URL=redis://redis.railway.internal:6379/0
# Or Railway variable reference from the Redis service:
# RAG_REDIS_URL=${{Redis.REDIS_URL}}

# Session blobs: 7 days. If Railway already has 86400, change it or the dashboard wins.
RAG_SESSION_TTL_SECONDS=604800
RAG_CACHE_TTL_SECONDS=3600
# Signed artifact download URLs (24h). If Railway still has 3600, change it or the dashboard wins.
RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS=86400
# RAG_REDIS_CONNECT_TIMEOUT_S=2
```

Confirm via `GET /ready` → `redis.connected: true` (or API logs: `session_store: connected to Redis`).

### Langfuse (optional)

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=production
```

Embeddings, hybrid search, and rerank models are **baked** in `Dockerfile.railway` (`RAG_EMBEDDINGS_MODE=fastembed`, hybrid on). Override only if your Qdrant index differs.

---

## Pre-deploy checklist

1. **Build**: redeploy with **build cache cleared** after YAML / graph / Dockerfile warmup changes.
2. **Qdrant**: all six collections exist and are populated (OTA may start empty).
3. **Ready**: `GET /ready` returns `ready` (needs Qdrant + LLM creds; when `BQ_PROJECT` is set also needs valid GCP SA + BigQuery reachability — see nested `bq`).
4. **Smoke queries**:
   - Meta: `{"query":"Who are you?"}`
   - Retrieval: maize or policy question — news / academic / policies paths populated
   - BQ: e.g. fertilizer use or maize production in Nigeria — BQ uses `staging_dev` tables
   - Rerank: Langfuse / response metadata shows `rerank_mode=cross_encoder` on a full_rag query

---

## Streamlit QA service (mirror API pipeline; Redis exception)

Streamlit runs in-process `run_rag()` — mirror **pipeline** vars from the API (Qdrant, LLM, BQ, GCP, rerank, Langfuse, etc.).

**Exception — Redis:** when Streamlit is in a **different** Railway environment than API+Redis, private DNS (`*.railway.internal`) does not reach Redis. Leave `RAG_REDIS_URL` / `REDIS_URL` **unset** on Streamlit (in-process memory fallback is fine for solo QA). Do not copy the API’s private Redis URL.

1. Copy API pipeline vars into the Streamlit environment (manually or via Railway share/copy).
2. **Omit** `RAG_REDIS_URL` / `REDIS_URL`.
3. Add **only** these QA flags:

```bash
RAG_STREAMLIT_DEBUG_DEFAULT=1
RAG_SHOW_SQL_DEBUG=1
```

Do **not** use a different rerank mode or omit rerank vars. See [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md).

Do **not** expose Streamlit as the public Ask ADZA UI.

---

## Railway apply (manual)

After merging doc/Dockerfile changes:

1. **API service (same env as Redis):**
   - Confirm Settings → Start command is **set in railway.toml** (`sh -c 'uvicorn ml.rag.api:app --host 0.0.0.0 --port ${PORT:-7860}'`). Clear any dashboard command that writes `/tmp/gcp-sa.json`.
   - Set `GOOGLE_APPLICATION_CREDENTIALS_BASE64` (copy from `GCP_SA_JSON_B64` or re-encode with `scripts/encode-gcp-key.sh`).
   - **Delete** `GCP_SA_JSON_B64`, `GCP_SA_JSON`, and `GOOGLE_APPLICATION_CREDENTIALS` if present (never `/tmp/gcp-sa.json`).
   - Set `RAG_RERANKER_MODE=cross_encoder` and `RAG_RERANKER_MODEL=BAAI/bge-reranker-base`; set private `RAG_REDIS_URL`; delete Cohere/OpenRouter rerank vars.
2. **Streamlit service (other env):** copy pipeline vars from API; **omit** Redis; add QA debug flags only.
3. Redeploy **both** with **build cache cleared**.
4. Smoke API (`GET /ready` → `bq.gcp.json_ok: true`, path `/tmp/gcp-sa-key.json`, Redis connected) and Streamlit (same query; inspector shows rerank scores).

---

## Related

- [README.md](./README.md) — GCE / Docker prod image
- [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md) — QA Streamlit on Railway
- [ml/rag/README.md](../ml/rag/README.md) — RAG pipeline overview
