# Production environment variables (RAG API)

Canonical checklist for **Railway**, **GCE / Cloud Run**, or any 12-factor deploy of `ml.rag.app.api`. Apply vars in your platform dashboard; this file is not loaded at runtime.

For Streamlit QA (second Railway service), copy the same vars and add the Streamlit section at the bottom. See [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md).

Full reference: [config/.env.example](../config/.env.example).

---

## Remove (obsolete or harmful)

| Variable | Why |
|----------|-----|
| `QDRANT_COLLECTION_DATA_DESCRIPTIONS` | Removed; BQ table selection uses staging YAML only |
| `NEWS_PUBLIC_REPORTS` or similar legacy alias | Use `QDRANT_COLLECTION_PUBLIC_REPORTS=public_reports` |
| `GOOGLE_APPLICATION_CREDENTIALS=config/keys/...` | Invalid on Railway; use `GOOGLE_APPLICATION_CREDENTIALS_BASE64` |
| `RAG_LLM_BASE_URL` pointing at LAN / LM Studio | Production must use a public LLM endpoint (e.g. OpenRouter) |
| Per-plan **70B** overrides | e.g. `RAG_LLM_MODEL_GOVERNMENT=...70b...`, `RAG_BQ_REASONER_MODEL_ID=...70b...` |
| `RAG_LLM_RERANK=off` as primary rerank guidance | Use `RAG_RERANKER_MODE` + Cohere or OpenRouter rerank instead |

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

### LLM (OpenRouter — all plan tiers use 8B)

```bash
RAG_LLM_BASE_URL=https://openrouter.ai/api/v1
RAG_LLM_API_KEY=<OPENROUTER_API_KEY>
RAG_LLM_MODEL_ID=meta-llama/llama-3.1-8b-instruct
RAG_SUMMARY_MODEL_ID=meta-llama/llama-3.1-8b-instruct
RAG_LLM_TIMEOUT_S=300
```

Unset or align per-plan vars so they do not override to a larger model:

- `RAG_LLM_MODEL_FREE`, `RAG_LLM_MODEL_FARMERS`, `RAG_LLM_MODEL_GOVERNMENT`, etc.

### BigQuery (staging YAML reasoner + NL2SQL)

```bash
BQ_PROJECT=opentrace-prod-5ga4
BQ_DATASET_SILVER=staging_dev
GOOGLE_APPLICATION_CREDENTIALS_BASE64=<base64 of GCP service account JSON>
```

RAG NL-to-SQL queries **`staging_dev`** only (`BQ_DATASET_SILVER`). `BQ_DATASET_BRONZE` is for data-eng tooling, not the live RAG path.

Encode key: [scripts/encode-gcp-key.sh](../scripts/encode-gcp-key.sh).

### Reranker (Railway slim image — no torch cross-encoder)

```bash
COHERE_API_KEY=<COHERE_API_KEY>
RAG_RERANKER_MODE=cohere
```

Alternative on OpenRouter:

```bash
RAG_RERANKER_MODE=openrouter
RAG_RERANK_MODEL_ID=cohere/rerank-4-pro
```

---

## Set — recommended

### BQ YAML schema packs (baked into image)

```bash
RAG_BQ_SKIP_LIVE_SCHEMA=on
RAG_BQ_NL2SQL_MODE=per_hint
RAG_BQ_MAX_TABLES=4
RAG_BQ_MAX_SQL_QUERIES=10
RAG_BQ_HINT_MAX_BYTES=8000
RAG_BQ_REASONER_INDEX_MAX_BYTES=12000
RAG_BQ_CONTEXT_MAX_BYTES=6000
```

### OpenRouter (optional)

```bash
OPENROUTER_HTTP_REFERER=https://opentrace.africa
OPENROUTER_APP_TITLE=Ask ADZA
# RAG_OPENROUTER_SESSION_ID=on   # default; bundles LLM calls per Langfuse trace
```

### Scaling (multi-replica)

```bash
RAG_REDIS_URL=redis://...        # or rediss://...
RAG_SESSION_TTL_SECONDS=86400
RAG_CACHE_TTL_SECONDS=3600
```

### Langfuse (optional)

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=production
```

Embeddings/hybrid are **baked** in `Dockerfile.railway` (`RAG_EMBEDDINGS_MODE=fastembed`, hybrid on). Override only if your Qdrant index differs.

---

## Pre-deploy checklist

1. **Build**: redeploy with **build cache cleared** after YAML / graph changes (FAOSTAT enums ship in the image; no extra env vars).
2. **Qdrant**: all six collections exist and are populated (OTA may start empty).
3. **Ready**: `GET /ready` returns `ready` (needs Qdrant + LLM creds).
4. **Smoke queries**:
   - Meta: `{"query":"Who are you?"}`
   - Retrieval: maize or policy question — inspector shows news / academic / policies paths
   - BQ: e.g. fertilizer use or maize production in Nigeria — BQ tab uses `staging_dev` tables

---

## Streamlit QA service (additions)

Copy all API vars above, then:

```bash
RAG_STREAMLIT_DEBUG_DEFAULT=1
RAG_SHOW_SQL_DEBUG=1
```

Do **not** expose Streamlit as the public Ask ADZA UI.

---

## Related

- [README.md](./README.md) — GCE / Docker prod image
- [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md) — QA Streamlit on Railway
- [ml/rag/README.md](../ml/rag/README.md) — RAG pipeline overview
