# Deploy Streamlit Pipeline Inspector on Railway

Internal QA UI for full RAG observability. **Not** the production Ask ADZA surface — keep the existing API service (`Dockerfile.railway`) as the user-facing backend.

## Overview

| Service | Dockerfile | Config | Port | Health |
|---------|------------|--------|------|--------|
| RAG API (existing) | `Dockerfile.railway` | `railway.toml` | 8080 | `/health`, `/ready` |
| Streamlit QA (new) | `Dockerfile.railway.streamlit` | `railway.streamlit.toml` | 8501 | `/_stcore/health` |

Streamlit runs **in-process** `run_rag()` by default so the inspector shows full chunk lists. Optional HTTP mode (`RAG_API_BASE_URL`) only returns trace counts.

## 1. Create the Railway service

1. In the same Railway project as the RAG API, click **New Service** → **GitHub Repo** (same repo).
2. Set **Root Directory** to `ml-eng/`.
3. Under **Settings → Config-as-code**, set config file to `railway.streamlit.toml`  
   (or manually set **Dockerfile Path** to `Dockerfile.railway.streamlit`).
4. Use a **memory tier ≥ 1 GB** — embedding warmup + first query can OOM on 512 MB (exit 137).

## 2. Environment variables

Copy from the API service, then add Streamlit-specific vars.

### Required (same as API)

| Variable | Notes |
|----------|--------|
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud |
| `RAG_LLM_BASE_URL` | e.g. `https://openrouter.ai/api/v1` |
| `RAG_LLM_API_KEY` | OpenRouter key |
| `RAG_LLM_MODEL_ID` | e.g. `openrouter/owl-alpha` |
| `BQ_PROJECT` / `BQ_DATASET_BRONZE` | BigQuery bronze |
| `GOOGLE_APPLICATION_CREDENTIALS_BASE64` | Base64 GCP SA JSON ([encode script](../scripts/encode-gcp-key.sh)) |

### Baked in image (override only if needed)

| Variable | Default in image |
|----------|------------------|
| `RAG_EMBEDDINGS_MODE` | `fastembed` |
| `RAG_SPARSE_EMBEDDINGS` | `on` |
| `RAG_QDRANT_HYBRID_SEARCH` | `on` |

### Recommended for QA

| Variable | Value |
|----------|--------|
| `RAG_STREAMLIT_DEBUG_DEFAULT` | `1` — inspector panel on after each turn |
| `RAG_SHOW_SQL_DEBUG` | `1` — show generated SQL in inspector |

### Optional

| Variable | Use |
|----------|-----|
| `RAG_API_BASE_URL` | Point sidebar at deployed API for HTTP-only traces |
| `RAG_REDIS_URL` | Session store (usually not needed for solo QA) |
| `LANGFUSE_*` | Tracing |
| `RAG_LLM_RERANK` | `off` unless rerank endpoint is configured |

**Do not** mount local key paths (`GOOGLE_APPLICATION_CREDENTIALS=config/keys/...`). Use `GOOGLE_APPLICATION_CREDENTIALS_BASE64` only.

## 3. Build and deploy

Railway builds from `ml-eng/` using `Dockerfile.railway.streamlit`. The image pre-warms fastembed models (~60s start period).

**Local prod-like build:**

```bash
cd ml-eng
docker build -f Dockerfile.railway.streamlit -t opentrace-rag-streamlit:latest .
```

**Clear build cache** on Railway after changes to `requirements.railway.txt` or embedding config.

## 4. Post-deploy smoke test

Replace with your Railway public URL (or internal hostname):

```bash
./scripts/smoke_streamlit_railway.sh https://your-streamlit-qa.up.railway.app
```

PowerShell:

```powershell
curl.exe -fsS "https://your-streamlit-qa.up.railway.app/_stcore/health"
```

Expected: HTTP 200, body `ok`.

## 5. Manual QA checklist

Open the app URL in a browser. Use sidebar **preset queries** and confirm the inspector **route** column:

| Query | Expected route | Retrieval |
|-------|----------------|-----------|
| `Who are you?` | `meta` | Zero |
| `What is OpenTrace?` | `product` | Zero |
| `Maize yields in Kenya 2020` | `full_rag` | News / research / BQ tabs populated |
| Farmers + Ghana preset | `full_rag` | Geo in decomposition |

Confirm **latency** and **token usage** appear after a full_rag turn.

## 6. Access control

Treat the public Railway URL as **team-internal only**:

- Do not link from production Ask ADZA.
- Prefer Railway private networking or IP allowlists if available.
- Streamlit has **no built-in auth** — add a reverse proxy or VPN if the URL is on the public internet.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Health check fails / 502 | Still starting (embedding warmup) | Wait 60–90s; check deploy logs |
| Exit 137 | OOM | Bump Railway memory to ≥ 1 GB |
| BQ tab empty / SQL errors | Missing GCP creds | Set `GOOGLE_APPLICATION_CREDENTIALS_BASE64` |
| All retrieval empty | Wrong Qdrant env or empty collections | Match API service vars; run ingestion |
| Inspector empty in HTTP mode | API `include_trace` is counts-only | Use in-process mode (default) or inspect API logs |
| LLM errors | Missing / wrong `RAG_LLM_*` | Copy from working API service |

## Related docs

- [ml/rag/README.md](../ml/rag/README.md) — Streamlit inspector features and local run
- [deploy/README.md](./README.md) — Production API on GCE
- [config/.env.example](../config/.env.example) — full env reference
