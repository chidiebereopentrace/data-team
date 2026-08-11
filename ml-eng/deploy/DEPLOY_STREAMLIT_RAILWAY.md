# Deploy Streamlit Pipeline Inspector on Railway

Internal QA UI for full RAG observability. **Not** the production Ask ADZA surface — keep the existing API service (`Dockerfile.railway`) as the user-facing backend.

## Overview

| Service | Dockerfile | Config | Port | Health |
|---------|------------|--------|------|--------|
| RAG API (existing) | `Dockerfile.railway` | `railway.toml` | 8080 | `/health`, `/ready` |
| Streamlit QA (new) | `Dockerfile.railway.streamlit` | `railway.streamlit.toml` | 8501 | `/_stcore/health` |

Streamlit runs **in-process** `run_rag()` by default so the inspector shows full chunk lists. Optional HTTP mode (`RAG_API_BASE_URL`) only returns trace counts.

Streamlit may live in a **different Railway environment** than the API + Redis pair. Private networking does not cross environments.

## Mirror API (required)

Streamlit is **not** a slimmed-down config. Retrieval, BQ, LLM, and rerank must match the API exactly.

1. Deploy and validate the **API** service first ([PRODUCTION_ENV.md](./PRODUCTION_ENV.md)).
2. Create the Streamlit service (same project or another environment).
3. Copy **pipeline** vars from the API (Qdrant, LLM, BQ, GCP, rerank, Langfuse, etc.).
4. **Omit** `RAG_REDIS_URL` / `REDIS_URL` when Streamlit is not in the same environment as Redis (private `*.railway.internal` is unreachable). Solo QA uses in-process memory.
5. Add **only** these QA flags:

```bash
RAG_STREAMLIT_DEBUG_DEFAULT=1
RAG_SHOW_SQL_DEBUG=1
```

**Do not:**

- Set a different `RAG_RERANKER_MODE` than the API (both use `cross_encoder` + `BAAI/bge-reranker-base`)
- Omit rerank vars (OpenRouter LLM config auto-promotes to OpenRouter rerank if mode is unset)
- Copy the API’s private Redis URL into a different Streamlit environment
- Maintain a separate minimum-var checklist — use [PRODUCTION_ENV.md](./PRODUCTION_ENV.md) as the single source of truth (with the Redis exception above)

The Streamlit image (`Dockerfile.railway.streamlit`) bakes the **same** embedding and cross-encoder models as the API image.

## 1. Create the Railway service

1. In Railway, create a service from the same GitHub repo (same project or a QA environment).
2. Set **Root Directory** to `ml-eng/`.
3. Under **Settings → Config-as-code**, set config file to `railway.streamlit.toml`  
   (or manually set **Dockerfile Path** to `Dockerfile.railway.streamlit`).
4. Use a **memory tier ≥ 1 GB** — embeddings + rerank model + in-process graph can OOM on 512 MB (exit 137).

## 2. Environment variables

Copy **pipeline** vars from the API service (see [PRODUCTION_ENV.md](./PRODUCTION_ENV.md)), then add the QA flags above. Do **not** set `RAG_REDIS_URL` for cross-environment Streamlit.

### Baked in image (override only if needed)

| Variable | Default in image |
|----------|------------------|
| `RAG_EMBEDDINGS_MODE` | `fastembed` |
| `RAG_SPARSE_EMBEDDINGS` | `on` |
| `RAG_QDRANT_HYBRID_SEARCH` | `on` |

### Optional (QA / ops)

| Variable | Use |
|----------|-----|
| `RAG_API_BASE_URL` | Point sidebar at deployed API for HTTP-only traces |
| `LANGFUSE_*` | Tracing (`LANGFUSE_BASE_URL=https://cloud.langfuse.com`) |
| `LANGFUSE_TRACING_RELEASE` | Deploy tag (git SHA) for regression comparison |

Redis (`RAG_REDIS_URL`) is **not** used on Streamlit in the cross-environment layout. API + Redis stay private in the production environment.

**Do not** mount local key paths (`GOOGLE_APPLICATION_CREDENTIALS=config/keys/...`). Use `GOOGLE_APPLICATION_CREDENTIALS_BASE64` only.

## 3. Build and deploy

Railway builds from `ml-eng/` using `Dockerfile.railway.streamlit`. The image pre-warms fastembed embedding + rerank models (~60s start period).

**Local prod-like build:**

```bash
cd ml-eng
docker build -f Dockerfile.railway.streamlit -t opentrace-rag-streamlit:latest .
```

**Clear build cache** on Railway after changes to `requirements.railway.txt`, Dockerfile warmup, or embedding/rerank config.

Redeploy **both** API and Streamlit when rerank or model-bake changes land.

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

Confirm **latency**, **token usage**, and **rerank scores** appear after a full_rag turn. Rerank behavior should match the API for the same query.

## 6. Access control

Treat the public Railway URL as **team-internal only**:

- Do not link from production Ask ADZA.
- Prefer Railway private networking or IP allowlists if available.
- Streamlit has **no built-in auth** — add a reverse proxy or VPN if the URL is on the public internet.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Health check fails / 502 | Still starting (model warmup) | Wait 60–90s; check deploy logs |
| Exit 137 | OOM | Bump Railway memory to ≥ 1 GB |
| BQ tab empty / SQL errors | Missing GCP creds | Set `GOOGLE_APPLICATION_CREDENTIALS_BASE64` |
| All retrieval empty | Wrong Qdrant env or empty collections | Copy pipeline vars from API; run ingestion |
| Rerank differs from API | Streamlit env not mirrored | Copy pipeline vars from API; set `RAG_RERANKER_MODE=cross_encoder` |
| Redis connect errors / timeouts on Streamlit | Private `RAG_REDIS_URL` copied from another env | Unset `RAG_REDIS_URL` / `REDIS_URL` on Streamlit |
| Inspector empty in HTTP mode | API `include_trace` is counts-only | Use in-process mode (default) or inspect API logs |
| LLM errors | Missing / wrong `RAG_LLM_*` | Copy from working API service |

## Related docs

- [ml/rag/README.md](../ml/rag/README.md) — Streamlit inspector features and local run
- [deploy/PRODUCTION_ENV.md](./PRODUCTION_ENV.md) — canonical production env checklist
- [deploy/README.md](./README.md) — Production API on GCE
- [config/.env.example](../config/.env.example) — full env reference
