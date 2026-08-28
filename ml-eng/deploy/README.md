# RAG API Production Deployment Guide (GCE + Docker)

This guide is for the **production FastAPI surface** (`ml.rag.app.api`).

**Environment variables:** use the canonical checklist in [PRODUCTION_ENV.md](./PRODUCTION_ENV.md) (six Qdrant collections, staging BQ, OpenRouter 8B, local cross_encoder rerank on Railway).

**Streamlit** is for **internal QA only** (local or Railway second service). Do not expose it as the Ask ADZA user-facing chat. See [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md) (step-by-step) and [ml/rag/README.md](../ml/rag/README.md) — *Deploy Streamlit on Railway*.

## 1. Build the production image

From the `data-team` root:

```bash
docker build -f ml-eng/Dockerfile -t opentrace-rag-api:latest .
```

The image is intentionally slim and contains **no** .env files or secrets.

## 2. Run locally (prod-like, 12-factor)

```bash
docker run --rm -p 8080:8080 \
  -e QDRANT_URL="https://..." \
  -e QDRANT_API_KEY="..." \
  -e RAG_LLM_BASE_URL="http://host.docker.internal:1234/v1" \
  -e RAG_LLM_MODEL_ID="qwen/qwen3-30b-a3b-instruct-2507" \
  -e RAG_GENERATE_TEMPERATURE="0.5" \
  -e BQ_PROJECT="opentrace-prod-5ga4" \
  -e BQ_DATASET_SILVER="staging_dev" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/opentrace-bq-key.json" \
  -v /path/to/bq-key.json:/secrets/opentrace-bq-key.json:ro \
  opentrace-rag-api:latest
```

Test:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Rice production trends in Nigeria since 2020"}'
```

## 3. Secret & configuration management on GCE

**Recommended pattern (12-factor)**: Never bake secrets. Inject at runtime.

Options:

### A. Google Secret Manager + mounted files (common on GCE VMs)

- Create secrets in Secret Manager.
- Use the Secret Manager sidecar or a startup script to write files to a tmpfs or dedicated dir.
- Mount the dir read-only into the container.
- Set `GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/...` etc.

### B. Cloud Run (often simpler for this service)

- Deploy the container to Cloud Run.
- Set environment variables directly in the Cloud Run service (supports Secret Manager references).
- For the BQ key, use a mounted secret volume or Workload Identity + ADC.

Example Cloud Run deploy (conceptual):

```bash
gcloud run deploy opentrace-rag-api \
  --image ... \
  --set-env-vars="RAG_LLM_BASE_URL=...,RAG_GENERATE_TEMPERATURE=0.5,..." \
  --set-secrets="QDRANT_API_KEY=projects/.../secrets/qdrant-api-key:latest" \
  --memory 2Gi \
  --cpu 2 \
  --port 8080
```

## 4. Health & readiness

The image exposes:

- `GET /health` — liveness (always fast)
- `GET /ready`  — readiness (checks presence of critical env vars without connecting)

Use `/ready` for GCE health checks / load balancer backends.

## 5. AskADZA integration notes (API contract)

See the OpenAPI docs at `/docs` once the service is running.

Important fields the AskADZA client should send on every `/query`:

- `query`
- `session_id` (for multi-turn; now durable across workers/restarts when Redis is configured)
- `conversation_history` (fully stateless alternative — send full prior turns from the client; server does not persist)
- `stakeholder_type` (optional, e.g. "government_public")
- `audience_instructions` (optional free-form text from the client UI)

The RAG service will forward these to the generator. The client (AskADZA) owns the user profile and tone logic.

## 6. Model caches (fastembed / HF)

The production image does **not** contain the embedding models.

Recommended:

- Use a persistent disk / volume mounted at the path you set for `FASTEMBED_CACHE_PATH` and `HF_HOME`.
- Or let the service download on cold start (first query will be slow).

Set the env vars accordingly in your GCE / Cloud Run config.

## 7. Redis for sessions & caches (required for scaled deployments)

The production image does **not** bundle Redis. Deploy Redis separately (Google Memorystore for Redis, Upstash, Railway Redis, or a small sidecar/container) and point `RAG_REDIS_URL` at it.

**Railway:** put Redis in the **same environment** as the API and use the private URL, e.g. `redis://redis.railway.internal:6379/0` or `${{Redis.REDIS_URL}}`. Do not use `REDIS_PUBLIC_URL` / `*.proxy.rlwy.net` for API↔Redis. Streamlit in a **different** environment should leave `RAG_REDIS_URL` unset — see [PRODUCTION_ENV.md](./PRODUCTION_ENV.md) and [DEPLOY_STREAMLIT_RAILWAY.md](./DEPLOY_STREAMLIT_RAILWAY.md).

Example environment (Cloud Run or GCE startup script / Secret Manager):

```
RAG_REDIS_URL=redis://10.x.x.x:6379/0          # or rediss:// for TLS + auth
RAG_SESSION_TTL_SECONDS=604800
RAG_CACHE_TTL_SECONDS=3600
RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS=86400
RAG_REDIS_CONNECT_TIMEOUT_S=2
```

- Chat sessions (`session_id` continuity) and the BQ schema / bronze catalog caches are now stored in Redis under `rag:session:*`, `rag:bq:schema:*`, `rag:catalog:bronze:*`.
- TTLs are honored; keys are JSON.
- If Redis is unreachable the service logs a warning once and falls back to per-process in-memory storage (sessions lost across workers/restarts; caches rebuilt per worker). This keeps dev and single-worker deploys simple.
- `/ready` includes a `redis` object when `RAG_REDIS_URL` is set (non-fatal).
- For local prod-like testing: see `docker-compose.prod.yml` (uncomment the `redis` service and the env var).

**Stateless alternative**: clients (AskADZA) can ignore server sessions entirely and always send `conversation_history` (array of {role, content}) on every `/query`. The server will compact it for that request only and not persist anything.

Update the AskADZA client to treat `session_id` as durable now that Redis backs it.

## 7.5 Langfuse tracing (optional, SDK v3+)

When Langfuse keys are set, each `/query` emits a unified trace. When absent, tracing is a silent no-op (zero overhead). The API flushes pending traces **after each request** and on shutdown. Push + redeploy the existing Railway RAG service — no Dockerfile change.

**Required env vars:**

| Variable | Value |
|----------|--------|
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` (EU Cloud) |

**Recommended for production:**

| Variable | Value |
|----------|--------|
| `LANGFUSE_TRACING_ENVIRONMENT` | `production` |
| `LANGFUSE_TRACING_RELEASE` | git SHA or version tag (optional; else `RAILWAY_GIT_COMMIT_SHA`) |
| `LANGFUSE_TRACING_SAMPLE_RATE` | `1.0` (optional; lower for high volume) |

**Expected spans** (when the corresponding path runs): root `rag.query`, LangGraph nodes, `decompose` / `merge` / `web_fallback` / `insufficient_context`, `retrieval.*`, `rerank`, `embedding.query`, and `llm_chat_complete` generations (`purpose`: `decompose`, `bq.nl2sql`, `generate`, …). Metadata includes `session_id`, optional `user_id`, soft-fail flags, and release tags.

**OpenRouter Sessions (LLM cost bundling):** When `RAG_LLM_BASE_URL` points at OpenRouter, leave `RAG_OPENROUTER_SESSION_ID` unset or `on` so decompose + NL2SQL + generate + `/rerank` share one `session_id` (= Langfuse trace id) in [OpenRouter Logs](https://openrouter.ai/logs?tab=sessions). Set `RAG_OPENROUTER_SESSION_ID=off` to disable.

**User feedback:** `POST /feedback` with `{ "trace_id": "...", "score": 1.0, "comment": "..." }` (score 0–1) when tracing is enabled. `/v1/chat` returns `langfuse_trace_id`.

**Verify:** `python scripts/verify_langfuse_tracing.py` (from `ml-eng/` with keys loaded).

## 8. Scaling & resources

**Important for multi-turn continuity**: the RAG service is stateless except for the server-side session store. With multiple workers (gunicorn) or replicas (MIG / Cloud Run), you **must** configure Redis (`RAG_REDIS_URL`) or clients must always pass the full `conversation_history` on every request (the stateless escape hatch).

- CPU: 1–2 vCPU is usually enough for the graph (most work is in the remote LLM and Qdrant).
- Memory: 1.5–2 GiB minimum (embeddings + graph state). Add headroom for Redis client and any in-memory fallback if Redis is temporarily unavailable.
- For production, consider gunicorn with multiple Uvicorn workers (see Dockerfile comments). Each worker is independent; Redis makes sessions and caches coherent.
- Readiness (`/ready`) reports Redis status (when configured) for observability but does not fail the check on Redis problems (graceful fallback).

## 9. OTA insights (included in initial handoff — collection starts empty)

The OTA retriever **is wired** as part of the v1 production API surface.

- At query time the system retrieves from the `OTA_insights` collection in Qdrant (using the existing ota_triple vector support: insight / metric / recommendation vectors).
- The collection is expected to start **empty** on day one of production. Analysts are still producing the source content by analyzing the BQ bronze layer; they will ingest it using the existing OTA ingestion pipeline (`ota_insights_preprocessor` + `ota_insights_load_to_vector_db`).
- When results are present they are merged into the main generation context (so the model can synthesize them with news/research/structured data) while clearly retaining citations as "[OTA Insight]", "[OTA Metric]", or "[OTA Recommendation]".
- Graceful empty behavior: the retriever simply returns [] and answers omit the OTA section until data arrives.

There is **no** direct BQ query path for OTA at runtime — only the vector retriever over the pre-ingested `OTA_insights` collection.

For the AskADZA v1 handoff, the full set of sources is: News + Research + BQ Structured Data + OTA (initially empty).

## 10. Next phases (after initial handoff)

- Analysts populate the `OTA_insights` Qdrant collection (data + ingestion work).
- Full BQ YAML schema coverage for more tables (improves NL→SQL quality).
- Optional server-side default tone mapping using `stakeholder_prompts.py` (only if the client later asks for it).

Contact the data-team / ML platform team for upgrades.

---

This deployment story, together with the hardened `app/api.py`, gives the software team everything they need to run the RAG service on GCE and integrate it into AskADZA.