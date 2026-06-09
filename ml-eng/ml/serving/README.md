# Serving (exposition API)

Public, versioned HTTP API for the OpenTrace chatbot. Retrieval internals (`bq_sql`, trace fields, tuning knobs) stay on the internal app [`ml/rag/app/api.py`](../rag/app/api.py) (served as `ml.rag.api:app`, `POST /query`). Full endpoint reference: [`ml/rag/docs/API.md`](../rag/docs/API.md).

## Endpoints (v1)

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/v1/health` | Liveness |
| GET | `/v1/meta` | `api_version`, `schema_version`, optional `build`, `stakeholder_types` catalog |
| POST | `/v1/sessions` | Body: `stakeholder_type` → `session_id` |
| POST | `/v1/chat` | Body: `query` or `message` + `session_id`, nested `user_profile`, optional `chat_history` |

**Canonical `POST /v1/chat` request** (same nesting as `/query`; user text via **`query`** or **`message`**):

```json
{
  "query": "What are rice yield trends?",
  "session_id": "abc123...",
  "user_profile": {
    "country": "Ghana",
    "audience_instructions": null,
    "stakeholder_type": "farmers_communities"
  },
  "chat_history": [
    { "role": "user", "content": "Previous question" },
    { "role": "assistant", "content": "Previous answer" }
  ]
}
```

`user_profile.country` filters retrieval **only** for **`farmers_communities`**. Deprecated: top-level `stakeholder_type` (bootstrap only), `conversation_history` (use `chat_history`).

OpenAPI: `/docs` on the same app.

## Run locally

From repo root (same dependencies as RAG: `requirements.txt` + `ml/rag/requirements.txt`):

```bash
PYTHONPATH=. uvicorn ml.serving.chat.app:app --host 0.0.0.0 --port 7861
```

Internal RAG API (debug / observability) on the same port convention as Hugging Face Spaces:

```bash
PYTHONPATH=. uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860
```

## Hugging Face Spaces (public chat API)

Use the **repo root [`Dockerfile`](../../Dockerfile)**. It runs **`ml.serving.chat.app`** on **port 7860** with [`scripts/hf-entrypoint.sh`](../../scripts/hf-entrypoint.sh) so GCP credentials can come from Space secrets.

**Internal RAG** (`ml.rag.api`, `POST /query`) for monitoring stays on **[`Dockerfile.rag`](../../Dockerfile.rag)** / Compose `rag-api`, not the root `Dockerfile`.

### Build prerequisites

1. **Qdrant:** RAG vector search uses **Qdrant** (see [`ml/rag/README.md`](../rag/README.md)). Set **`QDRANT_URL`**, **`QDRANT_API_KEY`**, and collection env vars in Space secrets or `data/local/.env`. Populate collections with the ingestion CLI or loaders in that README—not a local `vector_db` directory.

2. **From repo root:**

```bash
docker build -t opentrace-chat-hf .
```

### Space configuration

- **SDK:** Docker Space; **Dockerfile** at repo root (this repo already provides it).
- **Port:** **7860** (required by Spaces).

### Secrets and environment (Space → Settings → Variables and secrets)

| Variable | Purpose |
|----------|---------|
| `BQ_PROJECT` | GCP project ID for BigQuery |
| `BQ_DATASET_BRONZE` | Bronze dataset the RAG queries |
| `HF_API_TOKEN` | Hugging Face Inference API (LLM / embeddings as configured in RAG) |
| `QDRANT_URL` | Qdrant cluster URL (RAG vector retrieval) |
| `QDRANT_API_KEY` | Qdrant API key |
| `CHATBOT_CORS_ORIGINS` | Comma-separated browser origins (not `*` in production if you use credentials) |
| `GCP_SA_JSON` | Full GCP **service account JSON** (multiline). Written to `/tmp/gcp-sa.json` at startup; sets `GOOGLE_APPLICATION_CREDENTIALS`. |
| `GCP_SA_JSON_B64` | Same key, **base64-encoded** (use if the UI mangles multiline JSON). Takes precedence over `GCP_SA_JSON` when set. |

Optional: `CHATBOT_BUILD_ID` (shown on `/v1/meta`), `CHATBOT_DEBUG=1` (richer errors). For **HF API embeddings** (no local `sentence-transformers` load), set `RAG_EMBEDDINGS_MODE=hf_api` and see [`ml/rag/README.md`](../rag/README.md) (`RAG_EMBEDDING_MODEL_ID`).

If neither `GCP_SA_JSON` nor `GCP_SA_JSON_B64` is set, **`GOOGLE_APPLICATION_CREDENTIALS`** is left unchanged (e.g. local Docker with a mounted key file).

### GCP access (BigQuery)

Spaces run **outside** GCP; use a **service account key** via the table above (not GCE metadata). In GCP IAM, grant the SA at least:

- **`roles/bigquery.jobUser`** on the project (run queries).
- **`roles/bigquery.dataViewer`** on the **bronze** dataset (read tables), or tighter scopes your org allows.

The container needs HTTPS egress to **`bigquery.googleapis.com`**.

### Smoke test after deploy

Replace `YOUR_SPACE_URL` with your Space URL (e.g. `https://<user>-<space>.hf.space`).

```bash
curl -sS "$YOUR_SPACE_URL/v1/health"
curl -sS "$YOUR_SPACE_URL/v1/meta"
curl -sS -X POST "$YOUR_SPACE_URL/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{"stakeholder_type":"government_public"}'
# Use session_id from the response:
curl -sS -X POST "$YOUR_SPACE_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","session_id":"<session_id>"}'
```

Interactive docs: **`/docs`** on the same host.

### Local Docker (same image as HF)

```bash
docker run --rm -p 7860:7860 \
  -e BQ_PROJECT=... \
  -e BQ_DATASET_BRONZE=... \
  -e HF_API_TOKEN=... \
  -e GCP_SA_JSON="$(cat path/to/sa.json)" \
  opentrace-chat-hf
```

Or mount a key and omit `GCP_SA_JSON`:

```bash
docker run --rm -p 7860:7860 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp.json \
  -v /path/to/sa.json:/run/secrets/gcp.json:ro \
  -e BQ_PROJECT=... \
  -e BQ_DATASET_BRONZE=... \
  opentrace-chat-hf
```

See also [`ml/rag/README.md`](../rag/README.md) for BigQuery env and RAG tuning.
