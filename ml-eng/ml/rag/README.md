# RAG pipeline (graph / agentic)

Modular RAG that queries **BigQuery** and **Qdrant** (news, research, BQ table descriptions, OTA), then merges → reranks → generates an answer. Implemented as a **LangGraph** in [`chatbot/graph.py`](chatbot/graph.py).

| Document | Use when |
|----------|----------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, env matrix, extension points |
| **[docs/API.md](docs/API.md)** | HTTP API reference (`POST /query`, `POST /v1/chat`, schemas, errors) |
| **[docs/SCRIPTS.md](docs/SCRIPTS.md)** | Every CLI/module: flags, examples, workflows |
| [docs/BQ_NL2SQL_PLAN.md](docs/BQ_NL2SQL_PLAN.md) | Bronze NL-to-SQL design |
| [docs/EXPECTED_QUESTIONS.md](docs/EXPECTED_QUESTIONS.md) | Example question types |

## Graph shape (current)

```
START → decompose ─┬─ identity meta? ──→ generate_meta ──→ END
                   ├─ product query? ──→ generate_product ──→ END
                   └─ else → parallel_retrieve → bq_retrieve → merge → rerank ─┬─ weak context? → web_fallback → generate → END
                                                                              └─ else ───────────────────────────────→ generate → END
                                         │
                        ┌────────────────┼────────────────┐
                        ▼                ▼                ▼
                  bq_table_match    news vector    academic vector
```

- **decompose**: heuristics + optional LLM → geography, time range, entities, domains; routes identity meta (`who are you`), product KB (`what is the aim of OpenTrace`), or full RAG.
- **generate_meta** / **generate_product**: short-circuit paths with no retrieval; product answers use [`chatbot/data/opentrace_product.json`](chatbot/data/opentrace_product.json).
- **parallel_retrieve**: BQ table-description match + news + research Qdrant search (thread pool).
- **bq_retrieve**: NL-to-SQL (LM Studio or HF) from table hints → execute bronze SELECTs; up to `RAG_BQ_MAX_SQL_QUERIES` queries.
- **merge / rerank / web_fallback / generate**: fuse context; optional LLM rerank (`RAG_LLM_RERANK=off` recommended locally); when `RAG_WEB_FALLBACK_ENABLED=1` and internal context is weak, fetch Wikipedia (then Tavily if wiki empty) via [`retrievers/web_retriever.py`](retrievers/web_retriever.py); answer via [`llm_chat.py`](llm_chat.py).

Details: [ARCHITECTURE.md §4](ARCHITECTURE.md#4-runtime-pipeline-run_rag).

## Chat sessions and context memory (summary + verbatim window)

- **Retrieval** (decompose, BigQuery, vectors) always uses **only the latest user message**. Prior turns do not change retrieval.
- **Generation** sees a **rolling summary** of older dialogue plus the **last N user+assistant pairs** verbatim (default **N = 5**). When a new reply would exceed N pairs, the **oldest** pair is folded into the summary via an LLM call (`HF_API_TOKEN`; optional `RAG_SUMMARY_MODEL_ID`, else `RAG_LLM_MODEL_ID`). If the token is missing, folding uses a short **text stub** instead (see [`ml/rag/chatbot/chat_memory.py`](ml/rag/chatbot/chat_memory.py)).
- **Streamlit UI** keeps a **full message list** for scrolling; the compact **summary + recent_turns** is what gets sent to `run_rag` on the next turn.

**Env (optional)**

| Variable | Meaning |
|----------|---------|
| `RAG_CHAT_VERBATIM_TURNS` | Max verbatim **pairs** (overrides `RAG_CHAT_HISTORY_MAX_TURNS` if set) |
| `RAG_CHAT_HISTORY_MAX_TURNS` | Fallback max pairs (default **5**) |
| `RAG_CHAT_HISTORY_MAX_CHARS` | Soft cap on verbatim block size in the prompt (default **4000**) |
| `RAG_SUMMARY_MAX_CHARS` | Max length of the running summary string (default **2000**) |
| `RAG_SUMMARY_MODEL_ID` | Optional HF model for summarization |

**Redis / scaling (for durable sessions + cross-worker caches)**

| Variable | Meaning |
|----------|---------|
| `RAG_REDIS_URL` / `REDIS_URL` | Connection string for Redis (Memorystore, Upstash, sidecar, etc.). When present the session store and BQ/bronze caches become shared and durable. |
| `RAG_SESSION_TTL_SECONDS` | Expiry for conversation blobs (default **86400** = 24 h). 0 = no expiry. |
| `RAG_CACHE_TTL_SECONDS` | Default TTL for secondary caches (BQ schema, bronze catalog) (default **3600**). |
| `RAG_REDIS_CONNECT_TIMEOUT_S` | Socket timeout for initial connect/ping (default **2**). |

See also [`ml/rag/session_store.py`](ml/rag/session_store.py) (the facade) and `ARCHITECTURE.md §12.7`.

See also [`ml/rag/chat_history.py`](ml/rag/chat_history.py) (shim to [`chatbot/chat_history.py`](chatbot/chat_history.py)) for **legacy** `chat_history`-only truncation (no summary).

**Streamlit** ([`chatbot/streamlit_app.py`](ml/rag/chatbot/streamlit_app.py)): multiple **chat sessions** in the sidebar; **pipeline debug** shows the last run’s decomposition and retrieval stats.

**API** (`POST /query` and `POST /v1/chat`): responses include **`session_id`**, structured **`citations`**, and aggregated LLM **`usage`** (sum of all LLM calls in the request: decompose, BQ NL2SQL, memory fold, rerank if on, generation). Example shape:

```json
{
  "answer": "Prose with inline footnotes [14][18] only",
  "citations": [
    { "id": 14, "kind": "academic", "text": "[Academic] ...", "url": null },
    { "id": 18, "kind": "news", "text": "[News] ...", "url": "https://..." }
  ],
  "session_id": "...",
  "usage": {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

By default **`answer`** is prose-only (no trailing **Sources** markdown block); clients should render `citations`. Set **`RAG_APPEND_SOURCES_TO_ANSWER=1`** for legacy embedded Sources in `answer` (e.g. Streamlit during transition).

Reuse **`session_id`** for **server-side** `{conversation_summary, recent_turns}`. Backed by Redis when `RAG_REDIS_URL` is set (durable across workers/restarts, required for scaling). Without Redis the store is in-process only (single worker; lost on restart). Send **`chat_history`** to supply prior turns from the client (fully stateless path); history is compacted for that request only and the server store is **not** updated. Deprecated alias: **`conversation_history`**.

**Canonical `POST /query` request** (backend contract):

```json
{
  "query": "What are rice yield trends?",
  "session_id": "abc123...",
  "user_profile": {
    "country": "Ghana",
    "plan_type": "Farmers",
    "category": "Farmers"
  },
  "chat_history": [
    { "role": "user", "content": "Previous question" },
    { "role": "assistant", "content": "Previous answer" }
  ],
  "include_trace": false
}
```

**`user_profile`**: `plan_type` (access tier + retrieval gates) and `category` (generation persona) are required when the profile is sent; **`country`** is a **retrieval geo filter only** for **`plan_type: Farmers`**. Other plans use geography from query decomposition. Legacy `stakeholder_type`, `audience_instructions`, and top-level `geo_override` are rejected.

Meta / identity and product questions short-circuit retrieval (see `assistant_identity.py`, `product_knowledge.py`). Full RAG answers use numbered context sources (`[Source N]` in the LLM context) with Wikipedia-style inline footnotes (`[1]`, `[5]`) in prose. Referenced sources appear in **`citations`** by default (`RAG_CITATIONS_MODE=referenced`; set `all` for every packed source). BQ validation/execution failures are dropped before generation; model-written Sources appendices are stripped; table names and SQL are not shown to users.

**Redis config** (new in scaling release):
- `RAG_REDIS_URL` (or `REDIS_URL`): e.g. `redis://host:6379/0` or rediss:// for TLS.
- `RAG_SESSION_TTL_SECONDS` (default 86400), `RAG_CACHE_TTL_SECONDS` (default 3600 for BQ/bronze caches), `RAG_REDIS_CONNECT_TIMEOUT_S`.

See `session_store.py`, `ARCHITECTURE.md`, and the production deploy guide for details.

## Env and config

- **BigQuery**: `BQ_PROJECT` and `BQ_DATASET_BRONZE` (see `data/local/.env`). The BQ retriever loads schema and runs NL-to-SQL **only** against the bronze dataset. Silver/gold env vars remain for dbt and other tooling.
- **Bronze table hints (YAML + vectors)**: `match_bq_tables_from_descriptions` groups vector hits by `table_name` and fuses each with a compact column catalog from [`chatbot/bronze_dataset_catalog.py`](chatbot/bronze_dataset_catalog.py). Set **`RAG_BRONZE_MODEL_YAML`** to override the default path (`ml/rag/chatbot/bronze_dataset_model.yml`). **`RAG_BRONZE_MODEL_SOURCE`** selects the dbt `sources` entry by name (default **`bronze`**); set it empty to merge every source in that file. If the primary YAML is missing or parses to no tables, the loader falls back to **`dbt/models/sources.yml`** (still honoring `RAG_BRONZE_MODEL_SOURCE`). Live BigQuery introspection remains the source of truth for runnable SQL.
- **Vector DB**: **Qdrant** (set `QDRANT_URL`, `QDRANT_API_KEY`, and per-collection names such as `QDRANT_COLLECTION_NEWS`, `QDRANT_COLLECTION_RESEARCH_PAPERS`, `QDRANT_COLLECTION_DATA_DESCRIPTIONS`, `QDRANT_COLLECTION_OTA_INSIGHTS`). Populate via [`ingestion/cli`](ingestion/cli.py) rebuild or the `*_preprocessor` / `*_load_to_vector_db` scripts below. Debug payload counts/metadata: `PYTHONPATH=ml-eng python -m ml.rag.inspect_vector_db`.
- **Embeddings (Qdrant)** — per-corpus profiles in [`text_processors/chunking_config.py`](text_processors/chunking_config.py):

| Corpus | Collection | Model (default) | Dim | Qdrant mode |
|--------|------------|-----------------|-----|-------------|
| News | `news_data` | `intfloat/multilingual-e5-small` | 384 | `dense_named` (hybrid capable) |
| Research | `research_other_papers` | `intfloat/multilingual-e5-small` | 384 | `dense_named` (RAM-optimized, hybrid capable) |
| OTA | `OTA_insights` | `BAAI/bge-small-en-v1.5` | 384 | `ota_triple` |
| BQ descriptions | `BQ_table_descriptions` | `BAAI/bge-small-en-v1.5` | 384 | `sentence_named` |

| Variable | Meaning |
|----------|---------|
| `RAG_EMBEDDINGS_MODE` | `fastembed` (Railway, in-container ONNX) or `local` (dev with torch) |
| `RAG_EMBEDDING_MODEL_NEWS` / `_RESEARCH` / `_OTA` / `_DATA_DESCRIPTION` | Override per-corpus model ids |
| `RAG_CHUNK_TARGET_TOKENS_*` / `RAG_CHUNK_OVERLAP_PCT_*` | Override chunk sizes (see `chunking_config.py`) |
| `RAG_NEWS_GEO_FALLBACK` | Default **`1`**: retry news search without geo if geo filter returns nothing (disabled for multi-country **compare**) |
| `RAG_NEWS_TIME_FALLBACK` | Default **`1`**: retry news without date filter if still empty |
| `RAG_NEWS_TIME_QDRANT_FILTER` | Default **off**: apply news dates in Python (keeps articles missing `published_at` payload) |
| `RAG_NEWS_COMPARE_SEMANTIC_FALLBACK` | Default **on**: for compare + 2 countries, semantic search then post-filter by country name in text |
| `RAG_RESEARCH_GEO_FALLBACK` | Default **`1`**: same for research corpus |
| `RAG_RESEARCH_TIME_FALLBACK` | Default **`1`**: retry research without year/date filter if still empty |

**E5 prefixing:** news and research use `query:` at retrieval and `passage:` at index time (automatic in `vector_retriever`).

**Reindex:** after changing chunking or embedding models, run loaders with `--reset` or recreate collections via `python -m ml.rag.scripts.create_qdrant_collections`, then repopulate. Preprocessors skip unchanged chunks via `content_hash` in the ingest manifest (`INGEST_VERSION` bump forces re-chunk).

**Preprocess pipeline:** [`text_processors/preprocess/`](text_processors/preprocess/) — parse → section/schema blocks → corpus-specific chunking (~500 tokens) → hard token cap. Chunk metadata includes `hierarchy_path`, `parent_chunk_id`, `semantic_lane` (research/OTA).

| Qdrant collection | Chunking strategy |
|-------------------|-------------------|
| `news_data` | Recursive paragraphs + semantic fallback (`recursive_semantic`) |
| `research_other_papers` | Section blocks + semantic boundaries (`hierarchical_semantic`) |
| `OTA_insights` | Semantic within each lane (`lane_semantic`) |
| `BQ_table_descriptions` | Schema/table blocks, sentence cap only (`schema_only`) |

Semantic splits use the same E5 model as ingest (`profile.embedding_model`). Disable with `RAG_SEMANTIC_CHUNKING=0`. Tune breakpoints with `RAG_SEMANTIC_BREAKPOINT_PERCENTILE` (default `95`).

**Troubleshooting preprocess:** if you see `NumPy 2.x` / `PyTorch was not found` / `torch>=2.4` errors:

```bash
cd ml-eng && source venv/bin/activate
pip install 'numpy>=1.24,<2'
pip install 'transformers>=4.44,<5' 'sentence-transformers>=3.0,<5'
```

On **Intel Mac** (`x86_64`), PyPI often only offers **torch up to 2.2.2** — you cannot `pip install torch>=2.4`. Use the `transformers<5` pins above (works with torch 2.2.2). Preprocess falls back to sentence/token chunking if embeddings still cannot load.

**Eval:** `PYTHONPATH=ml-eng python -m ml.rag.eval.run_retrieval_eval --corpus all --k 5` (requires live Qdrant + populated collections).

- **Chat memory**: variables in the table above; summarization needs **`HF_API_TOKEN`** (same as the answer generator).

## Run

From repo root (recommended: install from `ml-eng/`):

```bash
# Install deps (ml-eng)
pip install -r ml-eng/requirements.txt -r ml-eng/requirements-dev.txt

# Create Qdrant collections (set QDRANT_URL + QDRANT_API_KEY in ml-eng/data/local/.env first)
cd ml-eng && set -a && source data/local/.env && set +a
PYTHONPATH=. python -m ml.rag.scripts.create_qdrant_collections

# Rebuild collections from Google Drive (OAuth user auth; run from ml-eng/)
# Required: QDRANT_*, GDRIVE_OAUTH_CLIENT_SECRET_JSON, GDRIVE_FOLDER_* (ID or folder URL)
# Research merges GDRIVE_FOLDER_RESEARCH_PAPERS_ID (academic_article) and
#   GDRIVE_FOLDER_OTHER_PAPERS_ID (policy_report) into research_other_papers.
PYTHONPATH=. python -m ml.rag.ingestion.cli rebuild --kind all --reset

# Preprocess only (structure-aware, token-bounded chunks → data/local/preprocessed_data/*.jsonl)
# Requires: pip install -r ml-eng/ml/rag/requirements.txt (pypdf, tiktoken, llama-index-core)
# Optional structure parsing: pip install -r ml-eng/ml/rag/requirements-preprocess-optional.txt && export RAG_USE_UNSTRUCTURED=1
cd ml-eng && PYTHONPATH=.

# Unified CLI
python -m ml.rag.text_processors.preprocess.cli run --corpus research \
  --input-dir ml/rag/data/Text_Documents
python -m ml.rag.text_processors.preprocess.cli validate \
  --jsonl data/local/preprocessed_data/research_chunks.jsonl

# Per-corpus wrappers (same engines)
python -m ml.rag.text_processors.research_papers_preprocessor \
  --input-dir ml/rag/data/Text_Documents
python -m ml.rag.text_processors.news_collection_preprocessor --input-dir data/local/web_news_rss
python -m ml.rag.text_processors.data_descriptions_preprocessor --input-dir /path/to/docx
python -m ml.rag.text_processors.ota_insights_preprocessor  # via consolidate_ota_staging import

# Load into Qdrant (separate step)
python -m ml.rag.text_processors.research_papers_load_to_vector_db --reset
python -m ml.rag.text_processors.news_load_to_vector_db --reset
python -m ml.rag.text_processors.data_descriptions_load_to_vector_db --reset

# Run with a question
PYTHONPATH=ml-eng python -m ml.rag.run "What tables exist in bronze for yields?"
PYTHONPATH=ml-eng python -m ml.rag.run
```

### Test with Streamlit (pipeline inspector)

```bash
PYTHONPATH=ml-eng streamlit run ml/rag/chatbot/streamlit_app.py
```

Open http://localhost:8501. The **pipeline inspector** panel (enabled by default) shows after each turn:

- **Route:** meta / product / full_rag / web_fallback / insufficient
- **Decomposition** (geography, time, entities)
- **Retrieval metrics** (news, research, OTA, BQ, web, latency, token usage)
- **Tabs** for every retrieval arm plus merged context and generator input
- **Preset queries** in the sidebar (identity, product, RAG, Farmers+Ghana)

**Backend modes:**

| Mode | Use |
|------|-----|
| **In-process** (default) | Full chunk detail via `run_rag()` |
| **HTTP API** | `POST /query` with `include_trace` — counts only; set `RAG_API_BASE_URL` |

Env: `RAG_STREAMLIT_DEBUG_DEFAULT=1`, `RAG_SHOW_SQL_DEBUG=1` (optional SQL panel).

### Deploy Streamlit on Railway (second service, internal QA)

Full deploy guide: [deploy/DEPLOY_STREAMLIT_RAILWAY.md](../../deploy/DEPLOY_STREAMLIT_RAILWAY.md). Smoke test: `ml-eng/scripts/smoke_streamlit_railway.sh <url>`.

Keep the existing **RAG API** service (`Dockerfile.railway`). Add a **second service** for the QA UI:

1. New Railway service, root directory **`ml-eng/`**
2. Config file: [`railway.streamlit.toml`](../railway.streamlit.toml) (or set `dockerfilePath` to `Dockerfile.railway.streamlit`)
3. Copy env vars from the API service: `QDRANT_*`, `RAG_LLM_*`, `BQ_*`, `GOOGLE_APPLICATION_CREDENTIALS_BASE64`, optional `RAG_REDIS_URL`, `LANGFUSE_*`
4. Recommended QA vars: `RAG_SHOW_SQL_DEBUG=1`, `RAG_STREAMLIT_DEBUG_DEFAULT=1`
5. Health check: `GET /_stcore/health`
6. Restrict access (team-only URL or private networking)

```bash
docker build -f ml-eng/Dockerfile.railway.streamlit -t opentrace-rag-streamlit:latest ml-eng/
```

**Verification checklist:**

| Query | Expected route |
|-------|----------------|
| `Who are you?` | meta — zero retrieval |
| `What is OpenTrace?` | product — zero retrieval |
| `Maize yields in Kenya 2020` | full_rag — populated tabs |
| Farmers + Ghana preset | full_rag — geo in decomposition |

Programmatic:

```python
from ml.rag.graph import run_rag

result = run_rag("Your question")
print(result["answer"])
# Multi-turn (generator): prefer summary + recent verbatim pairs
result2 = run_rag(
    "Follow-up using that context",
    conversation_summary="",
    recent_turns=[
        {"role": "user", "content": "Your question"},
        {"role": "assistant", "content": result["answer"]},
    ],
)
# Legacy: flat chat_history (verbatim only, truncated; no LLM summary fold)
# result3 = run_rag("Follow-up", chat_history=[{"role": "user", "content": "..."}, ...])
# result also has: bq_results, vector_results, merged_context, reranked_context, error
```

## Modules

| Path | Role |
|------|------|
| [`chatbot/state.py`](chatbot/state.py) | `RAGState` (legacy); graph uses `RAGGraphState` in [`chatbot/graph.py`](chatbot/graph.py) |
| [`retrievers/base.py`](retrievers/base.py) | `BaseRetriever` interface |
| [`retrievers/bq_retriever.py`](retrievers/bq_retriever.py) | BigQuery retrieval |
| [`retrievers/vector_retriever.py`](retrievers/vector_retriever.py) | Qdrant vector retrieval |
| [`chatbot/reranker.py`](chatbot/reranker.py) | `rerank(query, context_items, top_k)` |
| [`chatbot/generator.py`](chatbot/generator.py) | `generate(query, context_items)` |
| [`chatbot/graph.py`](chatbot/graph.py) (re-export [`graph.py`](graph.py)) | LangGraph build + `run_rag(query)` |
| [`chat_history.py`](chat_history.py) | Shim: `ml.rag.chat_history` → [`chatbot/chat_history.py`](chatbot/chat_history.py) |
| [`chat_memory.py`](chat_memory.py) | Shim: `ml.rag.chat_memory` → [`chatbot/chat_memory.py`](chatbot/chat_memory.py) |
| [`app/api.py`](app/api.py) | FastAPI app (`ml.rag.app.api`); use [`api.py`](api.py) for `uvicorn ml.rag.api:app` |
| [`run.py`](run.py) | CLI entrypoint |

## Extending

1. **Vector DB**: RAG uses **Qdrant** only ([`retrievers/vector_retriever.py`](retrievers/vector_retriever.py)). Configure `QDRANT_URL`, `QDRANT_API_KEY`, and collection env vars; extend `VectorRetriever` if you need a different backend.
2. **Reranker**: In [`chatbot/reranker.py`](chatbot/reranker.py), call your API or model and return ordered `list[dict]` with `content`/`text`.
3. **Generator**: In [`chatbot/generator.py`](chatbot/generator.py), call Vertex AI / OpenAI / local LLM with `query` and `context_items` and return the answer string.
4. **BQ NL-to-SQL**: In `BQRetriever.retrieve()`, add a step that turns `query` into SQL (e.g. LLM or templates) and pass it as `kwargs["sql"]` or set `sql` internally.

---

## Deploy on Hugging Face and expose API to the frontend

The RAG can be deployed as a **Hugging Face Space (Docker)** and its API called from your frontend chatbot.

### API

| Method | Path     | Description                    |
|--------|----------|--------------------------------|
| GET    | `/health`| Readiness check                |
| POST   | `/query` | Run RAG; body `{"query": "..."}`; optional `session_id` for chat memory |
| GET    | `/docs`  | Swagger UI                     |

**Example (frontend):**

```bash
curl -X POST "https://YOUR-SPACE-URL.hf.space/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What bronze data can we use for crop yields?"}'
# Next turn: reuse session_id from the response body
# -d '{"query": "Focus on Kenya", "session_id": "<id from previous response>"}'
```

Response:

```json
{
  "answer": "...",
  "session_id": "abc123...",
  "error": null
}
```

### Observability with Langfuse (optional)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_HOST` to emit traces for every `/query` request, LLM generation, retrieval step, and LangGraph node execution. Traces appear in the Langfuse UI (local or Cloud) with full token usage, latency, and metadata (session_id, plan_type, category, etc.).

- When keys are absent the integration is a silent no-op (safe for HF Spaces that do not need tracing).
- The same env vars work for local `docker compose`, GCE, and HF (point HOST at your Langfuse instance or cloud).

### Deploy to Hugging Face Spaces

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces): choose **Docker**, and either push this repo or a copy that includes `ml/rag` and the Dockerfile.

2. **Dockerfile**  
   Hugging Face expects a **Dockerfile at the repo root**. This repo has **`Dockerfile.rag`** at the root: in your Space, either **rename it to `Dockerfile`** (or copy its contents into `Dockerfile`) so the Space builds the RAG API. Build context is the repo root.

3. **Secrets (Space → Settings → Variables and Secrets)**  
   - `BQ_PROJECT` – GCP project ID  
   - `BQ_DATASET_BRONZE` – BigQuery bronze dataset (RAG queries this dataset only)  
   - **Qdrant**: `QDRANT_URL`, `QDRANT_API_KEY`, and collection variables as in [Env and config](#env-and-config)  
   - For BigQuery auth: either attach a **GCP service account key** (e.g. paste JSON as a secret and set `GOOGLE_APPLICATION_CREDENTIALS` to a path you write it to at startup) or use Workload Identity if running on GCP.  
   - `HF_API_TOKEN` (and optional embedding / LLM model ids) as needed.
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (optional; send traces to Langfuse Cloud or self-hosted instance)

4. **CORS**  
   For production, set **`RAG_CORS_ORIGINS`** to your frontend origin(s), comma-separated (e.g. `https://yourapp.com`). Default is `*`.

5. **Port**  
   The app listens on **7860** (required by Hugging Face Spaces).

### Deploy to Railway

1. **Service root directory:** `ml-eng/` (uses `Dockerfile.railway` + `railway.toml`).
2. **Redeploy with build cache cleared** after dependency changes (e.g. `fastembed`).
3. **Required variables** (Railway dashboard):

| Variable | Value |
|----------|--------|
| `RAG_LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `RAG_LLM_API_KEY` | Your OpenRouter API key |
| `RAG_LLM_MODEL_ID` | `openrouter/owl-alpha` |
| `RAG_EMBEDDINGS_MODE` | `fastembed` (recommended on Railway) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud |
| `BQ_PROJECT` / `BQ_DATASET_BRONZE` | BigQuery bronze |
| `GOOGLE_APPLICATION_CREDENTIALS_BASE64` | Base64 of GCP service account JSON |

4. **Recommended:** `RAG_LLM_RERANK=off`, `RAG_LLM_TIMEOUT_S=300`, `RAG_EMBEDDINGS_MODE=fastembed` (or `local` — auto-falls back to fastembed without torch).
5. **Do not set** `RAG_LLM_BASE_URL` to a LAN IP. Remove stale `GOOGLE_APPLICATION_CREDENTIALS=config/keys/...`.
6. **Optional:** `OPENROUTER_HTTP_REFERER=https://opentrace.africa`, `OPENROUTER_APP_TITLE=Ask ADZA`.
7. **Smoke test:** `GET /health`, `GET /ready`, `POST /query` with `{"query":"Who are you?"}`.

### Local API (same interface as HF)

```bash
pip install fastapi "uvicorn[standard]"
PYTHONPATH=ml-eng uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860
# Frontend: http://localhost:7860/docs and POST http://localhost:7860/query
```

### Docker Compose (local)

Prerequisites:

- **`data/local/.env` must exist** (Compose `env_file`); create it with your secrets — e.g. `BQ_PROJECT`, `BQ_DATASET_BRONZE`, `HF_API_TOKEN`, **`QDRANT_URL`**, **`QDRANT_API_KEY`**, and optional `QDRANT_COLLECTION_*` / `RAG_*` vars. If the file is missing, `docker compose` fails when starting RAG services.
- **GCP key** mounted read-only into the container (default host path **`data/local/keys/opentrace-bq-key.json`**). Override with env **`GCP_SA_KEY_HOST_PATH`** before `docker compose` if your key lives elsewhere.
- **Qdrant**: vectors live in Qdrant (cloud or self-hosted), not in a bind-mounted `vector_db` directory. Populate collections using [Run](#run) (ingestion CLI or loaders) before expecting non-empty retrieval.
- **Optional port overrides (shell env, not inside `.env` required):** `RAG_API_PORT` (default 7860), `RAG_STREAMLIT_PORT` (default 8501), `GCP_SA_KEY_HOST_PATH`.

From repo root:

```bash
# API only (port 7860)
docker compose --profile rag up --build rag-api

# API + Streamlit (7860 and 8501)
docker compose --profile rag up --build rag-api rag-streamlit
```

- **API docs:** http://localhost:7860/docs  
- **Streamlit:** http://localhost:8501  

Stop: `docker compose --profile rag down` (or `down` without profile if no other services use those containers).

### Docker Compose — “baked” images (no local vector index mount)

For Qdrant-backed RAG, **indexes are not copied from `data/local/vector_db`** (that path was legacy). Use **Qdrant Cloud** (or a reachable Qdrant URL) and pass **`QDRANT_*`** secrets at runtime. Optional baked images can still pre-install dependencies and app code; see your repo’s `Dockerfile.rag-baked` / root `Dockerfile` for how the **serving** image is built.

- **`Dockerfile.rag-baked`** (if present): may bundle app + Streamlit with [`scripts/hf-entrypoint.sh`](../../scripts/hf-entrypoint.sh) for optional **`GCP_SA_JSON`** / **`GCP_SA_JSON_B64`**.
- **Root `Dockerfile`:** may target the public chat API (`ml.serving.chat.app`); vector state still comes from Qdrant when RAG is wired in.

**Secrets:** do not put **`HF_API_TOKEN`** or GCP JSON **into** the Dockerfile. Pass them at runtime via **`data/local/.env`** or `-e` (Compose already uses `env_file`). For GCP, either mount a key file (default path in Compose) or set **`GCP_SA_JSON`** so the entrypoint writes `/tmp/gcp-sa.json`.

```bash
# Example: build and run baked profile services (names depend on your compose file)
docker compose --profile baked build rag-api-baked rag-streamlit-baked
docker compose --profile baked up rag-api-baked rag-streamlit-baked

# Optional: public chat API (port may differ)
docker compose --profile baked up chat-api-baked
```

### Docker (local or CI)

From the **`ml-eng/`** directory (so `requirements.txt` and `ml/` exist in the build context):

```bash
docker build -f ml/rag/Dockerfile -t rag-api .

docker run --rm -p 7860:7860 \
  -e BQ_PROJECT=your-project \
  -e BQ_DATASET_BRONZE=bronze \
  -e HF_API_TOKEN=your-hf-token \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/bq.json \
  -e QDRANT_URL=https://your-cluster.qdrant.io \
  -e QDRANT_API_KEY=your-qdrant-api-key \
  -v "$(pwd)/path/to/your-sa.json:/secrets/bq.json:ro" \
  rag-api
```

Adjust paths and secrets to match your machine. Without valid **Qdrant** and **BigQuery** credentials, retrieval or BQ steps may return empty context.
