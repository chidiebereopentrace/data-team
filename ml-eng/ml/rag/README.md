# RAG pipeline (graph / agentic)

Modular RAG with **two co-equal retrieval legs** — **Qdrant** (six vector corpora) and **BigQuery** (staging YAML + NL2SQL) — that fuse at `merge`, then rerank → **generation strategy** → LLM answer. Implemented as a **LangGraph** in [`chatbot/graph.py`](chatbot/graph.py).

| Document | Use when |
|----------|----------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, env matrix, extension points |
| **[docs/API.md](docs/API.md)** | HTTP API reference (`POST /query`, `POST /v1/chat`, schemas, errors) |
| **[docs/SCRIPTS.md](docs/SCRIPTS.md)** | Every CLI/module: flags, examples, workflows |
| [docs/BQ_NL2SQL_PLAN.md](docs/BQ_NL2SQL_PLAN.md) | Bronze NL-to-SQL design |
| [docs/EXPECTED_QUESTIONS.md](docs/EXPECTED_QUESTIONS.md) | Example question types |

## Graph shape (current)

```
START → decompose ─┬─ short-circuit routes (meta / product / social / clarify) ──→ END
                   └─ full_rag:
                        parallel_retrieve  ← VECTOR LEG (6 Qdrant corpora, thread pool)
                             ↓
                        bq_reason → bq_retrieve  ← BQ LEG (YAML reasoner + NL2SQL)
                             ↓
                        merge  ← first fusion of vector_*_results + bq_results
                             ↓
                        rerank + diversify
                             ↓
                        web_fallback? (optional)
                             ↓
                        node_generate:
                          build_generation_plan → generate (+ ACF, citations)
                             ↓
                        export → END
```

- **decompose**: enricher + heuristics + optional LLM → geography, time, entities, domains; [`retrieval_contract`](chatbot/retrieval_contract.py) + `task_mode`; routes meta/product/clarify or full RAG.
- **parallel_retrieve (VECTOR LEG)**: [`select_corpora`](chatbot/corpus_catalog.py) gates/boosts six collections; each runs [`VectorRetriever`](retrievers/vector_retriever.py) (E5 embed, geo/time filters, hybrid RRF, cascade fallbacks) → `vector_news_results`, `vector_academic_papers_results`, `vector_policies_results`, `vector_public_reports_results`, `vector_formation_results`, `vector_ota_results`.
- **bq_reason / bq_retrieve (BQ LEG)**: staging YAML reasoner selects tables → NL2SQL → `bq_results`. Independent of vector hits; graph node order is sequential but legs are peers until `merge`.
- **merge / rerank**: fuse BQ rows + all vector chunks; cross-encoder rerank (pool 24, top ~18) + diversity pack; optional web fallback (Wikipedia/Tavily).
- **build_generation_plan** ([`generation_plan.py`](chatbot/generation_plan.py)): post-retrieval strategy — answer shape, evidence priority, grounding rules — injected into the prompt before the LLM call.
- **generate**: context packing, [`llm_chat.py`](llm_chat.py), ACF scoring, structured citations.

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
  "answer": "Prose without inline footnotes by default; clients render citations[]",
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

Meta / identity and product questions short-circuit retrieval (see `assistant_identity.py`, `product_knowledge.py`). Full RAG answers use numbered context sources (`[Source N]` in the LLM context). **Default chat** does **not** put Wikipedia-style `[N]` footnotes in prose — clients should render the structured **`citations`** array (packed sources for that turn). Inline `[N]` footnotes are enabled for analytical write-ups, DOCX/PDF/multi exports, or when the user explicitly asks for footnotes/inline citations. When inline footnotes are on, `RAG_CITATIONS_MODE=referenced` (default) filters `citations` to those markers; set `all` for every packed source. BQ validation/execution failures are dropped before generation; model-written Sources appendices are stripped; table names and SQL are not shown to users.

**Redis config** (new in scaling release):
- `RAG_REDIS_URL` (or `REDIS_URL`): e.g. `redis://host:6379/0` or rediss:// for TLS.
- `RAG_SESSION_TTL_SECONDS` (default 86400), `RAG_CACHE_TTL_SECONDS` (default 3600 for BQ/bronze caches), `RAG_REDIS_CONNECT_TIMEOUT_S`.

See `session_store.py`, `ARCHITECTURE.md`, and the production deploy guide for details.

## Env and config

- **BigQuery**: `BQ_PROJECT` and **`BQ_DATASET_SILVER=staging_dev`** (see `config/.env.example`). The BQ retriever runs NL-to-SQL against **staging_dev** using per-table YAML in [`bq_tables_yaml_files/`](bq_tables_yaml_files/). `BQ_DATASET_BRONZE` is for data-eng tooling only.
- **BQ table selection (staging YAML reasoner)**: `node_bq_reason` / [`chatbot/bq_sql_reasoner.py`](chatbot/bq_sql_reasoner.py) picks staging tables from per-table YAML under [`bq_tables_yaml_files/`](bq_tables_yaml_files/) (via [`chatbot/bq_table_schema_yaml.py`](chatbot/bq_table_schema_yaml.py)) and writes `bq_table_candidates` for NL-to-SQL. Live BigQuery introspection remains the source of truth for runnable SQL.
- **Vector DB**: **Qdrant** — set `QDRANT_URL`, `QDRANT_API_KEY`, and **six** collection names: `QDRANT_COLLECTION_NEWS`, `QDRANT_COLLECTION_ACADEMIC_PAPERS`, `QDRANT_COLLECTION_POLICIES`, `QDRANT_COLLECTION_PUBLIC_REPORTS`, `QDRANT_COLLECTION_FORMATION`, `QDRANT_COLLECTION_OTA_INSIGHTS`. See [deploy/PRODUCTION_ENV.md](../../deploy/PRODUCTION_ENV.md). Populate via [`ingestion/cli`](ingestion/cli.py) rebuild or the `*_preprocessor` / `*_load_to_vector_db` scripts below. Debug payload counts/metadata: `PYTHONPATH=ml-eng python -m ml.rag.inspect_vector_db`.
- **Embeddings (Qdrant)** — per-corpus profiles in [`text_processors/chunking_config.py`](text_processors/chunking_config.py):

| Corpus | Collection | Model (default) | Dim | Qdrant mode |
|--------|------------|-----------------|-----|-------------|
| News | `news_data` | `intfloat/multilingual-e5-small` | 384 | `dense_named` (hybrid capable) |
| Research | `research_other_papers` | `intfloat/multilingual-e5-small` | 384 | `dense_named` (RAM-optimized, hybrid capable) |
| OTA | `OTA_insights` | `BAAI/bge-small-en-v1.5` | 384 | `ota_triple` |

| Variable | Meaning |
|----------|---------|
| `RAG_EMBEDDINGS_MODE` | `fastembed` (Railway, in-container ONNX) or `local` (dev with torch) |
| `RAG_EMBEDDING_MODEL_NEWS` / `_RESEARCH` / `_OTA` | Override per-corpus model ids |
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
python -m ml.rag.text_processors.ota_insights_preprocessor  # via consolidate_ota_staging import

# Load into Qdrant (separate step)
python -m ml.rag.text_processors.research_papers_load_to_vector_db --reset
python -m ml.rag.text_processors.news_load_to_vector_db --reset

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

### Observability with Langfuse (optional, SDK v3+)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL=https://cloud.langfuse.com` (EU Cloud) to emit **unified traces** for every RAG request.

**Product analytics (push → Railway redeploy):** with keys already on the Railway service, richer nested spans and soft-fail scores appear after redeploy — no Dockerfile or service change. Optional `user_id` on `/query` and `/v1/chat` maps to Langfuse Users. `LANGFUSE_TRACING_RELEASE` falls back to Railway’s `RAILWAY_GIT_COMMIT_SHA` when unset.

| Layer | What appears in Langfuse |
|-------|--------------------------|
| Root span | `rag.query` (API), `rag.chat_turn` (chat), `rag.streamlit` (QA UI) |
| LangGraph | Node spans via LangChain callback (decompose, retrieve, rerank, generate, …) |
| Control | `decompose`, `merge`, `web_fallback`, `insufficient_context` (+ route/corpus metadata) |
| Evidence | `citations` (post-generate attach; Path B ACF scored on cited sources, `acf_status=scored`) |
| Retrieval | `retrieval.qdrant`, `retrieval.bq_tables`, `retrieval.bq`, `retrieval.bq.nl2sql`, `retrieval.web` |
| Rerank / embed | `rerank` (mode/model/top score), `embedding.query` (dense + sparse) |
| LLM | `llm_chat_complete` generations with `purpose` (`decompose`, `bq.nl2sql`, `generate`, `generate_meta`, `generate_product`) |

Root metadata includes corpus counts, `empty_retrieval`, BQ soft-fail flags, `web_fallback_status`, and boolean scores when those flags are true. Tags: `session_id`, optional `user_id`, `plan_type`, `category`, `env:*`, `release:*`, `route:*`.

- When keys are absent the integration is a silent no-op (safe for HF Spaces / Railway without tracing).
- Optional: `LANGFUSE_TRACING_ENVIRONMENT=production|staging|development`
- Optional: `LANGFUSE_TRACING_RELEASE=<git-sha>` — else auto from `RAILWAY_GIT_COMMIT_SHA`
- Optional: `LANGFUSE_TRACING_SAMPLE_RATE=1.0` — reduce volume in production
- Legacy alias: `LANGFUSE_HOST` (mapped to `LANGFUSE_BASE_URL` at startup)

**Verify setup:** `PYTHONPATH=. python scripts/verify_langfuse_tracing.py` (from `ml-eng/`)

**User feedback:** `POST /feedback` with `{ "trace_id": "...", "score": 1.0, "comment": "..." }` (score 0–1). Serving chat returns `langfuse_trace_id` for the same.

**Suggested dashboards (Langfuse UI):** latency p95 by `route:*` tag, token cost by `plan_type:*`, error rate on `full_rag` vs `meta`, filter `empty_retrieval` / `bq_failure` scores.

**OpenRouter Sessions (LLM cost bundling):** When `RAG_LLM_BASE_URL` points at OpenRouter, each RAG run sends `session_id` (= Langfuse trace ID) on every `llm_chat_complete` **and** OpenRouter `/rerank` call so decompose + NL2SQL + generate + rerank share one session in [OpenRouter Logs](https://openrouter.ai/logs?tab=sessions). Disable with `RAG_OPENROUTER_SESSION_ID=off`. Optional `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_TITLE` on chat and rerank.

### Deploy to Hugging Face Spaces

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces): choose **Docker**, and either push this repo or a copy that includes `ml/rag` and the Dockerfile.

2. **Dockerfile**  
   Hugging Face expects a **Dockerfile at the repo root**. This repo has **`Dockerfile.rag`** at the root: in your Space, either **rename it to `Dockerfile`** (or copy its contents into `Dockerfile`) so the Space builds the RAG API. Build context is the repo root.

3. **Secrets (Space → Settings → Variables and Secrets)**  
   - `BQ_PROJECT` – GCP project ID  
   - **`BQ_DATASET_SILVER=staging_dev`** – dataset RAG NL-to-SQL queries (staging YAML tables)  
   - **Qdrant**: `QDRANT_URL`, `QDRANT_API_KEY`, and six `QDRANT_COLLECTION_*` vars (see [deploy/PRODUCTION_ENV.md](../../deploy/PRODUCTION_ENV.md))  
   - For BigQuery auth: either attach a **GCP service account key** (e.g. paste JSON as a secret and set `GOOGLE_APPLICATION_CREDENTIALS` to a path you write it to at startup) or use Workload Identity if running on GCP.  
   - `HF_API_TOKEN` (and optional embedding / LLM model ids) as needed.
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL=https://cloud.langfuse.com` (optional tracing)

4. **CORS**  
   For production, set **`RAG_CORS_ORIGINS`** to your frontend origin(s), comma-separated (e.g. `https://yourapp.com`). Default is `*`.

5. **Port**  
   The app listens on **7860** (required by Hugging Face Spaces).

### Deploy to Railway

1. **Service root directory:** `ml-eng/` (uses `Dockerfile.railway` + `railway.toml`).
2. **Redeploy with build cache cleared** after dependency or YAML/graph changes.
3. **Required variables** — full checklist: [deploy/PRODUCTION_ENV.md](../../deploy/PRODUCTION_ENV.md). Minimum:

| Variable | Value |
|----------|--------|
| `RAG_LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `RAG_LLM_API_KEY` | Your OpenRouter API key |
| `RAG_LLM_MODEL_ID` | `qwen/qwen3-30b-a3b-instruct-2507` |
| `RAG_SUMMARY_MODEL_ID` | `qwen/qwen3-30b-a3b-instruct-2507` |
| `RAG_EMBEDDINGS_MODE` | `fastembed` (baked in image) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud |
| Six `QDRANT_COLLECTION_*` | news, academic, policies, public_reports, formation, OTA |
| `BQ_PROJECT` / `BQ_DATASET_SILVER` | e.g. `opentrace-prod-5ga4` / `staging_dev` |
| `GOOGLE_APPLICATION_CREDENTIALS_BASE64` | Base64 of GCP service account JSON |
| `RAG_RERANKER_MODE=cross_encoder` / `RAG_RERANKER_MODEL=BAAI/bge-reranker-base` | Local rerank via fastembed ONNX (baked in image; set explicitly when using OpenRouter LLM) |

4. **Recommended:** `RAG_LLM_TIMEOUT_S=300`, `RAG_BQ_SKIP_LIVE_SCHEMA=on`, BQ hint byte budgets from [`.env.example`](../config/.env.example).
5. **Do not set** `RAG_LLM_BASE_URL` to a LAN IP, `QDRANT_COLLECTION_DATA_DESCRIPTIONS`, or `GOOGLE_APPLICATION_CREDENTIALS=config/keys/...`.
6. **Optional:** `OPENROUTER_HTTP_REFERER=https://opentrace.africa`, `OPENROUTER_APP_TITLE=Ask ADZA`.
7. **Observability (optional):**

| Variable | Value |
|----------|--------|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse project keys |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` |
| `LANGFUSE_TRACING_ENVIRONMENT` | `production` |
| `LANGFUSE_TRACING_RELEASE` | git SHA or version tag |
| `RAG_OPENROUTER_SESSION_ID` | leave unset/`on` (default) to bundle LLM calls per RAG run; `off` to disable |

When both Langfuse and OpenRouter are enabled, OpenRouter Sessions and Langfuse traces share the same run id (`session_id` = Langfuse trace id). See [Observability with Langfuse](#observability-with-langfuse-optional-sdk-v3) above and [deploy/README.md](../../deploy/README.md) §7.5.

8. **Smoke test:** `GET /health`, `GET /ready`, `POST /query` with `{"query":"Who are you?"}`. After a full RAG turn, confirm a nested trace in Langfuse and (if using OpenRouter) one session in OpenRouter Logs.

### Local API (same interface as HF)

```bash
pip install fastapi "uvicorn[standard]"
PYTHONPATH=ml-eng uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860
# Frontend: http://localhost:7860/docs and POST http://localhost:7860/query
```

### Docker Compose (local)

Prerequisites:

- **`data/local/.env` must exist** (Compose `env_file`); create it with your secrets — e.g. `BQ_PROJECT`, **`BQ_DATASET_SILVER`**, `HF_API_TOKEN`, **`QDRANT_URL`**, **`QDRANT_API_KEY`**, six `QDRANT_COLLECTION_*`, and optional `RAG_*` vars. If the file is missing, `docker compose` fails when starting RAG services.
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
