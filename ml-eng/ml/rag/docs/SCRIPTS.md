# RAG scripts reference (`ml/rag`)

Command reference for **every runnable entry point** under `ml-eng/ml/rag/`. All examples assume:

```bash
cd ml-eng
set PYTHONPATH=.
# Windows CMD; use export PYTHONPATH=. on Unix
```

Environment is loaded from `ml-eng/config/.env` and `ml-eng/data/local/.env` via [`local_env.py`](../local_env.py) where noted.

**See also:** [ARCHITECTURE.md](../ARCHITECTURE.md) (design), [README.md](../README.md) (setup).

---

## Quick index

| Category | Module |
|----------|--------|
| **Query / UI** | `ml.rag.run`, `ml.rag.chatbot.streamlit_app`, `ml.rag.api` |
| **Qdrant admin** | `ml.rag.scripts.create_qdrant_collections` |
| **Ingest (Drive)** | `ml.rag.ingestion.cli` |
| **Preprocess** | `ml.rag.text_processors.preprocess.cli` |
| **Load to Qdrant** | `news/research/data_descriptions/ota_insights_load_to_vector_db`, `load_pdf_chunks_to_vector_db` |
| **Corpus preprocess (legacy wrappers)** | `*_preprocessor.py`, `news_collection_preprocessor.py` |
| **Eval** | `ml.rag.eval.run_retrieval_eval` |
| **Diagnostics** | `ml.rag.inspect_vector_db`, `ml.rag.check_hf` |
| **Helpers** | `ml.rag.helpers.generate_table_yamls` |

---

## 1. Query and UI

### `python -m ml.rag.run`

**File:** [`run.py`](../run.py)

Run the full LangGraph pipeline from the shell.

| Invocation | Behavior |
|------------|----------|
| `python -m ml.rag.run "Your question"` | Single question; answer on **stdout** |
| `python -m ml.rag.run` | Read one line from stdin |

**Stderr (default on):**

- Generated BigQuery SQL (`RAG_CLI_SHOW_SQL=1`)
- Retrieval counts (`RAG_CLI_SHOW_RETRIEVAL=1`)
- Optional pipeline trace (`RAG_CLI_PIPELINE_TRACE=1`)

**Requires:** Qdrant, BQ credentials, LLM (`RAG_LLM_BASE_URL` or `HF_API_TOKEN`) for best results.

```bash
python -m ml.rag.run "What were maize yield trends in Nigeria from 2013 to 2022?"
```

---

### `streamlit run ml/rag/chatbot/streamlit_app.py`

**File:** [`chatbot/streamlit_app.py`](../chatbot/streamlit_app.py)

Interactive chat with sidebar sessions, retrieval sliders, LLM backend fields, and **pipeline debug** (decomposition, BQ SQL list, chunk counts).

```bash
streamlit run ml/rag/chatbot/streamlit_app.py
```

**Typical local settings:** LM Studio URL in sidebar or `.env`, `RAG_LLM_RERANK=off`, moderate `academic_top_k` / `bq_top_k`.

---

### `uvicorn ml.rag.api:app`

**Files:** [`api.py`](../api.py) → [`app/api.py`](../app/api.py)

FastAPI server: `POST /query` with optional `session_id` for server-side chat memory.

```bash
uvicorn ml.rag.api:app --host 0.0.0.0 --port 8000
```

---

### `python -m ml.rag.chatbot.run`

**File:** [`chatbot/run.py`](../chatbot/run.py)

Thin wrapper around `run_rag` (same as CLI, alternate module path).

---

## 2. Qdrant collection management

### `python -m ml.rag.scripts.create_qdrant_collections`

**File:** [`scripts/create_qdrant_collections.py`](../scripts/create_qdrant_collections.py)

Creates all four RAG collections using specs from [`scripts/qdrant_collection_specs.py`](../scripts/qdrant_collection_specs.py).

| Flag | Effect |
|------|--------|
| *(none)* | Delete + recreate each collection |
| `--skip-existing` | Create only missing collections |
| `--indexes-only` | Apply payload indexes only (idempotent) |

**Requires:** `QDRANT_URL`, `QDRANT_API_KEY`

```bash
python -m ml.rag.scripts.create_qdrant_collections
python -m ml.rag.scripts.create_qdrant_collections --indexes-only
python -m ml.rag.scripts.create_qdrant_collections --skip-existing
```

**Dims:** from `chunking_config.PROFILES` and `RAG_QDRANT_VECTOR_SIZE_*` env vars.

---

## 3. Ingestion (Google Drive → JSONL → Qdrant)

### `python -m ml.rag.ingestion.cli`

**File:** [`ingestion/cli.py`](../ingestion/cli.py)

End-to-end rebuild: sync Drive folder → preprocess → upsert Qdrant.

#### Subcommand: `rebuild`

| Argument | Values | Description |
|----------|--------|-------------|
| `--kind` | `news`, `research`, `data_descriptions`, `ota`, `all` | Which pipeline |
| `--reset` | flag | Delete Qdrant collection before upsert |
| `--json` | flag | Machine-readable JSON output |

**Requires:** `GDRIVE_FOLDER_*_ID` for the kind, `QDRANT_*`, credentials for Drive sync.

```bash
python -m ml.rag.ingestion.cli rebuild --kind news
python -m ml.rag.ingestion.cli rebuild --kind all --reset
python -m ml.rag.ingestion.cli rebuild --kind research --json
```

**Output JSONL paths** (under `ml-eng/data/local/preprocessed_data/`):

| `--kind` | JSONL file |
|----------|------------|
| `news` | `news_chunks.jsonl` |
| `research` | `research_chunks.jsonl` |
| `data_descriptions` | `data_descriptions_chunks.jsonl` |
| `ota` | `ota_insights_chunks.jsonl` |

---

## 4. Preprocess (local folders → JSONL)

### `python -m ml.rag.text_processors.preprocess.cli`

**File:** [`text_processors/preprocess/cli.py`](../text_processors/preprocess/cli.py)

Unified preprocessor using corpus **engines** under `preprocess/engines/`.

#### Subcommand: `run`

| Argument | Required | Description |
|----------|----------|-------------|
| `--corpus` | yes | `news`, `research`, `data_description`, `ota` |
| `--input-dir` | yes | Folder of source files |
| `--output` | no | JSONL path (default from [`paths.py`](../paths.py)) |

```bash
python -m ml.rag.text_processors.preprocess.cli run --corpus news --input-dir data/local/raw/news
python -m ml.rag.text_processors.preprocess.cli run --corpus research --input-dir data/local/raw/research
python -m ml.rag.text_processors.preprocess.cli run --corpus data_description --input-dir data/local/raw/bq_descriptions
python -m ml.rag.text_processors.preprocess.cli run --corpus ota --input-dir data/local/raw/ota
```

#### Subcommand: `validate`

| Argument | Description |
|----------|-------------|
| `--jsonl` | Path to chunk JSONL; prints schema/token stats |

```bash
python -m ml.rag.text_processors.preprocess.cli validate --jsonl data/local/preprocessed_data/news_chunks.jsonl
```

---

### Legacy corpus preprocessor wrappers

These call the same engines or older paths; prefer `preprocess.cli` for new work.

| Module | Description |
|--------|-------------|
| `ml.rag.text_processors.news_collection_preprocessor` | News folder → JSONL |
| `ml.rag.text_processors.research_papers_preprocessor` | Research PDFs → JSONL |
| `ml.rag.text_processors.data_descriptions_preprocessor` | BQ description docs → JSONL |
| `ml.rag.text_processors.ota_insights_preprocessor` | OTA docs → JSONL |
| `ml.rag.text_processors.pdf_preprocessor` | Generic PDF → JSONL |
| `ml.rag.text_processors.news_preprocessor` | Older news path |

Example:

```bash
python -m ml.rag.text_processors.research_papers_preprocessor --input-dir path/to/pdfs --output data/local/preprocessed_data/research_chunks.jsonl
```

(Run with `-h` on each module for exact flags.)

---

## 5. Load JSONL → Qdrant

All loaders use shared upsert logic in [`text_processors/load_pdf_chunks_to_vector_db.py`](../text_processors/load_pdf_chunks_to_vector_db.py) unless noted.

**Common flags:**

| Flag | Description |
|------|-------------|
| `--input` | Chunk JSONL path |
| `--collection` | Qdrant collection name (defaults from env) |
| `--batch-size` | Upsert batch size |
| `--reset` | Delete and recreate collection before load |

**Requires:** `QDRANT_URL`, `QDRANT_API_KEY`, embedding model available locally or via HF.

### Per-corpus loaders (recommended)

| Module | Default collection env | Default input |
|--------|------------------------|---------------|
| `ml.rag.text_processors.news_load_to_vector_db` | `QDRANT_COLLECTION_NEWS` | `news_chunks.jsonl` |
| `ml.rag.text_processors.research_papers_load_to_vector_db` | `QDRANT_COLLECTION_RESEARCH_PAPERS` | `research_chunks.jsonl` |
| `ml.rag.text_processors.data_descriptions_load_to_vector_db` | `QDRANT_COLLECTION_DATA_DESCRIPTIONS` | `data_descriptions_chunks.jsonl` |
| `ml.rag.text_processors.ota_insights_load_to_vector_db` | `QDRANT_COLLECTION_OTA_INSIGHTS` | `ota_insights_chunks.jsonl` |

```bash
python -m ml.rag.text_processors.news_load_to_vector_db --reset
python -m ml.rag.text_processors.research_papers_load_to_vector_db --input data/local/preprocessed_data/research_chunks.jsonl
python -m ml.rag.text_processors.data_descriptions_load_to_vector_db --reset
python -m ml.rag.text_processors.ota_insights_load_to_vector_db --reset
```

### Generic loader

**Module:** `ml.rag.text_processors.load_pdf_chunks_to_vector_db`

```bash
python -m ml.rag.text_processors.load_pdf_chunks_to_vector_db --input data/local/preprocessed_data/ota_insights_chunks.jsonl --collection OTA_insights --reset
```

**Note:** Default `--collection` is `opentrace_rag` (legacy); always pass `--collection` explicitly.

---

## 6. Evaluation

### `python -m ml.rag.eval.run_retrieval_eval`

**File:** [`eval/run_retrieval_eval.py`](../eval/run_retrieval_eval.py)

Smoke **recall@k** against live Qdrant using YAML questions in [`eval/questions/`](../eval/questions/).

| Argument | Description |
|----------|-------------|
| `--corpus` | `news`, `research`, `data_description`, or `all` |
| `--k` | Top-k to score (default 5) |

```bash
python -m ml.rag.eval.run_retrieval_eval --corpus news --k 5
python -m ml.rag.eval.run_retrieval_eval --corpus all --k 10
```

---

## 7. Diagnostics and utilities

### `python -m ml.rag.inspect_vector_db`

**File:** [`inspect_vector_db.py`](../inspect_vector_db.py)

Sample Qdrant payloads and metadata fields for a collection.

| Argument | Description |
|----------|-------------|
| `--collection` | Collection name |
| `--limit` | Number of points to scroll (default 5) |

```bash
python -m ml.rag.inspect_vector_db --collection news_data --limit 10
```

---

### `python -m ml.rag.check_hf`

**File:** [`check_hf.py`](../check_hf.py)

Verify Hugging Face token and router connectivity (when not using LM Studio).

```bash
python -m ml.rag.check_hf
```

---

### `python -m ml.rag.helpers.generate_table_yamls`

**File:** [`helpers/generate_table_yamls.py`](../helpers/generate_table_yamls.py)

Offline helper to generate or refresh per-table YAML under `bq_tables_yaml_files/` from bronze catalog sources. Run with `-h` for options.

---

## 8. Typical workflows

### A. Greenfield Qdrant + one corpus

```bash
python -m ml.rag.scripts.create_qdrant_collections --skip-existing
python -m ml.rag.text_processors.preprocess.cli run --corpus news --input-dir data/local/raw/news
python -m ml.rag.text_processors.news_load_to_vector_db
python -m ml.rag.eval.run_retrieval_eval --corpus news --k 5
```

### B. Full Drive rebuild (production-style)

```bash
python -m ml.rag.ingestion.cli rebuild --kind all --reset
```

### C. Re-embed after model change

1. Update `RAG_EMBEDDING_MODEL_*` or bump `INGEST_VERSION` in `chunking_config.py`.
2. Recreate collection: `create_qdrant_collections` (or `--reset` on loader).
3. Re-run loader with `--reset`.

### D. Local dev query loop

```bash
# .env: RAG_LLM_BASE_URL=http://127.0.0.1:1234/v1, RAG_LLM_RERANK=off
python -m ml.rag.run "IPC phase trends in Ethiopia since 2020"
streamlit run ml/rag/chatbot/streamlit_app.py
```

---

## 9. Environment checklist by script type

| Script type | Minimum env |
|-------------|-------------|
| Load / eval / inspect | `QDRANT_URL`, `QDRANT_API_KEY` |
| Ingest rebuild | Above + `GDRIVE_FOLDER_*_ID`, Drive credentials |
| `ml.rag.run` / Streamlit / API | Above + `BQ_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, LLM URL or `HF_API_TOKEN` |
| NL-to-SQL quality | `RAG_LLM_BASE_URL`, `RAG_BRONZE_MODEL_YAML`, populated `BQ_table_descriptions` |

---

## 10. Tests (not CLI, but related)

| File | Run |
|------|-----|
| [`retrievers/test_bq_filter_context.py`](../retrievers/test_bq_filter_context.py) | `python -m pytest ml/rag/retrievers/test_bq_filter_context.py` (if pytest installed) or `python ml/rag/retrievers/test_bq_filter_context.py` |

---

*Script reference for `ml/rag` only. Parent repo scripts live outside this package.*
