# OpenTrace RAG & Chatbot API Reference

**Software team handoff (one document):** [OpenTrace-Ask-ADZA-API-Software-Team.docx](OpenTrace-Ask-ADZA-API-Software-Team.docx) — production base URL, six plan-scoped `POST /query/{plan}` routes, session/feedback/health. Regenerate with `python scripts/generate_software_team_api_docx.py` (from `ml-eng/`). Live Swagger: https://data-team-production-db77.up.railway.app/docs

**Internal / detailed:** [OpenTrace-RAG-API-Documentation.docx](OpenTrace-RAG-API-Documentation.docx) · [OpenTrace-Chatbot-API-v1-Documentation.docx](OpenTrace-Chatbot-API-v1-Documentation.docx) · [OpenTrace-RAG-Pipeline-Architecture.docx](OpenTrace-RAG-Pipeline-Architecture.docx) · [OpenTrace-RAG-Pipeline-Architecture.pdf](OpenTrace-RAG-Pipeline-Architecture.pdf) — regenerate API detail with `python scripts/generate_api_documentation.py`; pipeline DOCX with `python scripts/generate_rag_architecture_docx.py`; pipeline PDF with `python scripts/generate_rag_architecture_pdf.py`.

Two FastAPI services share request/response models but serve different clients. **Production Railway** runs the RAG API (`ml.rag.api:app`) with plan-scoped routes `POST /query/free` … `/query/integrated` (path locks `plan_type`).

| Service | Entrypoint | Default port | Audience |
|---------|------------|--------------|----------|
| **RAG API** (internal / Ask ADZA backend) | `ml.rag.api:app` | **7860** | Full pipeline + tuning + debug trace |
| **Chatbot API** (public v1) | `ml.serving.chat.app:app` | **7861** (7860 on HF Spaces) | Versioned chat + sessions; no retrieval knobs |

**Run locally:**

```bash
# RAG API
PYTHONPATH=ml-eng uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860

# Chatbot API
PYTHONPATH=ml-eng uvicorn ml.serving.chat.app:app --host 0.0.0.0 --port 7861
```

**OpenAPI:** `GET /docs` and `GET /openapi.json` on each host.

**Authentication:** None today. Endpoints are open if the URL is known. CORS is configurable via `RAG_CORS_ORIGINS` / `CHATBOT_CORS_ORIGINS` (default `*`).

---

## Shared types

### `UserProfile`

When `user_profile` is sent, **`plan_type`** and **`category`** are required. Unknown keys (e.g. legacy `stakeholder_type`) → **422**.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `country` | `string \| null` | No | **Free / single-country plans:** preferred country when clamping decomposition to one geography (profile wins over query-extracted geo). **Farmers only:** also applied as retrieval `geo_override` filter. Not a retrieval geo filter for Free, Government, NGOs, Agribusinesses, or Integrated. |
| `plan_type` | `string` | **Yes** (if profile sent) | Access tier and retrieval gates. One of the [plan types](#plan-types). |
| `category` | `string` | **Yes** (if profile sent) | Generation persona / tone. One of the [categories](#categories). |

### `ChatMessage`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | `string` | Yes | `"user"` or `"assistant"` |
| `content` | `string` | Yes | Non-empty message text |

### `CitationItem`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Source id (matches inline `[N]` when footnotes are present; otherwise packed-context index). |
| `kind` | `string` | Normalized source type: `academic`, `news`, `structured_data`, `policy`, `ota`, `web_wikipedia`, `web_search`, etc. |
| `text` | `string` | Human-readable citation line, e.g. `[Academic] Branca et al. (2012). …`, `[News] Title — Publisher (2025-12-31)`. |
| `url` | `string \| null` | Link when available (news, DOI, Wikipedia, web). Often `null` for academic/structured data. |

**Default chat** returns prose **without** inline `[N]` footnotes; render `citations[]` in the UI. Inline footnotes appear for analytical write-ups, DOCX/PDF/multi exports, or when the user asks for footnotes / inline citations. In that mode, `RAG_CITATIONS_MODE=referenced` (default) keeps only cited sources; `all` includes every packed source. When inline footnotes are off, `citations` lists the packed sources for the turn.

### `UsageStats`

Aggregated LLM token usage for **one request** (decomposer, BQ NL→SQL, reranker if enabled, final generation). Embeddings are not counted.

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `integer` | Input / prompt tokens |
| `output_tokens` | `integer` | Output / completion tokens |
| `total_tokens` | `integer` | Total reported by provider |

---

## Plan types

`plan_type` controls access tier, retrieval gates, and generation depth. Returned by `GET /v1/meta` as `plan_types`.

| ID | Retrieval / generation gates |
|----|------------------------------|
| `Free` | Single country; no cross-country compare; brief answers. Profile `country` preferred when clamping decomposition geography (not a retrieval geo filter). |
| `Farmers` | Profile `country` as retrieval geo filter **and** single-country decomposition clamp; plain-language framing |
| `Government` | National/sub-national + historical trends; no cross-country |
| `NGOs` | Government-tier depth + multi-region overlap framing |
| `Agribusinesses` | Cross-country comparison allowed; market/volatility framing |
| `Integrated` | No artificial plan caps beyond category persona |

Invalid values → **422** with `invalid plan_type: '…'`.

---

## Categories

`category` controls generation persona (tone and framing). Returned by `GET /v1/meta` as `categories`.

| ID | Label |
|----|-------|
| `Government` | Government & Public Institutions |
| `NGOs` | Foundations, NGOs & Development Partners |
| `Agribusinesses` | Agribusinesses & Financial Institutions |
| `Farmers` | Farmers, Cooperatives & Communities |

**Resolution order** (both APIs): `user_profile.plan_type` + `user_profile.category` → session `category` fallback (when `session_id` is set and category omitted).

Invalid values → **422** with `invalid category: '…'`.

Legacy fields `stakeholder_type`, `audience_instructions`, and top-level `geo_override` are **rejected** (no fallback).

---

## Conversation memory

Two patterns:

### A. Server-side session (stateful)

1. Omit `chat_history` on each request.
2. Reuse `session_id` from the previous response.
3. Server stores `{conversation_summary, recent_turns}` in Redis (`RAG_REDIS_URL`) or in-process fallback.
4. Session TTL default: **604800s** (7 days) via `RAG_SESSION_TTL_SECONDS`. Responses include `session_ttl_seconds` (configured value) and `session_found` (`true` only when a prior blob was loaded for that id).
5. Expired or unknown `session_id` → empty memory and `session_found: false` (no error).

### B. Client-owned history (stateless)

1. Send `chat_history` with prior turns.
2. Server compacts history for that request only.
3. Server session is **not** updated when `chat_history` is present (`POST /query` and `POST /v1/chat`).

`conversation_history` is a **deprecated alias** for `chat_history`.

---

# RAG API (`ml.rag.api:app`)

Implementation: [`ml/rag/app/api.py`](../app/api.py). Uvicorn shim: [`ml/rag/api.py`](../api.py).

## `GET /`

Service index.

**Response 200:**

```json
{
  "message": "OpenTrace RAG API",
  "docs": "/docs",
  "health": "/health",
  "query": "POST /query"
}
```

---

## `GET /health`

Liveness probe. Always fast; does not check Qdrant/LLM.

**Response 200:**

```json
{
  "status": "ok",
  "service": "rag"
}
```

---

## `GET /ready`

Readiness probe for load balancers.

**Response 200:**

```json
{
  "status": "ready",
  "service": "rag",
  "missing_config_keys": [],
  "bq": {
    "project_set": true,
    "project": "opentrace-prod-5ga4",
    "gcp": {
      "credentials_path_set": true,
      "credentials_base64_set": true,
      "path": "/tmp/gcp-sa-key.json",
      "json_ok": true
    },
    "ok": true
  },
  "redis": {
    "backend": "redis",
    "connected": true
  }
}
```

| Field | Meaning |
|-------|---------|
| `status` | `"ready"` if Qdrant + LLM env present **and** BigQuery is ready when `BQ_PROJECT` is set; else `"not_ready"` |
| `missing_config_keys` | e.g. `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_LLM_BASE_URL+RAG_LLM_API_KEY (or HF_API_TOKEN)`, `BQ_PROJECT+GCP credentials` |
| `bq` | Always present. When `BQ_PROJECT` is set: GCP SA path must exist + parse as JSON, and a lightweight BigQuery `datasets.list` must succeed — otherwise `ok: false` and status is `not_ready`. When unset: `ok: true` with `skipped`. |
| `redis` | Present when `RAG_REDIS_URL` / `REDIS_URL` is set; informational only (non-fatal) |

Railway bootstrap writes validated credentials to **`/tmp/gcp-sa-key.json`** from `GOOGLE_APPLICATION_CREDENTIALS_BASE64`. Do **not** point `GOOGLE_APPLICATION_CREDENTIALS` at a stale `/tmp/gcp-sa.json`.

---

## Plan-scoped query routes (preferred)

Production RAG exposes one endpoint per plan. The **path locks `plan_type`**; a mismatched `user_profile.plan_type` in the body is ignored.

| Method | Path | Injected `plan_type` |
|--------|------|----------------------|
| POST | `/query/free` | `Free` |
| POST | `/query/farmers` | `Farmers` |
| POST | `/query/government` | `Government` |
| POST | `/query/ngos` | `NGOs` |
| POST | `/query/agribusinesses` | `Agribusinesses` |
| POST | `/query/integrated` | `Integrated` |

Request/response body matches [`POST /query`](#post-query). Send `user_profile.category`. For **Farmers**, also send `country` (retrieval geo filter). For **Free**, `country` is recommended so single-country decomposition prefers the profile country.

---

## `POST /query`

Main RAG endpoint (generic / backward compatible). Prefer plan-scoped `/query/{plan}` routes for new clients. Runs the full graph: decomposition → BigQuery + vector retrieval → rerank → generation.

### Request body (`QueryRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | `string` | **Yes** | — | Natural-language question (min length 1). |
| `session_id` | `string \| null` | No | new UUID | Reuse for server-side multi-turn memory. |
| `user_profile` | `UserProfile \| null` | No | `null` | `country`, `plan_type`, `category` (all three when sent). |
| `chat_history` | `ChatMessage[] \| null` | No | `null` | Prior turns (canonical). |
| `conversation_history` | `ChatMessage[] \| null` | No | `null` | Deprecated alias for `chat_history`. |
| `include_trace` | `boolean` | No | `false` | Include retrieval/debug counts in response. |
| `time_start_override` | `string \| null` | No | `null` | ISO date filter override for retrieval. |
| `time_end_override` | `string \| null` | No | `null` | ISO date filter override. |
| `news_top_k` | `integer \| null` | No | `null` | Max news chunks. |
| `academic_top_k` | `integer \| null` | No | `null` | Max academic chunks. |
| `bq_top_k` | `integer \| null` | No | `null` | Max BigQuery result rows/chunks. |
| `rerank_top_k` | `integer \| null` | No | `null` | Chunks after reranking. |
| `ota_top_k` | `integer \| null` | No | `null` | Max OTA insight chunks. |

Unknown top-level keys (e.g. `stakeholder_type`, `audience_instructions`, `geo_override`) → **422**.

### Canonical request example

```json
{
  "query": "What are rice yield trends in Nigeria?",
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

### Minimal request

```json
{
  "query": "Who are you?"
}
```

### Response 200 (`QueryResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | Prose answer. Default chat has **no** inline `[N]` footnotes (use `citations[]`). Inline footnotes appear for write-ups / DOCX-PDF exports / explicit request. No trailing Sources block by default. |
| `citations` | `CitationItem[]` | Structured sources for UI rendering. |
| `session_id` | `string` | Pass on the next request for continuity. |
| `session_found` | `boolean` | `true` only when a prior server session blob was loaded. `false` for new/expired/missing ids or when `chat_history` is sent. |
| `session_ttl_seconds` | `integer` | Configured session TTL (`RAG_SESSION_TTL_SECONDS`, default **604800**). |
| `usage` | `UsageStats` | Per-request LLM token totals. |
| `error` | `string \| null` | Pipeline-level error message if the graph set one; may still return partial `answer`. |
| `trace` | `object \| null` | Present only when `include_trace: true`. |
| `langfuse_trace_id` | `string \| null` | Langfuse trace id when tracing is enabled. |
| `acf` | `ACFSignal` | Confidence band, score, and explanation. |
| `artifacts` | `ArtifactItem[]` | Downloadable exports. Populated on **Agribusinesses** and **Integrated** when the user asks for CSV/chart/PDF/DOCX and builders succeed; otherwise `[]`. Production URLs are signed for **`RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS` (default 86400 = 24h)**. Refresh with [`GET /artifacts/{artifact_id}/url`](#get-artifactsartifact_idurl). |

Plan-scoped routes inherit the same response. Export gate: `/query/agribusinesses` and `/query/integrated` (and generic `/query` when `user_profile.plan_type` is one of those). Other plans keep `artifacts: []` and may append an upgrade note in `answer` if an export is requested.

### Example success response

```json
{
  "answer": "Nigeria's rice production has trended upward. According to Ariom and Dimon (2022), improved varieties are widely used among cereal growers. Business News Nigeria (2025) reports regional price variation in May 2026.",
  "citations": [
    {
      "id": 7,
      "kind": "academic",
      "text": "[Academic] Ariom, T.O.; Dimon, E.; (2022). DOI 10.3390/su141811370",
      "url": "https://doi.org/10.3390/su141811370"
    },
    {
      "id": 9,
      "kind": "news",
      "text": "[News] Agriculture in 2025: Trends that shaped Nigeria's food production — Business News Nigeria (2025-12-31)",
      "url": "https://news.google.com/rss/articles/..."
    }
  ],
  "session_id": "787a6c2201104ad9a704068cf525c1d4",
  "usage": {
    "input_tokens": 6977,
    "output_tokens": 653,
    "total_tokens": 7630
  },
  "error": null,
  "trace": null
}
```

### `trace` object (when `include_trace: true`)

```json
{
  "decomposition": { "intent": "descriptive", "geography": ["Nigeria"], "time_start": "", "time_end": "" },
  "bq_table_candidates_count": 3,
  "vector_news_count": 8,
  "vector_academic_count": 12,
  "merged_context_count": 20,
  "reranked_context_count": 10
}
```

### Error responses

| Status | When | Body |
|--------|------|------|
| **422** | Invalid `plan_type` / `category`, unknown fields, validation errors | `{"detail": "invalid plan_type: 'foo'"}` |
| **500** | Unhandled exception | `{"detail": "error message"}`; full traceback if `RAG_DEBUG=1` |

### Behavior notes

- **Geo (Free):** `user_profile.country` is preferred when clamping decomposition to a single country. It is **not** applied as a retrieval-level `geo_override`.
- **Geo (Farmers):** `user_profile.country` is both the decomposition preference and a retrieval geo filter.
- **Plan gates:** Cross-country retrieval/compare is limited to `Agribusinesses` and `Integrated` plans.
- **Omit `user_profile`:** Minimal queries (e.g. `"Who are you?"`) work with generic tone and no plan gates.
- **Meta questions** (“Who are you?”, product FAQs) may short-circuit retrieval without full RAG.
- **Citations in UI:** Map `[N]` in `answer` to `citations` where `id === N`.
- **Legacy Sources in answer:** Set server env `RAG_APPEND_SOURCES_TO_ANSWER=1` to append a markdown Sources block to `answer`.

---

# Chatbot API v1 (`ml.serving.chat.app:app`)

Implementation: [`ml/serving/chat/app.py`](../../serving/chat/app.py). See also [`ml/serving/README.md`](../../serving/README.md).

## `GET /`

```json
{
  "message": "OpenTrace Chatbot API",
  "docs": "/docs",
  "health": "/v1/health",
  "meta": "/v1/meta",
  "sessions": "POST /v1/sessions",
  "chat": "POST /v1/chat"
}
```

---

## `GET /v1/health`

**Response 200:**

```json
{
  "status": "ok",
  "service": "chatbot"
}
```

---

## `GET /v1/meta`

API catalog and stakeholder list.

**Response 200:**

```json
{
  "api_version": "1.0",
  "schema_version": "1",
  "build": null,
  "stakeholder_types": [
    {
      "id": "government_public",
      "label": "Government & Public Institutions",
      "description": "Planning, policy design, and resource allocation..."
    }
  ]
}
```

`build` is set when env `CHATBOT_BUILD_ID` is configured.

---

## `POST /v1/sessions`

Create a server-side session with a fixed category persona.

### Request (`SessionCreateRequest`)

| Field | Type | Required |
|-------|------|----------|
| `category` | `Category` | **Yes** |

```json
{
  "category": "Government"
}
```

### Response 200 (`SessionCreateResponse`)

```json
{
  "session_id": "a1b2c3d4e5f6...",
  "created_at": "2026-06-08T12:00:00+00:00",
  "category": "Government"
}
```

### Errors

| Status | When |
|--------|------|
| **422** | Invalid `category` |

---

## `POST /v1/chat`

Single chat turn through the same RAG pipeline as `/query`, with v1 response shape.

### Request (`ChatRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `string \| null` | One of `query` / `message` | User message. |
| `message` | `string \| null` | One of `query` / `message` | Alias for `query`. **Do not send both.** |
| `session_id` | `string \| null` | Conditional | From `POST /v1/sessions` or prior chat response. |
| `user_profile` | `UserProfile \| null` | No | Same as `/query`. |
| `chat_history` | `ChatMessage[] \| null` | No | Client-owned prior turns. |
| `conversation_history` | `ChatMessage[] \| null` | No | Deprecated alias. |
| `stakeholder_type` | `StakeholderType \| null` | No | **Deprecated bootstrap only.** Allowed only when `session_id` is omitted. Cannot combine with `session_id`. |

### Session bootstrap rules

| Scenario | Required |
|----------|----------|
| No `chat_history`, no `session_id` | `user_profile.stakeholder_type` (or deprecated top-level `stakeholder_type`) to auto-create session |
| No `chat_history`, with `session_id` | `session_id` from prior session/chat |
| With `chat_history` | `session_id` **or** `user_profile.stakeholder_type` |

### Canonical request

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
  ]
}
```

### Response 200 (`ChatSuccessResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `assistant_message` | `string` | Same content as `answer` on `/query`. |
| `citations` | `CitationItem[]` | Same as `/query`. |
| `session_id` | `string` | Session for next turn. |
| `session_found` | `boolean` | `true` only when a prior server session blob was loaded. `false` for new/expired/missing ids or when `chat_history` is sent. |
| `session_ttl_seconds` | `integer` | Configured session TTL (`RAG_SESSION_TTL_SECONDS`, default **604800**). |
| `usage` | `UsageStats` | Same as `/query`. |
| `request_id` | `string` | Unique ID for this HTTP request (support/debug). |
| `created_at` | `string` | ISO-8601 UTC timestamp. |
| `plan_type` | `string \| null` | Plan tier applied to this request (when using plan-scoped routes). |
| `acf` | `ACFSignal` | Confidence band, score, and explanation. |
| `artifacts` | `ArtifactItem[]` | Downloadable exports. Populated on **Agribusinesses** and **Integrated** for both `POST /query/{plan}` and `POST /v1/chat/{plan}` when export intent succeeds. Always `[]` on other plans. |

### Plan-scoped chat routes

Prefer these routes for new integrations. The plan tier is **locked by the URL**; the payload cannot override it.

| Route | Plan | Exports (`artifacts`) |
|-------|------|------------------------|
| `POST /query/free` / `POST /v1/chat/free` | Free | No |
| `POST /query/farmers` / `POST /v1/chat/farmers` | Farmers | No |
| `POST /query/government` / `POST /v1/chat/government` | Government | No |
| `POST /query/ngos` / `POST /v1/chat/ngos` | NGOs | No |
| `POST /query/agribusinesses` / `POST /v1/chat/agribusinesses` | Agribusinesses | **Yes** (CSV, chart, DOCX, PDF) |
| `POST /query/integrated` / `POST /v1/chat/integrated` | Integrated | **Yes** (CSV, chart, DOCX, PDF) |

When a user on a non-export route asks for a CSV, chart, or report, the assistant explains that exports require the Agribusinesses or Integrated endpoint.

### `ArtifactItem` (export-enabled routes only)

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Stable artifact id for this response |
| `kind` | `csv \| chart \| docx \| pdf` | Export format |
| `filename` | `string` | Suggested download filename |
| `mime_type` | `string` | MIME type |
| `url` | `string` | Presigned HTTPS URL (S3-compatible or GCS in production). **Expires after the signed TTL (default 86400s / 24h).** |
| `summary` | `string` | Short description of contents |
| `citation_ids` | `integer[]` | Citation ids from the parent answer |
| `byte_size` | `integer` | File size in bytes |

---

## `GET /artifacts/{artifact_id}/url`

Re-sign a download URL after the original presigned URL expires.

**Query params**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | `string` | **Yes** | Exact `filename` from the original `ArtifactItem` |

**Response 200**

```json
{
  "id": "art_abc123def456",
  "filename": "nigeria_maize.csv",
  "url": "https://...",
  "expires_in_seconds": 86400,
  "storage_uri": "s3://bucket/rag-exports/art_abc123def456/nigeria_maize.csv"
}
```

| Status | When |
|--------|------|
| **400** | Missing/unsafe `artifact_id` or `filename` |
| **404** | Object not found in configured storage |
| **500** | Signing / storage error |

Env: `RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS` (default **86400** / 24 hours, minimum 60). Session blobs last 7 days; download URLs last 24 hours. Refresh with this endpoint after expiry.

---

```json
{
  "assistant_message": "Maize production rose…\n\nDownloadable files are attached to this response: nigeria_maize.csv, nigeria_maize.png.",
  "citations": [{ "id": 3, "kind": "bigquery", "text": "...", "url": null }],
  "acf": { "band": "strong", "band_label": "Strong confidence", "score": 78, "explanation": "..." },
  "session_id": "abc123...",
  "usage": { "input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600 },
  "request_id": "f4e2...",
  "created_at": "2026-06-08T12:00:00+00:00",
  "plan_type": "Agribusinesses",
  "artifacts": [
    {
      "id": "art_a1b2c3d4e5f6",
      "kind": "csv",
      "filename": "nigeria_maize.csv",
      "mime_type": "text/csv",
      "url": "https://storage.googleapis.com/...",
      "summary": "CSV export (24 rows)",
      "citation_ids": [3],
      "byte_size": 12480
    }
  ]
}
```

### Error responses

| Status | When | Body shape |
|--------|------|------------|
| **422** | Validation (missing message, invalid bootstrap, empty `chat_history`, etc.) | `{"detail": "..."}` |
| **502** | RAG pipeline returned `error` | `{"error": {"code": "rag_pipeline_error", "message": "..."}, "session_id": "..."}` |
| **500** | Unhandled exception | `{"detail": "..."}`; traceback if `CHATBOT_DEBUG=1` |

---

## `/query` vs `/v1/chat`

| | `POST /query` | `POST /v1/chat` |
|--|---------------|-----------------|
| Answer field | `answer` | `assistant_message` |
| Retrieval tuning | `news_top_k`, `bq_top_k`, etc. | Not exposed |
| Debug trace | `include_trace` | Not exposed |
| Session create | Implicit UUID | `POST /v1/sessions` or bootstrap |
| Pipeline error | `error` string in 200 body | **502** JSON |
| Extra fields | `session_found`, `session_ttl_seconds` | `request_id`, `created_at`, `session_found`, `session_ttl_seconds` |

Both use the same RAG graph, `UserProfile`, `chat_history`, `citations`, and `usage`.

---

## Recommended integration (Ask ADZA backend)

```
1. POST /query (or /v1/chat) with service-to-service call
2. Store usage + your auth user_id in your DB for billing
3. Render answer prose + citations[] separately in the UI
4. Link inline [N] to citation cards
5. Reuse session_id OR send chat_history each turn (pick one pattern)
```

### Multi-turn with server session

```json
// Turn 1
{ "query": "Rice trends in Nigeria?", "user_profile": { "plan_type": "Government", "category": "Government" } }
// → session_id: "xyz"

// Turn 2
{ "query": "What about Ghana?", "session_id": "xyz", "user_profile": { "plan_type": "Government", "category": "Government" } }
```

### Multi-turn with client history

```json
{
  "query": "What about Ghana?",
  "user_profile": { "plan_type": "Government", "category": "Government" },
  "chat_history": [
    { "role": "user", "content": "Rice trends in Nigeria?" },
    { "role": "assistant", "content": "..." }
  ]
}
```

---

## Environment variables (client-relevant)

| Variable | Service | Purpose |
|----------|---------|---------|
| `RAG_CORS_ORIGINS` | RAG | Comma-separated allowed origins |
| `CHATBOT_CORS_ORIGINS` | Chat | Comma-separated allowed origins |
| `RAG_DEBUG` | RAG | Richer 500 errors |
| `CHATBOT_DEBUG` | Chat | Richer 500/502 debug |
| `RAG_APPEND_SOURCES_TO_ANSWER` | RAG | Append Sources block into `answer` |
| `RAG_CITATIONS_MODE` | RAG | `referenced` (default) or `all` |

---

## Source files

| File | Role |
|------|------|
| [`ml/rag/app/api.py`](../app/api.py) | RAG routes (`/query`, `/health`, `/ready`, `/artifacts/{id}/url`) |
| [`ml/rag/api_schemas.py`](../api_schemas.py) | Shared `UserProfile`, `CitationItem`, `UsageStats` |
| [`ml/rag/request_context.py`](../request_context.py) | Request field resolution |
| [`ml/serving/chat/app.py`](../../serving/chat/app.py) | v1 chat routes |
| [`ml/serving/chat/schemas.py`](../../serving/chat/schemas.py) | v1 request/response models |
