---
name: Structured API response
overview: Extend RAG responses with structured citations (separate from answer prose), aggregated LLM usage tokens, and session_id — on both POST /query and POST /v1/chat, while keeping backward-compatible answer text without embedded Sources block.
todos:
  - id: llm-usage
    content: Add TokenUsage accumulator in llm_chat.py; parse usage from provider responses; reset in run_rag
    status: completed
  - id: generator-structured
    content: GenerationResult + _referenced_citations(); generate() returns prose answer + citations array
    status: completed
  - id: graph-state
    content: Propagate citations/usage through graph state and run_rag result
    status: completed
  - id: api-query
    content: Extend QueryResponse with citations + usage in api.py using shared schemas
    status: completed
  - id: api-v1-chat
    content: Extend ChatSuccessResponse and execute_chat_turn for citations + usage
    status: completed
  - id: tests-docs
    content: Tests for citations split, usage sum, API models; update README and .env.example
    status: completed
isProject: false
---

# Structured API response (answer + citations + usage)

## Goal

Return a backend-friendly JSON shape from both APIs:

```json
{
  "answer": "Prose with inline footnotes [14][18] only",
  "citations": [
    { "id": 14, "kind": "academic", "text": "[Academic] ...", "url": null },
    { "id": 18, "kind": "news", "text": "[News] ...", "url": "https://..." }
  ],
  "session_id": "...",
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

- **`answer`**: no trailing `Sources` markdown block (UI uses `citations`).
- **`usage`**: sum of all LLM calls in the request (decompose, BQ NL2SQL, memory fold, rerank if on, generation).

## Architecture

```mermaid
flowchart LR
  runRag[run_rag] --> resetUsage[reset_llm_usage]
  resetUsage --> graph[LangGraph pipeline]
  graph --> llmCalls[llm_chat_complete x N]
  llmCalls --> accumulate[add_llm_usage]
  graph --> gen[generate]
  gen --> prose[answer prose]
  gen --> cites[citations array]
  runRag --> result[state + usage]
  result --> queryApi[POST /query]
  result --> v1Chat[POST /v1/chat]
```

## 1. Token usage plumbing — [`ml/rag/llm_chat.py`](ml-eng/ml/rag/llm_chat.py)

- Add frozen `TokenUsage` dataclass: `prompt_tokens`, `completion_tokens`, `total_tokens` with `to_dict()` including `input_tokens` / `output_tokens` aliases.
- Add request-scoped accumulator:
  - `reset_llm_usage()` — called at start of `run_rag`
  - `get_llm_usage()` — read aggregated totals
  - `add_llm_usage(raw: dict)` — parse OpenAI-style `usage` from provider JSON
- In `llm_chat_complete`, after `resp.json()`, call `add_llm_usage(data.get("usage") or {})` before returning content string.
- **No caller changes** required in decomposer / BQ / reranker / memory — they already call `llm_chat_complete`.

## 2. Generator structured output — [`ml/rag/chatbot/generator.py`](ml-eng/ml/rag/chatbot/generator.py)

- Add `GenerationResult` dataclass: `answer: str`, `citations: list[dict]`.
- Refactor citation collection:
  - `_referenced_citations(answer, source_registry) -> list[dict]` — reuse logic from `_append_structured_citations` (referenced mode default); each item:
    - `id` (int), `kind` (normalized string), `text` (citation_line), `url` (from metadata when present: news/web/wikipedia)
  - `_citation_url(kind, meta)` helper for URL extraction
- Change `generate()` to return `GenerationResult` instead of `str`:
  - Prose path: `_clean_answer` → `_normalize_inline_citations` → **no** `_append_structured_citations`
  - Fallback paths (no context, LLM failure hint) return `citations=[]`
- Keep `_append_structured_citations` for optional legacy env `RAG_APPEND_SOURCES_TO_ANSWER=1` (off by default) so Streamlit can opt in during transition.

## 3. Graph state — [`ml/rag/chatbot/graph.py`](ml-eng/ml/rag/chatbot/graph.py)

- Extend `RAGGraphState`: `citations: list[dict]`, `usage: dict`.
- `node_generate`: unpack `GenerationResult`; set `answer`, `citations`.
- `node_generate_meta` / `node_generate_product`: set `citations=[]` (no retrieval sources).
- `run_rag()`:
  - Call `reset_llm_usage()` before `graph.invoke`
  - After invoke, attach `usage=get_llm_usage().to_dict()` to returned dict (alongside `answer`, `citations`)

## 4. `POST /query` — [`ml/rag/app/api.py`](ml-eng/ml/rag/app/api.py)

- Add Pydantic models:
  - `CitationItem` (`id`, `kind`, `text`, `url` optional)
  - `UsageStats` (`prompt_tokens`, `completion_tokens`, `total_tokens`, optional aliases)
- Extend `QueryResponse`:
  - `citations: list[CitationItem] = []`
  - `usage: UsageStats` with zero defaults
- Map from `run_rag` result; keep `error`, `trace`, `session_id` unchanged.

## 5. `POST /v1/chat` — [`ml/serving/chat/schemas.py`](ml-eng/ml/serving/chat/schemas.py) + [`ml/serving/chat/app.py`](ml-eng/ml/serving/chat/app.py)

- Extend `ChatSuccessResponse`:
  - `citations: list[CitationItem]` (reuse shared model or duplicate thin schema in serving)
  - `usage: UsageStats`
- `execute_chat_turn` / `ChatTurnResult`: add `citations`, `usage` fields populated from `raw_result`.
- `v1_chat` handler returns them in success JSON.

Shared types: add [`ml/rag/api_schemas.py`](ml-eng/ml/rag/api_schemas.py) (or `ml/rag/chatbot/response_models.py`) imported by both `api.py` and `serving/chat/schemas.py` to avoid duplication.

## 6. Streamlit (minimal)

- [`chatbot/streamlit_app.py`](ml-eng/ml/rag/chatbot/streamlit_app.py): if `result.get("citations")`, render below answer; else fall back to embedded Sources in `answer` when legacy env on.

## 7. Tests

- [`test_generator_context.py`](ml-eng/ml/rag/chatbot/test_generator_context.py): `_referenced_citations` returns correct ids; `generate()` returns `GenerationResult` without `Sources` in answer.
- New [`test_llm_usage.py`](ml-eng/ml/rag/test_llm_usage.py): `add_llm_usage` sums; `reset` clears.
- API model test: `QueryResponse` serializes citations + usage.
- Optional: `execute_chat_turn` passes citations through from mocked `run_rag`.

## 8. Docs

- [`README.md`](ml-eng/ml/rag/README.md): document response schema and that `answer` no longer includes `Sources` block by default.
- [`.env.example`](ml-eng/config/.env.example): `RAG_APPEND_SOURCES_TO_ANSWER=0` legacy note.

## Backward compatibility

| Consumer | Impact |
|----------|--------|
| Backend integrating `/query` | New fields; `answer` prose-only — **intended** |
| Old clients expecting Sources in `answer` | Set `RAG_APPEND_SOURCES_TO_ANSWER=1` or read `citations` |
| OpenAI `choices[0]` format | Still not supported — out of scope |

## Files to touch

| File | Change |
|------|--------|
| [`llm_chat.py`](ml-eng/ml/rag/llm_chat.py) | Usage accumulator + parse from provider |
| [`generator.py`](ml-eng/ml/rag/chatbot/generator.py) | `GenerationResult`, structured citations |
| [`graph.py`](ml-eng/ml/rag/chatbot/graph.py) | State fields, `run_rag` usage reset |
| [`api.py`](ml-eng/ml/rag/app/api.py) | Extended `QueryResponse` |
| [`api_schemas.py`](ml-eng/ml/rag/api_schemas.py) | **New** shared Pydantic models |
| [`chat_turn.py`](ml-eng/ml/rag/chat_turn.py) | Pass citations/usage |
| [`serving/chat/schemas.py`](ml-eng/ml/serving/chat/schemas.py) + [`app.py`](ml-eng/ml/serving/chat/app.py) | v1 response fields |
| Tests + README + `.env.example` | As above |
