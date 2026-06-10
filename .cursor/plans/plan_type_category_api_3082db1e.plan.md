---
name: plan_type category API
overview: Migrate the RAG API from stakeholder_type/audience_instructions to backend plan_type + category enums, enforce Farmers-only country retrieval, apply plan-tier feature gates on decomposition/generation, slim usage to three fields, and remove all deprecated persona fields.
todos:
  - id: schemas
    content: "Update api_schemas.py: UserProfile plan_type/category, slim UsageStats to 3 fields"
    status: completed
  - id: plan-policy
    content: Add plan_policy.py + refactor stakeholder_prompts to category personas
    status: completed
  - id: geo-request
    content: Update geo_policy.py and request_context.py (ResolvedRequestContext, validation)
    status: completed
  - id: graph-generator
    content: Wire plan_type/category through graph.py, generator.py, assistant_identity, product_knowledge, chat_turn
    status: completed
  - id: api-v1
    content: Update app/api.py, serving chat schemas/app; remove deprecated request fields
    status: completed
  - id: tests-docs
    content: Update/add tests; refresh API.md and README links
    status: completed
isProject: false
---

# plan_type + category API migration

## Target contract

**Request** (`user_profile` when present must include all three fields):

```json
{
  "user_profile": {
    "country": "Ghana",
    "plan_type": "Farmers",
    "category": "Farmers"
  }
}
```

- `plan_type`: `Free` | `Farmers` | `Government` | `NGOs` | `Agribusinesses` | `Integrated`
- `category`: `Government` | `NGOs` | `Agribusinesses` | `Farmers`
- Reject `stakeholder_type`, `audience_instructions`, top-level `geo_override` (no fallback)

**Response `usage`**: only `input_tokens`, `output_tokens`, `total_tokens`

**`citations[]`**: unchanged shape `{ id, kind, text, url }`

```mermaid
flowchart LR
    req[QueryRequest] --> resolve[resolve_request_context]
    resolve --> plan[plan_type gates]
    resolve --> cat[category persona]
    resolve --> geo[geo_policy Farmers only]
    plan --> graph[run_rag graph]
    cat --> graph
    geo --> graph
    graph --> resp[QueryResponse]
```

## 1. Shared schemas — [`ml/rag/api_schemas.py`](ml-eng/ml/rag/api_schemas.py)

- Replace `UserProfile` fields:
  - `plan_type: Literal[...]` (6 backend enums)
  - `category: Literal[...]` (4 backend enums)
  - `country: str | None`
  - `model_config = ConfigDict(extra="forbid")` to reject old keys
- When `user_profile` is sent, require `plan_type` + `category` (Pydantic required fields on the nested model)
- Slim `UsageStats` to **only** `input_tokens`, `output_tokens`, `total_tokens`; update `from_usage_dict()` to map internal `prompt_tokens`/`completion_tokens` from [`llm_chat.py`](ml-eng/ml/rag/llm_chat.py)

## 2. New plan policy module — `ml/rag/chatbot/plan_policy.py`

Centralize enums and tier behavior extracted from the pricing spec:

| `plan_type` | RAG gates |
|-------------|-----------|
| `Free` | Single-country retrieval; no cross-country compare; shallow answers (brief generator addendum); no deep multi-year trend framing |
| `Farmers` | Profile `country` geo filter; single-country; plain-language reinforcement |
| `Government` | National/sub-national + historical trends OK; **no** cross-country |
| `NGOs` | Inherits Government + multi-region overlap framing in generator |
| `Agribusinesses` | Cross-country comparison allowed; market/volatility framing |
| `Integrated` | No artificial plan caps beyond category persona |

Functions:

- `is_valid_plan_type` / `is_valid_category`
- `instruction_for_category(category)` — persona text (migrate from current [`stakeholder_prompts.py`](ml-eng/ml/rag/chatbot/stakeholder_prompts.py))
- `plan_generation_addendum(plan_type)` — tier-specific system-prompt lines
- `apply_plan_decomposition_gates(decomposition, plan_type, country)` — when cross-country disallowed: clamp `geography` to one country (prefer profile `country`, else first decomposed country); downgrade `compare` intent to `descriptive` on gated plans

## 3. Refactor persona prompts — [`ml/rag/chatbot/stakeholder_prompts.py`](ml-eng/ml/rag/chatbot/stakeholder_prompts.py)

- Replace `STAKEHOLDER_TYPES` / snake_case IDs with `CATEGORIES` catalog (4 entries, backend Title Case ids)
- Replace `instruction_for_stakeholder` → `instruction_for_category`
- Remove `entrepreneurs_ecosystem` from public API surface
- Keep module path for minimal churn; update imports in `assistant_identity.py`, `product_knowledge.py`, `serving/chat/app.py`

## 4. Geo policy — [`ml/rag/chatbot/geo_policy.py`](ml-eng/ml/rag/chatbot/geo_policy.py)

- Change gate from `stakeholder_type == farmers_communities` to **`plan_type == "Farmers"`**
- `effective_geo_override(plan_type, user_profile)` signature (drop stakeholder param)

## 5. Request resolution — [`ml/rag/request_context.py`](ml-eng/ml/rag/request_context.py)

- `ResolvedRequestContext` fields: `plan_type`, `category`, `user_profile` (dict with `country`, `plan_type`, `category`), `history_messages`
- Remove `legacy_stakeholder_type`, `legacy_audience_instructions`, session `stakeholder_type` fallback → session `category` fallback
- `bootstrap_category()` replaces `bootstrap_stakeholder_type()`
- Validate enums via `plan_policy`; 422 messages: `invalid plan_type`, `invalid category`

## 6. API handlers

**[`ml/rag/app/api.py`](ml-eng/ml/rag/app/api.py)**

- Remove from `QueryRequest`: `stakeholder_type`, `audience_instructions`, `geo_override`
- Wire `resolve_request_context` without legacy args; pass `plan_type`, `category`, `user_profile` into `run_rag()`
- `UsageStats.from_usage_dict` for response (3 fields only)

**[`ml/serving/chat/schemas.py`](ml-eng/ml/serving/chat/schemas.py)** + **[`app.py`](ml-eng/ml/serving/chat/app.py)**

- `SessionCreateRequest`: `category` instead of `stakeholder_type`
- `ChatRequest`: remove top-level `stakeholder_type`; bootstrap via `user_profile.category`
- `/v1/meta`: expose `plan_types` + `categories` catalogs (replace `stakeholder_types`)

## 7. Graph pipeline — [`ml/rag/chatbot/graph.py`](ml-eng/ml/rag/chatbot/graph.py)

- Add `RAGGraphState` keys: `plan_type`, `category`, `user_profile`
- `run_rag()`: accept `plan_type`, `category`; set `geo_override` via `effective_geo_override(plan_type, user_profile)`; stop passing old kwargs
- `node_decompose`: after `decompose_query`, call `apply_plan_decomposition_gates`
- **`node_generate`**: forward `plan_type`, `category` to `generate()` (currently missing — persona was not applied on main RAG path)
- `node_generate_meta` / `node_generate_product`: pass `category` instead of `stakeholder_type` / `audience_instructions`

## 8. Generator — [`ml/rag/chatbot/generator.py`](ml-eng/ml/rag/chatbot/generator.py)

- `_build_prompt`: combine `instruction_for_category(category)` + `plan_generation_addendum(plan_type)`
- Remove `audience_instructions` free-text path
- `generate()` kwargs: `category`, `plan_type` (replace `stakeholder_type` / `audience_instructions`)

## 9. Session memory — [`ml/rag/chat_turn.py`](ml-eng/ml/rag/chat_turn.py)

- `create_session(category)`; blob field `category` (replace `stakeholder_type`)
- `execute_chat_turn`: pass `plan_type`, `category`, `user_profile` to `run_rag`

## 10. Tests

| File | Changes |
|------|---------|
| [`test_request_context.py`](ml-eng/ml/rag/test_request_context.py) | New enums; reject old fields |
| [`test_geo_policy.py`](ml-eng/ml/rag/chatbot/test_geo_policy.py) | `plan_type: Farmers` |
| `test_plan_policy.py` (new) | Gates, category instructions, decomposition clamp |
| [`test_api_schemas.py`](ml-eng/ml/rag/test_api_schemas.py) | 3-field usage serialization |
| Run full `ml/rag` pytest suite |

## 11. Documentation

- Update [`ml/rag/docs/API.md`](ml-eng/ml/rag/docs/API.md) to final contract
- Short pointers in [`ml/rag/README.md`](ml-eng/ml/rag/README.md) and [`ml/serving/README.md`](ml-eng/ml/serving/README.md)

## Out of scope (backend-owned)

- Monthly query limits (5 / 50 / 200 / …)
- Payments, exports, API access, multi-seat
- Citation footnote fallback when model omits `[N]` (separate fix)

## Default when `user_profile` omitted

Minimal queries like `{"query":"Who are you?"}` remain valid: no plan gates, generic tone, no profile geo filter. Sending `user_profile` without valid `plan_type`/`category` → 422.
