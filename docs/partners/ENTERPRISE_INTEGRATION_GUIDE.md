# Ask ADZA Enterprise API — Integration Guide

This guide describes how B2B partners integrate with the Ask ADZA API.

## Base URL

Production and sandbox URLs are issued per tenant during onboarding.

## Authentication

Every request except `GET /health` and `GET /ready` must include a tenant API key:

```http
X-API-Key: <your-api-key>
```

Or:

```http
Authorization: Bearer <your-api-key>
```

Store API keys server-side only. Never embed keys in mobile apps or browser code.

## Primary endpoint (Agribusinesses tier)

```http
POST /query/agribusinesses
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "query": "Compare maize production in Nigeria and Ghana over the last five years",
  "session_id": "optional-for-multi-turn"
}
```

`POST /query` (no plan suffix) uses your tenant's plan.

### Response fields

| Field | Description |
|-------|-------------|
| `answer` | Natural-language answer with inline citation markers `[N]` |
| `citations[]` | Structured source list mapped to footnote numbers |
| `acf` | Confidence band, score, and explanation |
| `usage` | Token usage for this request |
| `artifacts[]` | Export files (CSV, chart, DOCX, PDF) when requested |
| `session_id` | Reuse for multi-turn conversations |
| `langfuse_trace_id` | Support reference for this request |

## Sessions

**Stateful (recommended for copilots):** omit `session_id` on the first request, then send the returned `session_id` on each follow-up. The server retains conversation memory.

**Stateless:** send `chat_history` with prior turns on every request. The server does not persist memory.

`DELETE /session/{session_id}` clears a session.

## Plan routes

Enterprise tenants are provisioned with a plan slug. Use the matching route:

| Plan slug | Route | Exports |
|-----------|-------|---------|
| `agribusinesses` | `POST /query/agribusinesses` | Yes |
| `integrated` | `POST /query/integrated` | Yes |

Integrated tenants may call any plan route.

## Error codes

| Status | Meaning |
|--------|---------|
| 401 | Missing or invalid API key |
| 403 | Tenant not authorized for this plan route |
| 422 | Invalid request body or profile |
| 429 | Rate limit or monthly quota exceeded |
| 500 | Internal error; quote the trace id from `detail` to support |

## OpenAPI specification

Machine-readable schema: [openapi/chatbot-v1.json](./openapi/chatbot-v1.json)

Regenerate from the codebase:

```bash
cd ml-eng
PYTHONPATH=. python scripts/export_chatbot_openapi.py
```

## Tenant provisioning

Copy [`ml-eng/config/enterprise_tenants.example.json`](../../ml-eng/config/enterprise_tenants.example.json) and fill `api_key_sha256` with the SHA-256 hex digest of each issued key (the plaintext key is never stored):

```bash
python -c "import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" <key>
```

Deploy the registry as `ENTERPRISE_TENANTS_JSON` (or `ENTERPRISE_TENANTS_PATH`) with:

```bash
ENTERPRISE_METERING_ENABLED=1
RAG_REDIS_URL=redis://...   # required for multi-replica metering consistency
```

## Support

Enterprise partners: contact@opentrace.africa
