# Ask ADZA Enterprise API — Integration Guide

This guide describes how B2B partners integrate with the Ask ADZA Enterprise Chatbot API v1.

## Base URL

Production and sandbox URLs are issued per tenant during onboarding.

## Authentication

When `CHATBOT_ENTERPRISE_AUTH=required`, every request (except `/v1/health` and `/v1/meta`) must include a tenant API key:

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
POST /v1/chat/agribusinesses
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "message": "Compare maize production in Nigeria and Ghana over the last five years",
  "session_id": "optional-for-multi-turn"
}
```

### Response fields

| Field | Description |
|-------|-------------|
| `assistant_message` | Natural-language answer with inline citation markers `[N]` |
| `citations[]` | Structured source list mapped to footnote numbers |
| `acf` | Confidence band, score, and explanation |
| `usage` | Token usage for this request |
| `artifacts[]` | Export files (CSV, chart, DOCX, PDF) when requested |
| `session_id` | Reuse for multi-turn conversations |
| `request_id` | Support reference for this request |

## Sessions

**Stateful (recommended for copilots):**

1. `POST /v1/sessions` with `{ "category": "Agribusinesses" }`
2. Send `session_id` on each subsequent chat request
3. Omit `chat_history` — the server retains conversation memory

**Stateless:**

Send `chat_history` with prior turns on every request. The server does not persist memory.

## Usage and quotas

Check current-month consumption:

```http
GET /v1/usage
X-API-Key: <your-api-key>
```

Returns query count, token totals, export count, and configured quotas.

## Plan routes

Enterprise tenants are provisioned with a plan slug. Use the matching route:

| Plan slug | Route | Exports |
|-----------|-------|---------|
| `agribusinesses` | `POST /v1/chat/agribusinesses` | Yes |
| `integrated` | `POST /v1/chat/integrated` | Yes |

Integrated tenants may call any plan route.

## Error codes

| Status | Meaning |
|--------|---------|
| 401 | Missing or invalid API key |
| 403 | Tenant not authorized for this plan route |
| 422 | Invalid request body or profile |
| 429 | Rate limit or monthly quota exceeded |
| 502 | RAG pipeline error (retry may succeed) |

## OpenAPI specification

Machine-readable schema: [openapi/chatbot-v1.json](./openapi/chatbot-v1.json)

Regenerate from the codebase:

```bash
cd ml-eng
PYTHONPATH=. python scripts/export_chatbot_openapi.py
```

## Tenant provisioning

Copy [`ml-eng/config/enterprise_tenants.example.json`](../../ml-eng/config/enterprise_tenants.example.json) to `enterprise_tenants.json` (not committed) and set:

```bash
ENTERPRISE_TENANTS_PATH=ml-eng/config/enterprise_tenants.json
CHATBOT_ENTERPRISE_AUTH=required
ENTERPRISE_METERING_ENABLED=1
RAG_REDIS_URL=redis://...   # required for multi-replica metering consistency
```

## Support

Enterprise partners: contact@opentrace.africa
