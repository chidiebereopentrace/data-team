# Ask ADZA Enterprise API — Integration Guide

This guide is the handshake a caller completes with the Ask ADZA API. The live routes are `POST /query` and `POST /query/{plan}`.

## Base URL

Production and sandbox URLs are issued per tenant during onboarding.

AskADZA's own app is the `web-frontend` tenant (`plan_slug: integrated` in `ml-eng/config/enterprise_tenants.example.json`). It uses this same handshake. There is no separate product route.

## Authentication

Public unauthenticated paths are `GET /`, `GET /health`, `GET /ready`, `GET /docs`, `GET /redoc`, and `GET /openapi.json`. Every other request must include a tenant API key:

```http
X-API-Key: <your-api-key>
```

Or:

```http
Authorization: Bearer <your-api-key>
```

An unknown or missing key is HTTP 401:

```json
{"detail":"Missing or invalid API key"}
```

Store API keys server-side only. Never embed keys in mobile apps or browser code.

The AskADZA app adds `user_id` for the signed-in account on every query. That id is what the per-user limit counts. The browser never holds the API key.

## Entitlement

The path plan must match the tenant `plan_slug`. A tenant with `plan_slug: integrated` may call every plan route. Anything else is HTTP 403.

`user_profile.plan_type` does not grant access. `category` is the persona: `Government`, `NGOs`, `Agribusinesses`, `Farmers`, or `Research`. `country` is kept only as an African ISO3 code.

## AskADZA web app

The product tenant is `web-frontend`. Send every query to `POST /query/integrated` or `POST /query`. The app mints `session_id` once per conversation and reuses that id. The server loads and saves the thread in Redis under that id. Do not send `chat_history` or `conversation_history`. A missing `user_id`, `session_id`, `user_profile.plan_type`, or `user_profile.category` is HTTP 422.

```http
POST /query/integrated
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "query": "Compare maize production in Nigeria and Ghana over the last five years",
  "user_id": "account-id",
  "session_id": "conversation-id-from-the-app",
  "user_profile": {
    "plan_type": "Integrated",
    "category": "Farmers",
    "country": "NGA"
  }
}
```

`country` may be omitted. `plan_type` and `category` may not.

## Partner endpoint

A partner tenant may omit `session_id` and `user_profile`. `POST /query` with no plan suffix uses that tenant's own plan.

```http
POST /query/agribusinesses
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "query": "Compare maize production in Nigeria and Ghana over the last five years",
  "user_id": "account-id"
}
```

### Plan routes

| Plan slug | Route | Exports |
|-----------|-------|---------|
| `free` | `POST /query/free` | No |
| `farmers` | `POST /query/farmers` | No |
| `government` | `POST /query/government` | No |
| `ngos` | `POST /query/ngos` | No |
| `agribusinesses` | `POST /query/agribusinesses` | Yes |
| `integrated` | `POST /query/integrated` | Yes |
| `research` | `POST /query/research` | Yes — CSV, GeoJSON and shared report formats |

### Export intent

Send `export_intent` when a machine client already knows the file it wants. Omit it for a person, and ask in the question (for example, "Export maize prices as a CSV"). When both are present, the field wins.

```json
{"query": "Maize prices in Lagos", "export_intent": "csv"}
```

Allowed values: `csv`, `geojson`, `chart`, `docx`, `pdf`, `multi`. Any other value is HTTP 422. Free, Farmers, Government, and NGOs still receive no files.

### Response fields

| Field | Description |
|-------|-------------|
| `answer` | Natural-language answer |
| `citations` | Structured source list |
| `acf` | Confidence band, score, and explanation |
| `usage` | Token usage for this request |
| `session_id` | Conversation id. The web app sends it. A partner that omits it receives a new hex id |
| `session_found` | True when a stored session was loaded |
| `session_ttl_seconds` | Configured session TTL. Default `604800` |
| `langfuse_trace_id` | Support reference when tracing is enabled |
| `artifacts` | Export files when an export ran |

`include_trace` returns `trace` only when the tenant has `allow_trace: true`. Otherwise the field is omitted.

## Sessions

Sessions are stored under `tenant:{tenant_id}`.

The AskADZA app mints `session_id` per conversation and sends it on every query. The server fills the thread from Redis and writes this turn back to the same key. That tenant must not send `chat_history`.

A partner may omit `session_id` on the first call and reuse the hex id the response returns. If that partner sends `chat_history`, the server session is neither read nor written.

```http
DELETE /session/{session_id}
```

returns:

```json
{"status":"deleted","session_id":"..."}
```

## Rate limit and capacity

`rate_limit_rpm` is requests per minute for one `user_id` of that tenant. `0` disables it. On the product tenant the example value `120` is 120 requests per minute per signed-in account, not a cap on the whole app. The web app sends `user_id` on every query. A partner call with no `user_id` shares one anonymous bucket for that tenant. Monthly query and token quotas stay tenant-wide. The product tenant keeps `monthly_query_quota: 0`.

A rate-limit HTTP 429 carries `Retry-After` and `X-RateLimit-Limit` headers. A capacity failure is HTTP 503. With authentication off there is no tenant, so local calls are not rate limited.

## Research & Academia

The canonical API value is `Research` for both `user_profile.plan_type` and `user_profile.category`. Display it as **Research & Academia**. A Research tenant can use `/query/research` or `/query`; the authenticated route owns the plan tier. Free and Integrated users can select `"category": "Research"` without changing their subscription permissions. The perspective emphasizes methods, provenance, comparability and limits to inference.

```json
{
  "query": "Compare maize yields in Kenya and Tanzania since 2020, discuss comparability, and export the observations as CSV and GeoJSON",
  "user_profile": {"plan_type": "Research", "category": "Research"}
}
```

Send that body to `POST /query/research` with a Research or Integrated tenant key. Inspect `artifacts` for the download URLs. GeoJSON artifacts use the `geojson` kind, `.geojson` extension and `application/geo+json` MIME type. Verified WGS84 geometries or longitude/latitude pairs become feature geometries. Observations without coordinates remain features with `geometry: null`; country names are not geocoded into invented locations or boundaries. Units, periods and source attributes present in the retrieved observations are retained in properties.

Provision Research accounts with `plan_slug: "research"` in the server-side tenant registry. Category selection alone does not authorize paid exports. The same export storage settings and URL refresh endpoint apply to Research. For deployed downloads, configure the existing S3 or GCS artifact storage; the local `file://` development storage is only accessible on the server machine.

## Errors

Error bodies stay `{"detail":"..."}`.

| Status | Meaning |
|--------|---------|
| 401 | Missing or invalid API key |
| 403 | Tenant not authorized for this plan route |
| 422 | Invalid request body or profile |
| 429 | Per-user rate limit or tenant monthly quota exceeded |
| 503 | Process is at capacity, or a readiness check failed |
| 500 | Internal error |

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
RAG_REDIS_URL=redis://...   # required so two replicas share one per-user counter
```

## Support

Enterprise partners: contact@opentrace.africa
