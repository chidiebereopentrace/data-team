# Serving — API access control

`ml/serving/enterprise/` holds API-key authentication, tenant quotas and usage metering for the deployed RAG API [`ml/rag/app/api.py`](../rag/app/api.py) (served as `ml.rag.api:app`). Endpoint reference: [`ml/rag/docs/API.md`](../rag/docs/API.md).

| Module | Purpose |
|--------|---------|
| `enterprise/tenant_registry.py` | Loads tenants from `ENTERPRISE_TENANTS_JSON` / `ENTERPRISE_TENANTS_PATH`; keys are stored as SHA-256 digests only |
| `enterprise/auth.py` | `EnterpriseAuthMiddleware`: 401 without a valid key, 403 on another tenant's `/query/{plan}` route, monthly quotas and per-tenant rpm |
| `enterprise/metering.py` | Per-tenant monthly query/token/export counters (Redis when `RAG_REDIS_URL` is set) |

Public routes: `/`, `/health`, `/ready`. Everything else (including `/docs`) needs `X-API-Key` or `Authorization: Bearer`.

Tenant fields: `tenant_id`, `name`, `api_key_sha256`, `plan_slug`, `environment`, `monthly_query_quota`, `monthly_token_quota`, `rate_limit_rpm`, `allow_trace`, `active`. See [`config/enterprise_tenants.example.json`](../../config/enterprise_tenants.example.json).

Local development without keys: `RAG_API_AUTH=off`.
