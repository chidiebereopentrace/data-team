"""Enterprise B2B API support for Ask ADZA Chatbot API."""

from ml.serving.enterprise.auth import EnterpriseAuthMiddleware, extract_api_key, get_request_tenant
from ml.serving.enterprise.metering import (
    check_tenant_quotas,
    check_tenant_rate_limit,
    get_usage,
    record_usage,
    usage_to_dict,
)
from ml.serving.enterprise.tenant_registry import (
    EnterpriseTenant,
    enterprise_auth_mode,
    reload_tenants,
    resolve_tenant,
)

__all__ = [
    "EnterpriseAuthMiddleware",
    "EnterpriseTenant",
    "check_tenant_quotas",
    "check_tenant_rate_limit",
    "enterprise_auth_mode",
    "extract_api_key",
    "get_request_tenant",
    "get_usage",
    "record_usage",
    "reload_tenants",
    "resolve_tenant",
    "usage_to_dict",
]
