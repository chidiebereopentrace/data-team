#!/usr/bin/env python3
"""Generate OpenTrace API documentation as Word (.docx) files for the software team."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from api_docx_builder import (
    add_architecture_diagram,
    add_bullets,
    add_endpoint_section,
    add_field_table,
    add_heading,
    add_json_example,
    add_paragraph,
    add_title_page,
    add_toc_placeholder,
    new_document,
    save_document,
)

DOCS_DIR = Path(__file__).resolve().parents[1] / "ml" / "rag" / "docs"
RAG_DOCX = DOCS_DIR / "OpenTrace-RAG-API-Documentation.docx"

PLAN_TYPES = [
    ("Free", "Single country; no cross-country compare; brief answers"),
    ("Farmers", "Profile country geo filter; single country; plain-language framing"),
    ("Government", "National/sub-national + historical trends; no cross-country"),
    ("NGOs", "Government-tier depth + multi-region overlap framing"),
    ("Agribusinesses", "Cross-country comparison allowed; market/volatility framing; exports enabled"),
    ("Integrated", "No artificial plan caps beyond category persona; exports enabled"),
]

CATEGORIES = [
    ("Government", "Government & Public Institutions"),
    ("NGOs", "Foundations, NGOs & Development Partners"),
    ("Agribusinesses", "Agribusinesses & Financial Institutions"),
    ("Farmers", "Farmers, Cooperatives & Communities"),
]

SHARED_TYPES_USER_PROFILE = [
    ("country", "string | null", "No", "Retrieval geo filter only when plan_type is Farmers"),
    ("plan_type", "string", "Yes (if profile sent)", "One of: Free, Farmers, Government, NGOs, Agribusinesses, Integrated"),
    ("category", "string", "Yes (if profile sent)", "One of: Government, NGOs, Agribusinesses, Farmers"),
]

SHARED_TYPES_CHAT_MESSAGE = [
    ("role", "string", "Yes", '"user" or "assistant"'),
    ("content", "string", "Yes", "Non-empty message text"),
]

SHARED_TYPES_CITATION = [
    ("id", "integer", "Footnote number matching inline [N] in the answer"),
    ("kind", "string", "Normalized source type: academic, news, structured_data, policy, ota, web_wikipedia, web_search, ..."),
    ("text", "string", "Human-readable citation line"),
    ("url", "string | null", "Link when available; often null for structured data"),
]

SHARED_TYPES_USAGE = [
    ("input_tokens", "integer", "Input / prompt tokens for this request"),
    ("output_tokens", "integer", "Output / completion tokens"),
    ("total_tokens", "integer", "Total reported by LLM provider"),
]

SHARED_TYPES_ACF = [
    ("band", "string", "very_strong | strong | moderate | limited | low | no_evidence"),
    ("band_label", "string", "Human-readable band label"),
    ("score", "integer", "Composite confidence score 0–100"),
    ("explanation", "string", "One-sentence rationale"),
    ("note", "string | null", "Alias of explanation (backward compatible)"),
    ("components", "object | null", "Optional T/A/F/G component breakdown"),
    ("applied_ceiling", "string | null", "Ceiling safeguard applied, if any"),
    ("config_version", "string | null", "ACF config version used"),
    ("claim_level", "string | null", "national | sub_national | global"),
    ("question_type", "string | null", "time_sensitive | structural | ..."),
]


def _add_shared_types(doc) -> None:
    add_heading(doc, "Shared types", level=1)
    add_heading(doc, "UserProfile", level=2)
    add_paragraph(doc, "When user_profile is sent, plan_type and category are required. Unknown keys → 422.")
    add_field_table(doc, ("Field", "Type", "Required", "Description"), SHARED_TYPES_USER_PROFILE)

    add_heading(doc, "ChatMessage", level=2)
    add_field_table(doc, ("Field", "Type", "Required", "Description"), SHARED_TYPES_CHAT_MESSAGE)

    add_heading(doc, "CitationItem", level=2)
    add_field_table(doc, ("Field", "Type", "Description"), SHARED_TYPES_CITATION)

    add_heading(doc, "UsageStats", level=2)
    add_field_table(doc, ("Field", "Type", "Description"), SHARED_TYPES_USAGE)

    add_heading(doc, "ACFSignal", level=2)
    add_paragraph(doc, "ADZA Confidence Framework (Path B) attached to every successful response.")
    add_field_table(doc, ("Field", "Type", "Description"), SHARED_TYPES_ACF)


def _add_plan_types_and_categories(doc) -> None:
    add_heading(doc, "Plan types", level=1)
    add_field_table(doc, ("ID", "Retrieval / generation gates"), PLAN_TYPES)
    add_paragraph(doc, "Invalid plan_type values return HTTP 422.")

    add_heading(doc, "Categories", level=1)
    add_field_table(doc, ("ID", "Label"), CATEGORIES)
    add_paragraph(doc, "Invalid category values return HTTP 422.")
    add_paragraph(doc, "Resolution order: user_profile.plan_type + user_profile.category → session category fallback.")


def _add_conversation_memory(doc) -> None:
    add_heading(doc, "Conversation memory", level=1)
    add_heading(doc, "Server-side session (stateful)", level=2)
    add_bullets(
        doc,
        [
            "Omit chat_history on each request.",
            "Reuse session_id from the previous response.",
            "Server stores {conversation_summary, recent_turns} in Redis (RAG_REDIS_URL) or in-process fallback.",
            "Session TTL default: 604800 seconds (7 days).",
        ],
    )
    add_heading(doc, "Client-owned history (stateless)", level=2)
    add_bullets(
        doc,
        [
            "Send chat_history with prior turns.",
            "Server compacts history for that request only.",
            "Server session is NOT updated when chat_history is present.",
        ],
    )
    add_paragraph(doc, "conversation_history is a deprecated alias for chat_history.")


def build_rag_document() -> None:
    doc = new_document()
    add_title_page(
        doc,
        title="OpenTrace RAG API",
        subtitle="Internal API Reference — Ask ADZA Backend & Pipeline Tuning",
        version="0.1.0",
        generated=date.today(),
    )
    add_toc_placeholder(doc)

    add_heading(doc, "Overview", level=1)
    add_bullets(
        doc,
        [
            "Entrypoint: uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860",
            "Default local base URL: http://localhost:7860",
            "OpenAPI: GET /docs and GET /openapi.json",
            "Implementation: ml/rag/app/api.py",
        ],
    )

    add_heading(doc, "Authentication & security", level=1)
    add_bullets(
        doc,
        [
            "Every route except /, /health and /ready requires X-API-Key or Authorization: Bearer.",
            "Tenants (ENTERPRISE_TENANTS_JSON) store only the SHA-256 of each key, a plan_slug, "
            "quotas, rate_limit_rpm and allow_trace.",
            "A tenant may call /query/{its plan} only (integrated tenants: any plan); bare /query "
            "uses the tenant's plan. Wrong plan route: HTTP 403. Missing/invalid key: HTTP 401.",
            "include_trace is honoured only for tenants with allow_trace=true.",
            "Per-IP rate limits per plan (FREE_RATE_LIMIT_RPM etc.) default to 10-60 rpm; HTTP 429.",
            "CORS allows no origins unless RAG_CORS_ORIGINS lists them; credentials are disabled.",
            "RAG_API_AUTH=off disables auth for local development only.",
        ],
    )

    add_architecture_diagram(doc)
    _add_shared_types(doc)
    _add_plan_types_and_categories(doc)
    _add_conversation_memory(doc)

    add_heading(doc, "Endpoints", level=1)

    add_endpoint_section(
        doc,
        method="GET",
        path="/",
        description="Service index.",
        response_json='{\n  "message": "OpenTrace RAG API",\n  "docs": "/docs",\n  "health": "/health",\n  "query": "POST /query"\n}',
    )

    add_endpoint_section(
        doc,
        method="GET",
        path="/health",
        description="Liveness probe. Always fast; does not check Qdrant or LLM.",
        response_json='{\n  "status": "ok",\n  "service": "rag"\n}',
    )

    add_endpoint_section(
        doc,
        method="GET",
        path="/ready",
        description="Readiness probe for load balancers.",
        response_fields=[
            ("status", "string", '"ready" if Qdrant + LLM env present; else "not_ready"'),
            ("service", "string", '"rag"'),
            ("missing_config_keys", "string[]", 'e.g. QDRANT_URL, RAG_LLM_BASE_URL'),
            ("redis", "object | null", "Present when RAG_REDIS_URL is set; informational"),
        ],
        response_json=(
            '{\n  "status": "ready",\n  "service": "rag",\n  "missing_config_keys": [],\n'
            '  "redis": { "backend": "redis", "connected": true }\n}'
        ),
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/query",
        description="Main RAG endpoint. Runs decomposition → BigQuery + vector retrieval → rerank → generation.",
        request_fields=[
            ("query", "string", "Yes", "Natural-language question (min length 1)"),
            ("session_id", "string | null", "No", "Reuse for server-side multi-turn memory; new UUID if omitted"),
            ("user_id", "string | null", "No", "Product user id for Langfuse analytics"),
            ("user_profile", "UserProfile | null", "No", "country, plan_type, category when sent"),
            ("chat_history", "ChatMessage[] | null", "No", "Prior turns (canonical); skips server session update"),
            ("conversation_history", "ChatMessage[] | null", "No", "Deprecated alias for chat_history"),
            ("include_trace", "boolean", "No", "Include decomposition and retrieval counts (default false)"),
            ("time_start_override", "string | null", "No", "ISO date filter override for retrieval"),
            ("time_end_override", "string | null", "No", "ISO date filter override"),
            ("news_top_k", "integer | null", "No", "Max news chunks"),
            ("academic_top_k", "integer | null", "No", "Max academic chunks"),
            ("bq_top_k", "integer | null", "No", "Max BigQuery result rows/chunks"),
            ("rerank_top_k", "integer | null", "No", "Chunks after reranking"),
            ("ota_top_k", "integer | null", "No", "Max OTA insight chunks"),
        ],
        response_fields=[
            ("answer", "string", "Prose answer with inline footnotes [N]"),
            ("citations", "CitationItem[]", "Structured sources for UI rendering"),
            ("acf", "ACFSignal", "Confidence band, score, explanation"),
            ("session_id", "string", "Pass on next request for continuity"),
            ("usage", "UsageStats", "Per-request LLM token totals"),
            ("error", "string | null", "Pipeline error if graph set one; may still return partial answer"),
            ("trace", "object | null", "Present only when include_trace is true"),
            ("langfuse_trace_id", "string | null", "Langfuse trace id for feedback / debugging"),
        ],
        curl_example=(
            'curl -X POST http://localhost:7860/query \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"query": "What are rice yield trends in Nigeria?", '
            '"user_profile": {"plan_type": "Government", "category": "Government"}}\''
        ),
        request_json=(
            '{\n  "query": "What are rice yield trends in Nigeria?",\n'
            '  "session_id": "abc123...",\n'
            '  "user_profile": {\n    "country": "Ghana",\n'
            '    "plan_type": "Farmers",\n    "category": "Farmers"\n  },\n'
            '  "include_trace": false\n}'
        ),
        response_json=(
            '{\n  "answer": "Nigeria\'s rice production has trended upward...[1]",\n'
            '  "citations": [{"id": 1, "kind": "news", "text": "[News] ...", "url": "https://..."}],\n'
            '  "acf": {"band": "strong", "band_label": "Strong confidence", "score": 78, "explanation": "..."},\n'
            '  "session_id": "787a6c22...",\n'
            '  "usage": {"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600},\n'
            '  "error": null,\n  "langfuse_trace_id": "trace-abc..."\n}'
        ),
        errors=[
            ("422", "Invalid plan_type/category; unknown request fields"),
            ("500", "Unhandled exception; traceback if RAG_DEBUG=1"),
        ],
    )

    add_heading(doc, "include_trace payload", level=2)
    add_paragraph(doc, "When include_trace is true, the trace object contains:")
    add_field_table(
        doc,
        ("Field", "Type", "Description"),
        [
            ("decomposition", "object", "Query decomposer output"),
            ("bq_table_candidates_count", "integer", "BQ table candidates"),
            ("vector_news_count", "integer", "News retrieval count"),
            ("vector_academic_papers_count", "integer", "Academic papers count"),
            ("vector_policies_count", "integer", "Policy chunks count"),
            ("vector_public_reports_count", "integer", "Public reports count"),
            ("vector_formation_count", "integer", "Formation chunks count"),
            ("vector_academic_count", "integer", "Deprecated academic alias count"),
            ("vector_ota_count", "integer", "OTA insights count"),
            ("merged_context_count", "integer", "Merged context items"),
            ("reranked_context_count", "integer", "Reranked context items"),
            ("langfuse_trace_id", "string | null", "Langfuse trace id"),
        ],
    )

    add_endpoint_section(
        doc,
        method="DELETE",
        path="/session/{session_id}",
        description="Delete a session and its conversation memory. Call when user starts a new conversation or logs out.",
        response_json='{\n  "status": "deleted",\n  "session_id": "abc123..."\n}',
        errors=[("400", "Empty session_id")],
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/feedback",
        description="Record user feedback (thumbs up/down) on a Langfuse trace.",
        request_fields=[
            ("trace_id", "string", "Yes", "Langfuse trace id from response"),
            ("score", "float", "Yes", "1.0 = thumbs up, 0.0 = thumbs down"),
            ("comment", "string | null", "No", "Optional comment (max 500 chars)"),
        ],
        response_json='{\n  "status": "ok",\n  "trace_id": "trace-abc..."\n}',
        errors=[("503", "Langfuse not configured or invalid trace id")],
    )

    add_heading(doc, "Integration guide", level=1)
    add_bullets(
        doc,
        [
            "Call POST /query from your backend service.",
            "Store usage and your auth user_id for billing analytics.",
            "Render answer prose and citations[] separately in the UI.",
            "Link inline [N] footnotes to citation cards.",
            "Reuse session_id OR send chat_history each turn — pick one pattern, not both.",
            "Use langfuse_trace_id with POST /feedback for thumbs up/down.",
        ],
    )
    add_json_example(
        doc,
        "Multi-turn with server session",
        '// Turn 1\n{"query": "Rice trends in Nigeria?", '
        '"user_profile": {"plan_type": "Government", "category": "Government"}}\n\n'
        '// Turn 2 — reuse session_id from Turn 1 response\n'
        '{"query": "What about maize?", "session_id": "<from-turn-1>"}',
    )

    add_heading(doc, "Environment variables", level=1)
    add_field_table(
        doc,
        ("Variable", "Description"),
        [
            ("QDRANT_URL", "Qdrant cluster URL"),
            ("QDRANT_API_KEY", "Qdrant API key"),
            ("RAG_LLM_BASE_URL", "LLM provider base URL"),
            ("RAG_LLM_API_KEY", "LLM provider API key"),
            ("RAG_LLM_MODEL_ID", "Default generation model"),
            ("BQ_PROJECT", "BigQuery project id"),
            ("BQ_DATASET_SILVER", "Staging dataset (default staging_dev)"),
            ("RAG_REDIS_URL", "Redis URL for session storage"),
            ("RAG_CORS_ORIGINS", "Comma-separated CORS origins (default: none allowed)"),
            ("RAG_API_AUTH", "required (default) or off (local development only)"),
            ("ENTERPRISE_TENANTS_JSON", "Tenant registry JSON (api_key_sha256, plan_slug, quotas)"),
            ("RAG_CITATIONS_MODE", "referenced (default) or all"),
            ("RAG_DEBUG", "1 for verbose 500 error tracebacks"),
            ("LANGFUSE_*", "Langfuse tracing and feedback"),
        ],
    )

    save_document(doc, str(RAG_DOCX))
    print(f"Wrote {RAG_DOCX}")


def main() -> None:
    argparse.ArgumentParser(description="Generate OpenTrace RAG API DOCX documentation").parse_args()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    build_rag_document()


if __name__ == "__main__":
    main()
