#!/usr/bin/env python3
"""Generate OpenTrace API documentation as Word (.docx) files for the software team."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from api_docx_builder import (
    add_architecture_diagram,
    add_bullets,
    add_comparison_table,
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
CHAT_DOCX = DOCS_DIR / "OpenTrace-Chatbot-API-v1-Documentation.docx"

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
            "No built-in API authentication. Endpoints are open if the URL is known.",
            "Configure CORS via RAG_CORS_ORIGINS (default: *).",
            "Recommend API gateway auth for production deployments.",
            "user_id is client-supplied for Langfuse analytics until auth is implemented.",
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
            ("RAG_CORS_ORIGINS", "Comma-separated CORS origins (default *)"),
            ("RAG_CITATIONS_MODE", "referenced (default) or all"),
            ("RAG_DEBUG", "1 for verbose 500 error tracebacks"),
            ("LANGFUSE_*", "Langfuse tracing and feedback"),
        ],
    )

    save_document(doc, str(RAG_DOCX))
    print(f"Wrote {RAG_DOCX}")


def build_chatbot_document() -> None:
    doc = new_document()
    add_title_page(
        doc,
        title="OpenTrace Chatbot API v1",
        subtitle="Public Integration Reference — Product Frontend & External Clients",
        version="1.0 (schema_version 2)",
        generated=date.today(),
    )
    add_toc_placeholder(doc)

    add_heading(doc, "Overview", level=1)
    add_bullets(
        doc,
        [
            "Entrypoint: uvicorn ml.serving.chat.app:app --host 0.0.0.0 --port 7861",
            "Default local base URL: http://localhost:7861",
            "OpenAPI: GET /docs and GET /openapi.json",
            "Implementation: ml/serving/chat/app.py",
            "All chat routes call execute_chat_turn() → run_rag().",
        ],
    )

    add_heading(doc, "Authentication & rate limiting", level=1)
    add_bullets(
        doc,
        [
            "No built-in API authentication.",
            "Optional per-plan rate limits on POST /v1/chat/{plan} routes.",
            "Configure via FREE_RATE_LIMIT_RPM, FARMERS_RATE_LIMIT_RPM, etc. (0 = disabled).",
            "When exceeded: HTTP 429.",
            "GET /v1/meta exposes current rate_limits_rpm configuration.",
            "Auth/subscription enforcement is expected upstream at the API gateway.",
        ],
    )

    add_architecture_diagram(doc)
    _add_shared_types(doc)

    add_heading(doc, "ArtifactItem (export routes only)", level=2)
    add_field_table(
        doc,
        ("Field", "Type", "Description"),
        [
            ("id", "string", "Stable artifact id for this response"),
            ("kind", "csv | chart | docx | pdf | html", "Export format"),
            ("filename", "string", "Suggested download filename"),
            ("mime_type", "string", "MIME type"),
            ("url", "string", "Signed HTTPS URL (GCS) or local file URI in dev"),
            ("summary", "string", "Short description of contents"),
            ("citation_ids", "integer[]", "Citation ids from parent answer"),
            ("byte_size", "integer", "File size in bytes"),
        ],
    )

    _add_plan_types_and_categories(doc)

    add_heading(doc, "Plan-scoped chat routes", level=1)
    add_field_table(
        doc,
        ("Route", "Plan", "Exports (artifacts)"),
        [
            ("POST /v1/chat/free", "Free", "No"),
            ("POST /v1/chat/farmers", "Farmers", "No"),
            ("POST /v1/chat/government", "Government", "No"),
            ("POST /v1/chat/ngos", "NGOs", "No"),
            ("POST /v1/chat/agribusinesses", "Agribusinesses", "Yes (CSV, chart, DOCX, PDF)"),
            ("POST /v1/chat/integrated", "Integrated", "Yes (CSV, chart, DOCX, PDF)"),
        ],
    )
    add_paragraph(doc, "Plan tier is locked by the URL; the request payload cannot override it.")

    add_heading(doc, "Session lifecycle", level=1)
    add_bullets(
        doc,
        [
            "POST /v1/sessions — create session with fixed category",
            "POST /v1/sessions/{plan_slug} — create session scoped to plan (free, farmers, government, ngos, agribusinesses, integrated)",
            "GET /v1/sessions/{session_id} — check alive status (404 if expired)",
            "DELETE /v1/sessions/{session_id} — clear session (idempotent)",
        ],
    )
    _add_conversation_memory(doc)

    add_heading(doc, "Endpoints", level=1)

    add_endpoint_section(
        doc,
        method="GET",
        path="/v1/health",
        description="Liveness probe.",
        response_json='{\n  "status": "ok",\n  "service": "chatbot"\n}',
    )

    add_endpoint_section(
        doc,
        method="GET",
        path="/v1/meta",
        description="API metadata: plan types, categories, route map, rate limits.",
        response_json=(
            '{\n  "api_version": "1.0",\n  "schema_version": "2",\n  "build": null,\n'
            '  "plan_types": ["Free", "Farmers", ...],\n  "categories": ["Government", ...],\n'
            '  "plan_routes": {"free": "/v1/chat/free", ...},\n'
            '  "rate_limits_rpm": {"free": 0, "farmers": 0, ...}\n}'
        ),
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/v1/sessions",
        description="Create a server-side session with a fixed category persona.",
        request_fields=[("category", "Category", "Yes", "Government | NGOs | Agribusinesses | Farmers")],
        response_fields=[
            ("session_id", "string", "New session id"),
            ("created_at", "string", "ISO-8601 UTC timestamp"),
            ("category", "string", "Session category"),
        ],
        request_json='{\n  "category": "Government"\n}',
        response_json='{\n  "session_id": "a1b2c3...",\n  "created_at": "2026-06-08T12:00:00+00:00",\n  "category": "Government"\n}',
        errors=[("422", "Invalid category")],
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/v1/sessions/{plan_type_slug}",
        description="Create a session scoped to a plan type from URL slug.",
        response_json='{\n  "session_id": "a1b2c3...",\n  "created_at": "2026-06-08T12:00:00+00:00",\n  "category": "Government"\n}',
        errors=[("404", "Unknown plan slug")],
    )

    add_endpoint_section(
        doc,
        method="GET",
        path="/v1/sessions/{session_id}",
        description="Check if a session is alive.",
        response_fields=[
            ("session_id", "string", "Session id"),
            ("alive", "boolean", "Always true when found"),
            ("category", "string | null", "Session category"),
            ("turn_count", "integer", "Number of recent turns stored"),
            ("has_summary", "boolean", "Whether conversation summary exists"),
        ],
        errors=[("404", "Session not found or expired")],
    )

    add_endpoint_section(
        doc,
        method="DELETE",
        path="/v1/sessions/{session_id}",
        description="Explicitly clear a session (logout / clear chat). Idempotent.",
        response_json='{\n  "session_id": "abc123...",\n  "deleted": true\n}',
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/v1/chat",
        description="Single chat turn through the RAG pipeline. Prefer plan-scoped routes for new integrations.",
        request_fields=[
            ("query", "string | null", "One of query/message", "User message"),
            ("message", "string | null", "One of query/message", "Alias for query — do not send both"),
            ("session_id", "string | null", "No", "From POST /v1/sessions or prior chat response"),
            ("user_id", "string | null", "No", "Product user id for Langfuse analytics"),
            ("user_profile", "UserProfile | null", "No", "country, plan_type, category"),
            ("chat_history", "ChatMessage[] | null", "No", "Client-owned prior turns"),
            ("conversation_history", "ChatMessage[] | null", "No", "Deprecated alias for chat_history"),
        ],
        response_fields=[
            ("assistant_message", "string", "Same content as answer on POST /query"),
            ("citations", "CitationItem[]", "Structured sources"),
            ("acf", "ACFSignal", "Confidence signal"),
            ("session_id", "string", "Session for next turn"),
            ("usage", "UsageStats", "LLM token totals"),
            ("request_id", "string", "Unique HTTP request id"),
            ("created_at", "string", "ISO-8601 UTC timestamp"),
            ("plan_type", "string | null", "Plan tier applied (plan-scoped routes)"),
            ("langfuse_trace_id", "string | null", "Langfuse trace id"),
            ("artifacts", "ArtifactItem[]", "Only on agribusinesses/integrated routes"),
        ],
        curl_example=(
            'curl -X POST http://localhost:7861/v1/chat/government \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"query": "What are rice yield trends?", "session_id": "abc123..."}\''
        ),
        request_json=(
            '{\n  "query": "What are rice yield trends?",\n'
            '  "session_id": "abc123...",\n'
            '  "user_profile": {\n    "country": "Ghana",\n'
            '    "plan_type": "Farmers",\n    "category": "Farmers"\n  }\n}'
        ),
        response_json=(
            '{\n  "assistant_message": "Maize production rose...[3]",\n'
            '  "citations": [{"id": 3, "kind": "structured_data", "text": "...", "url": null}],\n'
            '  "acf": {"band": "strong", "band_label": "Strong confidence", "score": 78, "explanation": "..."},\n'
            '  "session_id": "abc123...",\n'
            '  "usage": {"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600},\n'
            '  "request_id": "f4e2...",\n  "created_at": "2026-06-08T12:00:00+00:00",\n'
            '  "plan_type": "Agribusinesses",\n  "artifacts": []\n}'
        ),
        errors=[
            ("422", "Missing query/message, both sent, empty chat_history, validation errors"),
            ("429", "Rate limit exceeded (plan-scoped routes when RPM > 0)"),
            ("502", "RAG pipeline error — see error.code rag_pipeline_error"),
            ("500", "Unhandled; traceback if CHATBOT_DEBUG=1"),
        ],
    )

    add_heading(doc, "Export response example", level=2)
    add_json_example(
        doc,
        "Agribusinesses route with artifacts",
        '{\n  "assistant_message": "Maize production rose...\\n\\nDownloadable files: nigeria_maize.csv",\n'
        '  "artifacts": [{\n    "id": "art_a1b2c3",\n    "kind": "csv",\n'
        '    "filename": "nigeria_maize.csv",\n    "mime_type": "text/csv",\n'
        '    "url": "https://storage.googleapis.com/...",\n'
        '    "summary": "CSV export (24 rows)",\n    "citation_ids": [3],\n    "byte_size": 12480\n  }]\n}',
    )

    add_comparison_table(
        doc,
        "POST /query vs POST /v1/chat",
        ("Feature", "POST /query (RAG API)", "POST /v1/chat (Chatbot v1)"),
        [
            ("Answer field", "answer", "assistant_message"),
            ("Retrieval tuning", "news_top_k, bq_top_k, etc.", "Not exposed"),
            ("Debug trace", "include_trace", "Not exposed"),
            ("Session create", "Implicit UUID", "POST /v1/sessions or auto on first chat"),
            ("Pipeline error", "error string in 200 body", "HTTP 502 JSON"),
            ("Exports", "Not available", "artifacts[] on agribusinesses/integrated"),
            ("Feedback", "POST /feedback", "Not exposed"),
            ("Extra fields", "trace, langfuse_trace_id", "request_id, created_at, plan_type"),
        ],
    )

    add_heading(doc, "Recommended integration flow", level=1)
    add_bullets(
        doc,
        [
            "1. POST /v1/sessions/{plan_slug} or POST /v1/sessions to obtain session_id.",
            "2. POST /v1/chat/{plan_slug} with query and session_id.",
            "3. Render assistant_message + citations[]; link [N] footnotes to citation cards.",
            "4. Display acf.band_label and acf.score in the UI confidence indicator.",
            "5. On agribusinesses/integrated: render artifacts[] download links.",
            "6. DELETE /v1/sessions/{session_id} on logout or new conversation.",
        ],
    )

    add_heading(doc, "Environment variables", level=1)
    add_field_table(
        doc,
        ("Variable", "Description"),
        [
            ("CHATBOT_CORS_ORIGINS", "Comma-separated CORS origins (default *)"),
            ("CHATBOT_DEBUG", "1 for verbose 500 error tracebacks"),
            ("CHATBOT_BUILD_ID", "Build id surfaced in GET /v1/meta"),
            ("FREE_RATE_LIMIT_RPM", "Requests per minute for /v1/chat/free (0 = off)"),
            ("FARMERS_RATE_LIMIT_RPM", "Rate limit for farmers route"),
            ("GOVERNMENT_RATE_LIMIT_RPM", "Rate limit for government route"),
            ("NGOS_RATE_LIMIT_RPM", "Rate limit for ngos route"),
            ("AGRI_RATE_LIMIT_RPM", "Rate limit for agribusinesses route"),
            ("INTEGRATED_RATE_LIMIT_RPM", "Rate limit for integrated route"),
            ("RAG_REDIS_URL", "Shared session storage (same as RAG API)"),
            ("QDRANT_*, RAG_LLM_*, BQ_*", "Shared RAG pipeline dependencies — see RAG API doc"),
        ],
    )

    save_document(doc, str(CHAT_DOCX))
    print(f"Wrote {CHAT_DOCX}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenTrace API DOCX documentation")
    parser.add_argument("--rag-only", action="store_true", help="Generate RAG API doc only")
    parser.add_argument("--chat-only", action="store_true", help="Generate Chatbot v1 doc only")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if args.chat_only:
        build_chatbot_document()
    elif args.rag_only:
        build_rag_document()
    else:
        build_rag_document()
        build_chatbot_document()


if __name__ == "__main__":
    main()
