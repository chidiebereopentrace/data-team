#!/usr/bin/env python3
"""Generate one simple Ask ADZA API Word doc for the software team (production RAG)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from api_docx_builder import (
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
OUTPUT = DOCS_DIR / "OpenTrace-Ask-ADZA-API-Software-Team.docx"

BASE_URL = "https://data-team-production-db77.up.railway.app"
SWAGGER_URL = f"{BASE_URL}/docs"

PLAN_ROUTES = [
    ("free", "Free", "Single country; top-line answers only"),
    ("farmers", "Farmers", "Localized crop/rainfall/market; profile country geo filter"),
    ("government", "Government", "National/sub-national + historical trends"),
    ("ngos", "NGOs", "Government-tier depth + multi-region program framing"),
    ("agribusinesses", "Agribusinesses", "Cross-country comparison; market/volatility framing"),
    ("integrated", "Integrated", "Full access; category persona per message"),
]

CATEGORIES = [
    ("Government", "Government & Public Institutions"),
    ("NGOs", "Foundations, NGOs & Development Partners"),
    ("Agribusinesses", "Agribusinesses & Financial Institutions"),
    ("Farmers", "Farmers, Cooperatives & Communities"),
]


def _curl_plan(slug: str, plan_type: str, category: str, *, country: str | None = None) -> str:
    country_line = f',\n      "country": "{country}"' if country else ""
    return f"""curl -s -X POST "{BASE_URL}/query/{slug}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "query": "What is the maize outlook?",
    "user_profile": {{
      "plan_type": "{plan_type}",
      "category": "{category}"{country_line}
    }}
  }}'"""


def build() -> None:
    doc = new_document()
    add_title_page(
        doc,
        title="Ask ADZA API — Software Team Guide",
        subtitle="Production RAG API (plan-scoped query routes)",
        version="1.1",
        generated=date.today(),
    )
    add_toc_placeholder(doc)

    add_heading(doc, "Base URL", level=1)
    add_paragraph(doc, f"Production: {BASE_URL}")
    add_paragraph(doc, f"Live Swagger: {SWAGGER_URL}")
    add_bullets(
        doc,
        [
            "Entrypoint: uvicorn ml.rag.api:app",
            "No authentication today — treat the URL as sensitive; gateway auth recommended.",
            "CORS: RAG_CORS_ORIGINS (default *).",
        ],
    )

    add_heading(doc, "How plan types work", level=1)
    add_paragraph(
        doc,
        "There is one API service. Each subscription plan has its own query URL. "
        "The path locks plan_type — the client cannot escalate tier via the JSON body.",
    )
    add_field_table(
        doc,
        ("Method", "Path", "Plan", "Gates"),
        [
            ("POST", f"/query/{slug}", plan, gates)
            for slug, plan, gates in PLAN_ROUTES
        ],
    )
    add_paragraph(
        doc,
        "Generic POST /query still exists (plan_type from user_profile). Prefer the plan-scoped URLs above.",
    )

    add_heading(doc, "Categories", level=2)
    add_paragraph(doc, "category sets generation persona/tone. Send it in user_profile.")
    add_field_table(doc, ("ID", "Label"), CATEGORIES)

    add_heading(doc, "Quick start", level=1)
    add_json_example(
        doc,
        "Farmers plan (recommended pattern)",
        _curl_plan("farmers", "Farmers", "Farmers", country="Kenya"),
    )
    add_bullets(
        doc,
        [
            "Reuse session_id from the previous response for multi-turn chat.",
            "Omit chat_history to use server-side memory; send chat_history for client-owned history.",
            "Every successful response includes acf (confidence band, score 0–100, explanation).",
        ],
    )

    add_heading(doc, "POST /query/{plan} — request", level=1)
    add_field_table(
        doc,
        ("Field", "Type", "Required", "Description"),
        [
            ("query", "string", "Yes", "Natural language question"),
            ("session_id", "string | null", "No", "Omit to start; reuse for continuity"),
            ("user_id", "string | null", "No", "Optional product user id for analytics"),
            ("user_profile.plan_type", "string", "If profile sent", "Overridden by path on plan routes"),
            ("user_profile.category", "string", "If profile sent", "Generation persona"),
            ("user_profile.country", "string | null", "No", "Geo filter when plan is Farmers"),
            ("chat_history", "ChatMessage[] | null", "No", "Client-owned prior turns"),
            ("include_trace", "boolean", "No", "Debug: decomposition + retrieval counts"),
        ],
    )

    add_heading(doc, "Response", level=1)
    add_field_table(
        doc,
        ("Field", "Type", "Description"),
        [
            ("answer", "string", "Assistant prose (may include [N] footnotes)"),
            ("citations", "CitationItem[]", "Referenced sources by default"),
            ("acf", "ACFSignal", "Confidence: band, band_label, score, explanation"),
            ("session_id", "string", "Pass on the next request"),
            ("usage", "UsageStats", "LLM token totals for this request"),
            ("error", "string | null", "Pipeline-level error if any"),
            ("langfuse_trace_id", "string | null", "For POST /feedback"),
            ("trace", "object | null", "Only when include_trace=true"),
        ],
    )

    add_heading(doc, "ACFSignal", level=2)
    add_field_table(
        doc,
        ("Field", "Type", "Description"),
        [
            ("band", "string", "very_strong | strong | moderate | limited | low | no_evidence"),
            ("band_label", "string", "Human-readable band"),
            ("score", "integer", "0–100 composite confidence"),
            ("explanation", "string", "One-sentence rationale"),
        ],
    )

    add_heading(doc, "CitationItem", level=2)
    add_field_table(
        doc,
        ("Field", "Type", "Description"),
        [
            ("id", "integer", "Footnote number matching [N] in answer"),
            ("kind", "string", "academic | news | structured_data | policy | ota | …"),
            ("text", "string", "Human-readable citation line"),
            ("url", "string | null", "Link when available"),
        ],
    )

    add_heading(doc, "cURL examples (all plans)", level=1)
    for slug, plan, _gates in PLAN_ROUTES:
        category = plan if plan in {"Farmers", "Government", "NGOs", "Agribusinesses"} else "Government"
        country = "Kenya" if plan == "Farmers" else None
        add_json_example(doc, f"POST /query/{slug}", _curl_plan(slug, plan, category, country=country))

    add_heading(doc, "Other endpoints", level=1)

    add_endpoint_section(
        doc,
        method="POST",
        path="/query",
        description=(
            "Generic query (backward compatible). Send user_profile.plan_type in the body. "
            "Prefer POST /query/{plan} for new integrations."
        ),
        request_fields=[
            ("query", "string", "Yes", "Natural language question"),
            ("user_profile", "object", "Recommended", "plan_type + category required when sent"),
            ("session_id", "string | null", "No", "Multi-turn continuity"),
        ],
        response_fields=[
            ("answer", "string", "Assistant prose"),
            ("acf", "ACFSignal", "Confidence signal"),
            ("session_id", "string", "Reuse for next turn"),
        ],
    )

    add_endpoint_section(
        doc,
        method="DELETE",
        path="/session/{session_id}",
        description="Delete server-side conversation memory (new chat / logout).",
        errors=[("400", "Empty session_id")],
    )

    add_endpoint_section(
        doc,
        method="POST",
        path="/feedback",
        description="Record thumbs up/down on a Langfuse trace (use langfuse_trace_id from a prior response).",
        request_fields=[
            ("trace_id", "string", "Yes", "langfuse_trace_id from QueryResponse"),
            ("score", "number", "Yes", "1.0 = thumbs up, 0.0 = thumbs down"),
            ("comment", "string | null", "No", "Optional note (max 500 chars)"),
        ],
        errors=[("503", "Langfuse not configured or invalid trace id")],
    )

    add_endpoint_section(
        doc,
        method="GET",
        path="/health",
        description="Liveness probe. Always returns quickly.",
    )
    add_endpoint_section(
        doc,
        method="GET",
        path="/ready",
        description="Readiness probe. Reports missing Qdrant/LLM config keys without revealing secrets.",
    )

    add_heading(doc, "Errors", level=1)
    add_bullets(
        doc,
        [
            "422 — validation (invalid plan_type/category, bad body shape).",
            "500 — unexpected pipeline/server error (detail may include hints when RAG_DEBUG is on).",
            "503 — feedback when Langfuse is unavailable.",
        ],
    )

    add_heading(doc, "Related docs", level=1)
    add_bullets(
        doc,
        [
            "OpenTrace-RAG-Pipeline-Architecture.pdf — pipeline diagrams (internal)",
            "ml/rag/docs/API.md — full markdown reference (internal detail)",
            "Regenerate this file: python scripts/generate_software_team_api_docx.py from ml-eng/",
        ],
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_document(doc, str(OUTPUT))
    print(f"Wrote {OUTPUT}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
