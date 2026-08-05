"""Orchestrate export builders and artifact upload for the RAG graph."""
from __future__ import annotations

import logging
import time
from typing import Any

from ml.rag.chatbot.artifact_storage import mime_type_for_filename, upload_artifact
from ml.rag.chatbot.export_intent import ExportKind
from ml.rag.chatbot.exports.chart_builder import build_chart
from ml.rag.chatbot.exports.csv_builder import build_csv
from ml.rag.chatbot.exports.docx_builder import build_docx
from ml.rag.chatbot.exports.pdf_builder import build_pdf
from ml.rag.chatbot.exports.tabular import rows_from_bq_results, slugify_filename
from ml.rag.chatbot.plan_policy import allows_export
from ml.rag.observability import observed_span, trace_elapsed_ms, update_current_span_metadata

logger = logging.getLogger(__name__)


def _acf_summary(state: dict[str, Any]) -> str:
    band = state.get("acf_band_label") or state.get("acf_band") or ""
    score = state.get("acf_score")
    explanation = state.get("acf_explanation") or state.get("acf_note") or ""
    parts = []
    if band:
        parts.append(f"Confidence band: {band}")
    if score is not None:
        parts.append(f"Score: {score}/100")
    if explanation:
        parts.append(str(explanation))
    return ". ".join(parts)


def _citation_ids(citations: list[dict[str, Any]] | None) -> list[int]:
    ids: list[int] = []
    for c in citations or []:
        try:
            ids.append(int(c.get("id")))
        except (TypeError, ValueError):
            continue
    return ids


def _report_sections(query: str, answer: str) -> list[dict[str, str]]:
    return [
        {"heading": "Executive summary", "body": answer},
        {"heading": "Question", "body": query},
    ]


def _chart_title(query: str) -> str:
    q = (query or "").strip()
    return q[:80] if q else "OpenTrace data"


def run_exports(
    *,
    export_kind: ExportKind,
    query: str,
    answer: str,
    bq_results: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None,
    state: dict[str, Any],
    export_enabled: bool,
    plan_type: str | None,
) -> list[dict[str, Any]]:
    """
    Build and upload artifacts. Raises ValueError when export is not allowed or data is missing.
    """
    if not export_enabled:
        raise ValueError("export not enabled for this API route")
    if not allows_export(plan_type):
        raise ValueError(f"export not allowed for plan_type={plan_type!r}")

    rows = rows_from_bq_results(bq_results)
    base = slugify_filename(query)
    citation_ids = _citation_ids(citations)
    acf = _acf_summary(state)
    sections = _report_sections(query, answer)
    kinds: list[ExportKind]
    if export_kind == "multi":
        kinds = ["csv", "chart", "pdf"]
    else:
        kinds = [export_kind]

    artifacts: list[dict[str, Any]] = []
    chart_png: bytes | None = None

    with observed_span("export", input_data={"export_kind": export_kind, "kinds": kinds}):
        t0 = time.perf_counter()
        for kind in kinds:
            try:
                if kind == "csv":
                    data, fname = build_csv(rows, filename=f"{base}.csv")
                    summary = f"CSV export ({len(rows)} rows)"
                elif kind == "chart":
                    data, fname = build_chart(
                        rows,
                        title=_chart_title(query),
                        filename=f"{base}.png",
                    )
                    chart_png = data
                    summary = f"Chart visualization ({len(rows)} data points)"
                elif kind == "docx":
                    if chart_png is None and rows:
                        try:
                            chart_png, _ = build_chart(rows, title=_chart_title(query), filename="tmp.png")
                        except ValueError:
                            chart_png = None
                    data, fname = build_docx(
                        title=_chart_title(query),
                        sections=sections,
                        table_rows=rows or None,
                        chart_png=chart_png,
                        citations=citations,
                        acf_summary=acf,
                        filename=f"{base}.docx",
                    )
                    summary = "Word report with citations and confidence summary"
                elif kind == "pdf":
                    if chart_png is None and rows:
                        try:
                            chart_png, _ = build_chart(rows, title=_chart_title(query), filename="tmp.png")
                        except ValueError:
                            chart_png = None
                    data, fname = build_pdf(
                        title=_chart_title(query),
                        sections=sections,
                        table_rows=rows or None,
                        chart_png=chart_png,
                        citations=citations,
                        acf_summary=acf,
                        filename=f"{base}.pdf",
                    )
                    summary = "PDF report with citations and confidence summary"
                else:
                    continue

                uploaded = upload_artifact(data, fname)
                artifacts.append(
                    {
                        "id": uploaded["id"],
                        "kind": kind,
                        "filename": uploaded["filename"],
                        "mime_type": uploaded.get("mime_type") or mime_type_for_filename(fname),
                        "url": uploaded["url"],
                        "summary": summary,
                        "citation_ids": citation_ids,
                        "byte_size": uploaded["byte_size"],
                    }
                )
            except Exception as exc:
                logger.warning("export builder failed kind=%s: %s", kind, exc)
                raise

        update_current_span_metadata(
            {
                "export_kind": export_kind,
                "artifact_count": len(artifacts),
                "plan_type": plan_type,
                "latency_ms": trace_elapsed_ms(t0),
            }
        )

    return artifacts


__all__ = ["run_exports"]
