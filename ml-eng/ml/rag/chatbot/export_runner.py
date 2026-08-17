"""Orchestrate export builders and artifact upload for the RAG graph."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from ml.rag.chatbot.artifact_storage import mime_type_for_filename, upload_artifact
from ml.rag.chatbot.export_intent import ExportKind
from ml.rag.chatbot.exports.chart_builder import build_chart
from ml.rag.chatbot.exports.csv_builder import build_csv
from ml.rag.chatbot.exports.docx_builder import build_docx
from ml.rag.chatbot.exports.pdf_builder import build_pdf
from ml.rag.chatbot.exports.tabular import report_topic, rows_from_bq_results
from ml.rag.chatbot.plan_policy import allows_export
from ml.rag.observability import observed_span, trace_elapsed_ms, update_current_span_metadata

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


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
            id_raw = c.get("id")
            if not isinstance(id_raw, int):
                id_raw = 0
            ids.append(int(id_raw))
        except (TypeError, ValueError):
            continue
    return ids


def sections_from_answer(query: str, answer: str) -> list[dict[str, str]]:
    """Split markdown ## headings into report sections; fall back to summary + question."""
    text = (answer or "").strip()
    if not text:
        return [
            {"heading": "Executive summary", "body": ""},
            {"heading": "Question", "body": query},
        ]

    text = re.sub(r"(?<![#\n])(#{1,3}\s+)", r"\n\1", text)
    text = re.sub(
        r"^(#{1,3}\s+)([A-Z][A-Za-z]+(?:\s+[a-z]+){0,4})[ \t]+(?=[A-Z])",
        r"\1\2\n",
        text,
        flags=re.MULTILINE,
    )
    matches = list(_HEADING_RE.finditer(text))
    if len(matches) < 2:
        return [
            {"heading": "Executive summary", "body": text},
            {"heading": "Question", "body": query},
        ]

    sections: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        heading = m.group(1).strip() or f"Section {i + 1}"
        if body:
            sections.append({"heading": heading, "body": body})
    if not sections:
        sections = [{"heading": "Executive summary", "body": text}]
    sections.append({"heading": "Question", "body": query})
    return sections


def _report_sections(query: str, answer: str) -> list[dict[str, str]]:
    return sections_from_answer(query, answer)


def _caption_sections(query: str, answer: str) -> list[dict[str, str]]:
    """Short caption-only sections for data_export_only mode."""
    text = (answer or "").strip()
    # Keep a short caption body for PDF/DOCX wrappers.
    if len(text) > 600:
        cut = text[:600]
        stop = max(cut.rfind(". "), cut.rfind(".\n"))
        text = cut[: stop + 1].strip() if stop > 120 else cut.strip() + "…"
    return [
        {"heading": "Data summary", "body": text or "Structured data export."},
        {"heading": "Question", "body": query},
    ]


def _expand_export_kinds(
    export_kind: ExportKind,
    *,
    rows: list[dict[str, Any]],
    analytical: bool,
    data_export_only: bool = False,
) -> list[ExportKind]:
    if export_kind == "multi":
        kinds: list[ExportKind] = ["csv", "chart", "pdf"]
    else:
        kinds = [export_kind]

    if (analytical or data_export_only) and rows:
        # Ensure data package accompanies narrative / caption reports.
        if "csv" not in kinds:
            kinds.insert(0, "csv")
        if "chart" not in kinds and export_kind in ("pdf", "docx", "multi", "chart"):
            kinds.insert(1 if kinds and kinds[0] == "csv" else 0, "chart")
    if data_export_only and rows:
        # Prefer tabular delivery; keep pdf/docx as caption-only when requested.
        preferred: list[ExportKind] = []
        for k in ("csv", "chart", "pdf", "docx"):
            if k in kinds and k not in preferred:
                preferred.append(k)
        for k in kinds:
            if k not in preferred:
                preferred.append(k)
        kinds = preferred
    return kinds


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
    analytical = bool(state.get("analytical_mode"))
    task_mode = str(state.get("task_mode") or ("analytical" if analytical else "chat"))
    data_export_only = task_mode == "data_export_only"
    data_export = export_kind in {"csv", "chart", "multi"} or data_export_only
    if data_export and not rows:
        return []

    dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else {}
    doc_title, base = report_topic(query, decomposition=dec)
    citation_ids = _citation_ids(citations)
    acf = _acf_summary(state)
    if data_export_only:
        sections = _caption_sections(query, answer)
    else:
        sections = _report_sections(query, answer)
    kinds = _expand_export_kinds(
        export_kind,
        rows=rows,
        analytical=analytical,
        data_export_only=data_export_only,
    )

    artifacts: list[dict[str, Any]] = []
    chart_png: bytes | None = None

    with observed_span("export", input_data={"export_kind": export_kind, "kinds": kinds}):
        t0 = time.perf_counter()
        for kind in kinds:
            try:
                if kind == "csv":
                    if not rows:
                        continue
                    data, fname = build_csv(rows, filename=f"{base}.csv")
                    summary = f"CSV export ({len(rows)} rows)"
                elif kind == "chart":
                    if len(rows) < 2:
                        continue
                    data, fname = build_chart(
                        rows,
                        title=doc_title,
                        filename=f"{base}.png",
                    )
                    chart_png = data
                    summary = f"Chart visualization ({len(rows)} data points)"
                elif kind == "docx":
                    if chart_png is None and len(rows) >= 2:
                        try:
                            chart_png, _ = build_chart(rows, title=doc_title, filename="tmp.png")
                        except ValueError:
                            chart_png = None
                    data, fname = build_docx(
                        title=doc_title,
                        sections=sections,
                        table_rows=rows or None,
                        chart_png=chart_png,
                        citations=citations,
                        acf_summary=acf,
                        filename=f"{base}.docx",
                    )
                    summary = "Word report with citations and confidence summary"
                elif kind == "pdf":
                    if chart_png is None and len(rows) >= 2:
                        try:
                            chart_png, _ = build_chart(rows, title=doc_title, filename="tmp.png")
                        except ValueError:
                            chart_png = None
                    data, fname = build_pdf(
                        title=doc_title,
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
                "analytical_mode": analytical,
                "task_mode": task_mode,
                "section_count": len(sections),
                "latency_ms": trace_elapsed_ms(t0),
            }
        )

    return artifacts


__all__ = ["run_exports", "sections_from_answer"]
