"""Detect user requests for downloadable exports (CSV, chart, DOCX, PDF)."""
from __future__ import annotations

import re
from typing import Literal

ExportKind = Literal["csv", "chart", "docx", "pdf", "multi"]

_CSV_RE = re.compile(
    r"\b(csv|spreadsheet|excel|tabular\s+data|download\s+data|export\s+data)\b",
    re.IGNORECASE,
)
_CHART_RE = re.compile(
    r"\b(chart|graph|plot|visuali[sz]e|visuali[sz]ation|trend\s+line|bar\s+chart)\b",
    re.IGNORECASE,
)
_DOCX_RE = re.compile(
    r"\b(word\s+doc(ument)?|docx|\.docx)\b",
    re.IGNORECASE,
)
_PDF_RE = re.compile(
    r"\b(pdf|printable\s+report|download\s+report|export\s+report)\b",
    re.IGNORECASE,
)
_EXPORT_RE = re.compile(
    r"\b(export|download|save\s+as|generate\s+a\s+report|make\s+me\s+a\s+(report|document))\b",
    re.IGNORECASE,
)
_MULTI_RE = re.compile(
    r"\b(full\s+report|complete\s+package|report\s+with\s+charts?)\b",
    re.IGNORECASE,
)


def detect_export_intent(query: str) -> ExportKind | None:
    """Return the primary export kind requested, or None if no export intent."""
    q = (query or "").strip()
    if not q:
        return None

    if _MULTI_RE.search(q):
        return "multi"
    kinds: list[ExportKind] = []
    if _CSV_RE.search(q):
        kinds.append("csv")
    if _CHART_RE.search(q):
        kinds.append("chart")
    if _DOCX_RE.search(q):
        kinds.append("docx")
    if _PDF_RE.search(q):
        kinds.append("pdf")

    if len(kinds) > 1:
        return "multi"
    if kinds:
        return kinds[0]

    if _EXPORT_RE.search(q):
        return "pdf"
    return None


EXPORT_UPGRADE_MESSAGE = (
    "Downloadable exports (CSV, charts, Word reports, and PDFs) are available on the "
    "Agribusinesses and Integrated Ask ADZA plans. Upgrade your subscription or switch "
    "to the Agribusinesses or Integrated API endpoint to access this feature."
)


__all__ = ["ExportKind", "detect_export_intent", "EXPORT_UPGRADE_MESSAGE"]
