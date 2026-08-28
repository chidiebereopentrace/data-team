"""Detect analytical / comparative report intent for specialized RAG mode."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.export_intent import detect_export_intent
from ml.rag.chatbot.geo_regions import detect_regions_in_text

_REPORT_RE = re.compile(
    r"\b("
    r"analytical\s+report|detailed\s+(analytical\s+)?report|full\s+report|"
    r"comparative\s+(analytics?|analysis|report)|"
    r"analytics?|benchmark(?:ing)?|"
    r"past\s+\d+\s+years?|last\s+\d+\s+years?|over\s+the\s+(past|last)\s+\d+|"
    r"multi[- ]country|all\s+the\s+countr(?:y|ies)|across\s+countries|"
    r"time\s+series|historical\s+trends?|"
    r"assessment|assess(?:ing|ed)?\b|food\s+security\s+risk|hunger\s+pressure|"
    r"situation\s+(?:overview|assessment|analysis)"
    r")\b",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"\b(compar(?:e|ative|ison)|versus|vs\.?|rank(?:ing)?|top\s+\d+)\b",
    re.IGNORECASE,
)

_AGRI_RE = re.compile(
    r"\b("
    r"agricultur(?:e|al)|farming|crops?|commodit(?:y|ies)|production|"
    r"yields?|livestock|cereal|maize|rice|cassava|sorghum|millet|wheat|"
    r"faostat|food\s+security|investment|gdp|trade|prices?"
    r")\b",
    re.IGNORECASE,
)

_INVESTOR_BEST_RE = re.compile(
    r"\b("
    r"best\s+(?:african\s+)?countr(?:y|ies)\s+for\s+(?:agri(?:cultural)?\s+)?investment|"
    r"where\s+to\s+invest\s+in\s+agri|"
    r"agri(?:cultural)?\s+investment\s+attractiveness|"
    r"best\s+for\s+(?:agri(?:cultural)?\s+)?investment"
    r")\b",
    re.IGNORECASE,
)


def is_analytical_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    """
    True when the user wants multi-country / multi-year structured analysis or a report.

    Used to force BQ multi-intent plans, region expansion, and report-shaped generation.
    """
    q = (query or "").strip()
    if not q:
        return False

    if _INVESTOR_BEST_RE.search(q):
        return True
    # which + african + investment shaped asks
    ql = q.lower()
    if (
        "investment" in ql
        and re.search(r"\b(best|which)\b", ql)
        and re.search(r"\b(country|countries|african)\b", ql)
        and re.search(r"\b(agri|agricultur)", ql)
    ):
        return True

    dec = decomposition if isinstance(decomposition, dict) else {}
    intent = str(dec.get("intent") or "").strip().lower()
    geo_raw = dec.get("geography")
    geo = geo_raw if isinstance(geo_raw, list) else []
    has_multi_geo = len([g for g in geo if str(g).strip()]) >= 2
    has_region = bool(detect_regions_in_text(q))
    export = detect_export_intent(q)
    reportish = bool(_REPORT_RE.search(q))
    compareish = bool(_COMPARE_RE.search(q)) or intent == "compare"
    agri = bool(_AGRI_RE.search(q))

    if reportish and (agri or has_region or compareish or export):
        return True
    # Regional multi-pillar assessment (production / prices / hunger / IPC).
    if has_region and agri and re.search(
        r"\b(assessment|assess\b|diagnostic|risk\s+across|hunger|ipc|insecurit)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    # Document packages with agri/region/compare — not plain CSV/chart data pulls.
    if export in ("pdf", "docx", "multi") and (compareish or has_region or agri):
        return True
    if export in ("csv", "chart") and agri and (compareish or has_region or reportish):
        return True
    if compareish and (has_region or has_multi_geo) and agri:
        return True
    if has_region and agri and (compareish or reportish or _years_span(dec)):
        return True
    return False


def _years_span(decomposition: dict[str, Any]) -> bool:
    ts = str(decomposition.get("time_start") or "")[:4]
    te = str(decomposition.get("time_end") or "")[:4]
    if ts.isdigit() and te.isdigit():
        return abs(int(te) - int(ts)) >= 5
    return False


__all__ = ["is_analytical_query"]
