"""Tests for export intent detection and inline-citation gating."""
from __future__ import annotations

from ml.rag.chatbot.export_intent import detect_export_intent, want_inline_citations


def test_detect_csv_intent() -> None:
    assert detect_export_intent("Export maize prices as a CSV") == "csv"


def test_detect_chart_intent() -> None:
    assert detect_export_intent("Show this as a chart") == "chart"


def test_detect_pdf_intent() -> None:
    assert detect_export_intent("Generate a PDF report") == "pdf"


def test_detect_docx_intent() -> None:
    assert detect_export_intent("Make me a Word document") == "docx"


def test_detect_multi_intent() -> None:
    assert detect_export_intent("Full report with charts and CSV") == "multi"


def test_no_export_intent() -> None:
    assert detect_export_intent("What are maize yields in Kenya?") is None


def test_want_inline_citations_default_off() -> None:
    assert want_inline_citations("What are maize yields in Kenya?") is False
    assert want_inline_citations("Export maize as CSV", export_intent="csv") is False
    assert want_inline_citations("Show a chart", export_intent="chart") is False


def test_want_inline_citations_writeups_and_explicit() -> None:
    assert want_inline_citations("Make a PDF", export_intent="pdf") is True
    assert want_inline_citations("Word doc please", export_intent="docx") is True
    assert want_inline_citations("Full package", export_intent="multi") is True
    assert want_inline_citations("Regional maize overview", task_mode="analytical") is True
    assert want_inline_citations("Answer with footnotes please") is True
    assert want_inline_citations("Please use inline citations") is True
