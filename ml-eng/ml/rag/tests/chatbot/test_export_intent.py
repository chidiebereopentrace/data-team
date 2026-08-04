"""Tests for export intent detection."""
from __future__ import annotations

from ml.rag.chatbot.export_intent import detect_export_intent


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
