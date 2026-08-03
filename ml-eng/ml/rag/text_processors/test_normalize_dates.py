"""Unit tests for shared ISO / RFC-822 date normalization."""
from __future__ import annotations

from datetime import date

from ml.rag.chatbot.acf_metadata import enrich_acf_payload_fields
from ml.rag.text_processors.normalize_dates import normalize_published_at, normalize_to_iso_date


def test_iso_date() -> None:
    assert normalize_to_iso_date("2025-01-15") == "2025-01-15"
    assert normalize_to_iso_date("2025-01-15T12:30:00Z") == "2025-01-15"
    assert normalize_to_iso_date("2025-03") == "2025-03-01"
    assert normalize_to_iso_date("2024") == "2024-01-01"
    assert normalize_to_iso_date(date(2023, 6, 1)) == "2023-06-01"


def test_rfc822_full() -> None:
    assert normalize_to_iso_date("Fri, 25 Sep 2024 12:00:00 GMT") == "2024-09-25"
    assert normalize_published_at("Fri, 25 Sep 2024 12:00:00 GMT") == "2024-09-25"


def test_truncated_rejected() -> None:
    assert normalize_to_iso_date("Fri, 25 Se") is None
    assert normalize_published_at("Fri, 25 Se") == ""
    assert normalize_to_iso_date("not a date") is None


def test_enrich_clears_garbage_published_at() -> None:
    out = enrich_acf_payload_fields(
        {
            "published_at": "Fri, 25 Se",
            "geo_countries": "Kenya",
            "document_id": "doc-x",
        }
    )
    assert "published_at" not in out
    assert "as_of_date" not in out


def test_enrich_repairs_rfc822_published_at() -> None:
    out = enrich_acf_payload_fields(
        {
            "published_at": "Fri, 25 Sep 2024 12:00:00 GMT",
            "geo_countries": "Kenya",
            "document_id": "doc-y",
        }
    )
    assert out["published_at"] == "2024-09-25"
    assert out["as_of_date"] == "2024-09-25"
