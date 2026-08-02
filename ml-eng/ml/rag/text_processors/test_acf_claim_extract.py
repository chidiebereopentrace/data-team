"""Unit tests for Hybrid C ACF claim extraction at preprocess."""
from __future__ import annotations

from unittest import mock

from ml.rag.text_processors.acf_claim_extract import (
    apply_claim_extract_to_meta,
    claim_extract_mode,
    extract_acf_claim,
)
from ml.rag.text_processors.chunk_contract import enrich_metadata


def test_declining_yoy_rules() -> None:
    text = "Maize yields declined by 12% YoY in Kenya's Rift Valley."
    claim = extract_acf_claim(text, meta={"domains": "yield"}, force_llm=False)
    assert claim["direction"] == "decreasing"
    assert claim["magnitude"] == -12.0
    assert claim["unit"] == "pct_yoy"
    assert claim["metric"] == "yield"
    assert "declined" in claim["finding"].lower() or "12%" in claim["finding"]


def test_rising_pct_rules() -> None:
    text = "Rice production increased 8.5% this season across Senegal."
    claim = extract_acf_claim(text, meta={"domains": "production"}, force_llm=False)
    assert claim["direction"] == "increasing"
    assert claim["magnitude"] == 8.5
    assert claim["unit"] == "pct"


def test_stable_rules() -> None:
    text = "Market prices remained stable through the quarter."
    claim = extract_acf_claim(text, meta={"domains": "price"}, force_llm=False)
    assert claim["direction"] == "stable"


def test_no_signal_unknown() -> None:
    text = "The workshop convened stakeholders from several ministries."
    claim = extract_acf_claim(text, force_llm=False)
    assert claim["direction"] == "unknown"
    assert "magnitude" not in claim
    assert claim["metric"] == "general"
    assert claim["finding"]


def test_signed_percent_infers_direction() -> None:
    text = "National cereal output changed by -5% year-on-year."
    claim = extract_acf_claim(text, force_llm=False)
    assert claim["direction"] == "decreasing"
    assert claim["magnitude"] == -5.0
    assert claim["unit"] == "pct_yoy"


def test_llm_not_called_when_rules_hit() -> None:
    text = "Drought reduced yields by 20%."
    with mock.patch(
        "ml.rag.text_processors.acf_claim_extract._llm_extract"
    ) as llm:
        claim = extract_acf_claim(text, force_llm=True)
        llm.assert_not_called()
    assert claim["direction"] == "decreasing"
    assert claim["magnitude"] == -20.0


def test_llm_called_only_on_miss_when_enabled() -> None:
    text = "Stakeholders discussed agricultural policy options."
    fake = {
        "finding": "Policy discussion without a numeric trend.",
        "metric": "policy",
        "direction": "stable",
        "magnitude": None,
        "unit": None,
    }
    with mock.patch(
        "ml.rag.text_processors.acf_claim_extract._llm_extract",
        return_value=fake,
    ) as llm:
        claim = extract_acf_claim(text, force_llm=True)
        llm.assert_called_once()
    assert claim["direction"] == "stable"
    assert claim["metric"] == "policy"


def test_llm_skipped_when_force_false_even_on_miss() -> None:
    text = "Stakeholders discussed agricultural policy options."
    with mock.patch(
        "ml.rag.text_processors.acf_claim_extract._llm_extract"
    ) as llm:
        claim = extract_acf_claim(text, force_llm=False)
        llm.assert_not_called()
    assert claim["direction"] == "unknown"


def test_claim_extract_mode_env(monkeypatch) -> None:
    monkeypatch.delenv("RAG_ACF_CLAIM_EXTRACT", raising=False)
    assert claim_extract_mode() == "off"
    monkeypatch.setenv("RAG_ACF_CLAIM_EXTRACT", "llm")
    assert claim_extract_mode() == "llm"
    monkeypatch.setenv("RAG_ACF_CLAIM_EXTRACT", "off")
    assert claim_extract_mode() == "off"


def test_apply_idempotent() -> None:
    meta = {
        "finding": "Existing finding.",
        "direction": "increasing",
        "metric": "maize_yield",
    }
    out = apply_claim_extract_to_meta(
        meta, "Yields declined 50%.", force_llm=False
    )
    assert out["finding"] == "Existing finding."
    assert out["direction"] == "increasing"


def test_skip_data_description() -> None:
    out = apply_claim_extract_to_meta(
        {}, "Yields declined 12% YoY.", corpus="data_description", force_llm=False
    )
    assert "direction" not in out


def test_ota_prefers_metric_text_lane() -> None:
    meta = {
        "metric_text": "Maize price fell 9% YoY in Nakuru.",
        "insight_text": "Farmers are worried about markets.",
        "domains": "price",
    }
    out = apply_claim_extract_to_meta(
        meta, "ignored body", force_llm=False
    )
    assert out["direction"] == "decreasing"
    assert out["magnitude"] == -9.0


def test_enrich_metadata_stamps_claim_fields() -> None:
    meta = enrich_metadata(
        {"geo_countries": "Kenya", "published_at": "2025-01-01", "domains": "yield"},
        corpus="news",
        document_id="doc1",
        chunk_index=0,
        total_chunks=1,
        text="Maize yields declined by 12% YoY across Kenya.",
    )
    assert meta["direction"] == "decreasing"
    assert meta["magnitude"] == -12.0
    assert meta["unit"] == "pct_yoy"
    assert meta["finding"]
    assert meta["tier"] == 2
    assert meta["as_of_date"] == "2025-01-01"
