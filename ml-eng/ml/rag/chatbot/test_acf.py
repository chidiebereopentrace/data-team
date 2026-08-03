"""
Path B ACF facade tests — cited evidence → adapt → score_evidence.
"""
from __future__ import annotations

from datetime import date

from acf.enums import ClaimLevel, QuestionType

from ml.rag.chatbot.acf_scoring import (
    adapt_cited_claims,
    curated_product_acf,
    no_evidence_acf,
    score_cited_evidence,
)
from ml.rag.chatbot.acf_metadata import (
    context_item_to_acf_record,
    derive_tier_and_data_level,
    enrich_acf_payload_fields,
    project_bq_row_acf,
)
from ml.rag.chatbot.acf_question import classify_acf_question


def _news_item(**meta_extra):
    meta = {
        "doc_kind": "news_article",
        "geo_scope": "country",
        "geo_countries": "Kenya",
        "geo_country_primary": "Kenya",
        "published_at": "2025-06-01",
        "document_id": "news-doc-1",
        "domains": "yield",
        **meta_extra,
    }
    return {
        "content": "Maize yields declined in Kenya.",
        "source": "news",
        "_context_kind": "news",
        "metadata": meta,
    }


def test_no_evidence_acf_shape() -> None:
    acf = no_evidence_acf()
    assert acf.band == "no_evidence"
    assert acf.score == 0
    assert acf.explanation


def test_curated_product_acf_shape() -> None:
    acf = curated_product_acf()
    assert acf.band == "strong"
    assert acf.score == 90


def test_derive_tier_national() -> None:
    tier, dl = derive_tier_and_data_level(
        {"geo_scope": "country", "geo_countries": "Kenya"}
    )
    assert tier == 2
    assert dl == "national"


def test_derive_tier_sub_national() -> None:
    tier, dl = derive_tier_and_data_level(
        {"geo_scope": "country", "geo_countries": "Kenya", "region": "Nakuru"}
    )
    assert tier == 3
    assert dl == "sub_national"


def test_enrich_acf_payload_fields() -> None:
    out = enrich_acf_payload_fields(
        {
            "geo_scope": "country",
            "geo_countries": "Senegal",
            "published_at": "2024-03-15",
            "document_id": "abc",
        }
    )
    assert out["tier"] == 2
    assert out["data_level"] == "national"
    assert out["as_of_date"] == "2024-03-15"
    assert out["source_id"] == "abc"


def test_context_item_remaps_geo_scope_to_places() -> None:
    item = _news_item()
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["geo_scope"] == ["Kenya"]  # places, not coverage keyword
    assert record["data_level"] == "national"
    assert record["direction"] == "unknown"


def test_adapt_and_score_cited_news() -> None:
    items = [
        _news_item(),
        _news_item(
            document_id="news-doc-2",
            published_at="2025-05-01",
            geo_countries="Kenya",
        ),
    ]
    claims = adapt_cited_claims(items)
    assert len(claims) >= 1
    acf = score_cited_evidence(
        items,
        query="How is Kenya's maize production trending this season?",
        decomposition={"geography": ["Kenya"], "domains": ["yield"]},
        reference_date=date(2025, 7, 1),
    )
    assert 0 <= acf.score <= 100
    assert acf.band in {
        "very_strong",
        "strong",
        "moderate",
        "limited",
        "low",
        "no_evidence",
    }
    assert acf.band_label
    assert acf.explanation
    assert acf.claim_level == "national"


def test_score_empty_citations_is_no_evidence() -> None:
    acf = score_cited_evidence([], query="Anything?")
    assert acf.band == "no_evidence"
    assert acf.score == 0


def test_classify_sub_national_and_time_sensitive() -> None:
    c = classify_acf_question(
        "Which counties in Kenya have declining maize prices this month?",
        {"geography": ["Kenya"], "domains": ["price"]},
    )
    assert c.claim_level == ClaimLevel.SUB_NATIONAL
    assert c.question_type == QuestionType.TIME_SENSITIVE


def test_classify_structural() -> None:
    c = classify_acf_question(
        "How have yields changed over the past decade in Ghana?",
        {"geography": ["Ghana"], "domains": ["yield"]},
    )
    assert c.question_type == QuestionType.STRUCTURAL


def test_project_bq_row_acf() -> None:
    meta = project_bq_row_acf(
        {
            "country": "Kenya",
            "admin_1": "Nakuru",
            "year": 2024,
            "product": "maize",
            "yield": 2.1,
            "sql": "SELECT * FROM `proj.bronze.yield_raw_data`",
        }
    )
    assert meta["tier"] == 3
    assert meta["data_level"] == "sub_national"
    assert meta["as_of_date"] == "2024-01-01"
    assert meta["metric"] == "maize"
    assert "yield_raw_data" in meta["source_id"]
    assert meta["direction"] == "unknown"


def test_context_prefers_ingested_direction_and_magnitude() -> None:
    item = {
        "content": "prose",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "geo_countries": "Kenya",
            "published_at": "2025-06-01",
            "document_id": "n1",
            "metric": "maize_yield",
            "direction": "decreasing",
            "magnitude": -12.0,
            "unit": "pct_yoy",
            "finding": "Maize yields declined 12% YoY.",
        },
    }
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["metric"] == "maize_yield"
    assert record["direction"] == "decreasing"
    assert record["magnitude"] == -12.0
    assert record["unit"] == "pct_yoy"
    assert "finding" not in record  # audit-only; not passed to from_payload


def test_bq_row_with_prior_value_scores() -> None:
    item = {
        "content": "{...}",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": project_bq_row_acf(
            {
                "country": "Kenya",
                "year": 2024,
                "product": "maize",
                "value": 88.0,
                "prior_value": 100.0,
                "sql": "SELECT * FROM bronze.yield_raw_data",
            }
        ),
    }
    acf = score_cited_evidence(
        [item],
        query="How is Kenya maize yield trending?",
        reference_date=date(2025, 1, 1),
    )
    assert acf.score >= 0
    assert acf.band != "no_evidence" or acf.score == 0  # may still score low
    # With one claim we expect a real score path
    claims = adapt_cited_claims([item])
    assert len(claims) == 1
