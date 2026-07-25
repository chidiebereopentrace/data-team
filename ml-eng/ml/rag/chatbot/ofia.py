"""
OpenTrace Federated Intelligence Architecture (OFIA) — shared tier utilities.

Every context chunk in the RAG pipeline belongs to one of the three OFIA evidence tiers:

    Tier 1 — Authoritative institutional sources
              (FAO, World Bank, AfDB, national stats, government reports,
               satellite systems, peer-reviewed academic publications, BQ structured data)
              ACF weight: 25%

    Tier 2 — Local validation networks
              (cooperatives, NGOs, market boards, extension services, news outlets,
               OTA insights and metrics)
              ACF weight: 40%

    Tier 3 — User-contributed intelligence
              (Ask ADZA user-submitted observations — future capability)
              ACF weight: 35%

Tier 3 is not yet in the corpus; user-contributed records will be tagged when the
feature ships. Until then, any unrecognised source maps to Tier 2 (conservative).

Usage:
    from ml.rag.chatbot.ofia import infer_source_tier

    tier = infer_source_tier(chunk)   # returns 1, 2, or 3
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Canonical doc_kind → OFIA tier mapping
# ---------------------------------------------------------------------------
# Uses the `doc_kind` field (from Qdrant payload metadata) and `info_type`
# for records where the content-type is more specific than the collection type.
# When `source_tier` is added to ingested records by the Data team, this
# mapping will be replaced by a direct field read.

_DOC_KIND_TO_TIER: dict[str, int] = {
    # Tier 1 — institutional / peer-reviewed / government
    "bigquery": 1,
    "academic_article": 1,
    "academic": 1,
    "policy_document": 1,
    "policy": 1,
    "public_report": 1,
    "government_report": 1,
    "research": 1,
    # Tier 2 — local validation networks / news / OTA / formation
    "agricultural_practise": 2,
    "formation": 2,
    "news_article": 2,
    "news": 2,
    "ota_insight": 2,
    "ota_metric": 2,
    "ota_recommendation": 2,
    # Tier 2 — supplemental web (lower trust than internal corpus)
    "web_wikipedia": 2,
    "web_search": 2,
    # Tier 3 — user-contributed (placeholder; no records yet)
    "user_submitted": 3,
    "user_observation": 3,
}

# Default when doc_kind is absent or unrecognised — conservative Tier 2.
_DEFAULT_TIER: int = 2


def infer_source_tier(item: dict[str, Any]) -> int:
    """Infer the OFIA evidence tier for a context chunk.

    Checks (in priority order):
    1. ``source_tier`` field on the item or its metadata — explicit tag from Data team.
    2. ``info_type`` in metadata — more specific than doc_kind for some records
       (e.g. ``government_report`` within the academic_papers collection).
    3. ``doc_kind`` in metadata — the canonical collection-level type tag.
    4. ``_context_kind`` / ``source`` on the item — retrieval-time tag.
    5. Default: Tier 2.

    Parameters
    ----------
    item:
        A context chunk dict as produced by the RAG pipeline (post-merge).

    Returns
    -------
    int: 1, 2, or 3.
    """
    meta: dict[str, Any] = item.get("metadata") or {}

    # 1. Explicit source_tier field (future — Data team will add this)
    explicit = meta.get("source_tier") or item.get("source_tier")
    if explicit is not None:
        try:
            t = int(explicit)
            if t in (1, 2, 3):
                return t
        except (ValueError, TypeError):
            pass

    # 2. info_type override (e.g. government_report inside academic_papers collection)
    info_type = str(meta.get("info_type") or "").strip().lower()
    if info_type and info_type in _DOC_KIND_TO_TIER:
        return _DOC_KIND_TO_TIER[info_type]

    # 3. doc_kind from metadata (preferred — set at ingest)
    doc_kind = str(meta.get("doc_kind") or "").strip().lower()
    if doc_kind and doc_kind in _DOC_KIND_TO_TIER:
        return _DOC_KIND_TO_TIER[doc_kind]

    # 4. _context_kind / source from the chunk itself (retrieval-time tag)
    context_kind = str(
        item.get("_context_kind") or item.get("source") or ""
    ).strip().lower()
    if context_kind and context_kind in _DOC_KIND_TO_TIER:
        return _DOC_KIND_TO_TIER[context_kind]

    # 5. Conservative default
    return _DEFAULT_TIER


def tier_label(tier: int) -> str:
    """Human-readable label for a tier integer."""
    return {1: "Tier 1 (institutional)", 2: "Tier 2 (local network)", 3: "Tier 3 (user-contributed)"}.get(tier, f"Tier {tier}")
