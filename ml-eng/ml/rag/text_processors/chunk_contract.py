"""
Stable chunk / document identifiers and content hashing for idempotent Qdrant upserts.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from ml.rag.chatbot.acf_metadata import enrich_acf_payload_fields
from ml.rag.text_processors.acf_claim_extract import apply_claim_extract_to_meta
from ml.rag.text_processors.chunking_config import CHUNK_ID_NAMESPACE, INGEST_VERSION, CorpusKey

_NS = uuid.UUID(CHUNK_ID_NAMESPACE)


def normalize_chunk_text(text: str) -> str:
    return " ".join((text or "").split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_chunk_text(text).encode("utf-8")).hexdigest()


def document_id_from_path(path: str, *, dedupe_id: str | None = None) -> str:
    if dedupe_id and str(dedupe_id).strip():
        return str(dedupe_id).strip()
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:32]


def make_chunk_id(
    *,
    corpus: CorpusKey,
    document_id: str,
    chunk_index: int,
    text: str,
) -> str:
    ch = content_hash(text)
    return str(
        uuid.uuid5(
            _NS,
            f"{corpus}|{document_id}|{chunk_index}|{ch[:16]}",
        )
    )


def enrich_metadata(
    meta: dict[str, Any],
    *,
    corpus: CorpusKey,
    document_id: str,
    chunk_index: int,
    total_chunks: int,
    text: str,
    section_path: str = "",
    section_title: str = "",
    hierarchy_path: str = "",
    parent_chunk_id: str | None = None,
    semantic_lane: str = "",
    section_role: str = "",
    content_type: str = "",
) -> dict[str, Any]:
    out = dict(meta)
    out["document_id"] = document_id
    out["chunk_index"] = chunk_index
    out["total_chunks"] = total_chunks
    out["content_hash"] = content_hash(text)
    out["ingest_version"] = INGEST_VERSION
    if section_path:
        out["section_path"] = section_path
    if section_title:
        out["section_title"] = section_title
    if hierarchy_path:
        out["hierarchy_path"] = hierarchy_path
    if parent_chunk_id:
        out["parent_chunk_id"] = parent_chunk_id
    if semantic_lane:
        out["semantic_lane"] = semantic_lane
    if section_role:
        out["section_role"] = section_role
    if content_type:
        out["content_type"] = content_type
    out["id"] = make_chunk_id(
        corpus=corpus,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
    )
    # ACF claim/finding + structured D/M (Hybrid C: rules always, optional LLM)
    out = apply_claim_extract_to_meta(out, text, corpus=corpus)
    # ACF Path B provenance (tier / data_level / as_of_date / region / source_id)
    return enrich_acf_payload_fields(out)
