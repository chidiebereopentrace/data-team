"""
Per-corpus chunking and embedding profiles (single source of truth).

Override via env, e.g. RAG_EMBEDDING_MODEL_NEWS, RAG_CHUNK_TARGET_TOKENS_NEWS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

CorpusKey = Literal["news", "research", "ota"]

ChunkingStrategy = Literal[
    "recursive_semantic",  # news_data: paragraphs + semantic fallback
    "hierarchical_semantic",  # research: section blocks + semantic boundaries
    "lane_semantic",  # OTA_insights: semantic within each lane
]

INGEST_VERSION = "2026.08-acf-claim-v2"

# OpenTrace namespace for deterministic UUID5 chunk ids
CHUNK_ID_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


@dataclass(frozen=True)
class ChunkingProfile:
    corpus: CorpusKey
    qdrant_collection: str
    target_tokens: int
    overlap_pct: float
    max_chunks_per_doc: int | None
    min_tokens: int
    embedding_model: str
    vector_dim: int
    e5_prefix_passage: bool
    qdrant_vector_mode: Literal[
        "legacy",
        "dense_named",
        "sentence_named",
        "research_dual",
        "ota_triple",
    ]
    chunking_strategy: ChunkingStrategy = "recursive_semantic"

    @property
    def overlap_tokens(self) -> int:
        return max(0, int(self.target_tokens * self.overlap_pct))


@dataclass(frozen=True)
class CollectionProfile:
    chunking: ChunkingProfile
    env_collection_var: str


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_model(corpus: CorpusKey, default: str) -> str:
    key = f"RAG_EMBEDDING_MODEL_{corpus.upper()}"
    return os.environ.get(key, "").strip() or default


_CORPUS_VECTOR_DIM_DEFAULTS: dict[CorpusKey, int] = {
    "news": 384,
    "research": 384,
    "ota": 384,
}

_CORPUS_VECTOR_DIM_ENV: dict[CorpusKey, str] = {
    "news": "RAG_QDRANT_VECTOR_SIZE_NEWS",
    "research": "RAG_QDRANT_VECTOR_SIZE_RESEARCH",
    "ota": "RAG_QDRANT_VECTOR_SIZE_OTA",
}


def _vector_dim_for(corpus: CorpusKey) -> int:
    """Per-corpus dense dimension (defaults 384 unless RAG_QDRANT_VECTOR_SIZE_* set)."""
    env_key = _CORPUS_VECTOR_DIM_ENV[corpus]
    if os.environ.get(env_key, "").strip():
        return max(32, _env_int(env_key, _CORPUS_VECTOR_DIM_DEFAULTS[corpus]))
    return max(32, _env_int("RAG_QDRANT_VECTOR_SIZE", _CORPUS_VECTOR_DIM_DEFAULTS[corpus]))


def _news_profile() -> ChunkingProfile:
    return ChunkingProfile(
        corpus="news",
        qdrant_collection=os.environ.get("QDRANT_COLLECTION_NEWS", "news_data").strip() or "news_data",
        target_tokens=_env_int("RAG_CHUNK_TARGET_TOKENS_NEWS", 400),
        overlap_pct=_env_float("RAG_CHUNK_OVERLAP_PCT_NEWS", 0.15),
        max_chunks_per_doc=None,
        min_tokens=_env_int("RAG_CHUNK_MIN_TOKENS_NEWS", 40),
        embedding_model=_env_model("news", "intfloat/multilingual-e5-base"),
        vector_dim=_vector_dim_for("news"),
        e5_prefix_passage=True,
        qdrant_vector_mode="dense_named",
        chunking_strategy="recursive_semantic",
    )


def _research_profile() -> ChunkingProfile:
    return ChunkingProfile(
        corpus="research",
        qdrant_collection=os.environ.get("QDRANT_COLLECTION_RESEARCH_PAPERS", "research_other_papers").strip()
        or "research_other_papers",
        target_tokens=_env_int("RAG_CHUNK_TARGET_TOKENS_RESEARCH", 500),
        overlap_pct=_env_float("RAG_CHUNK_OVERLAP_PCT_RESEARCH", 0.10),
        max_chunks_per_doc=_env_int("RAG_CHUNK_MAX_CHUNKS_RESEARCH", 400),
        min_tokens=_env_int("RAG_CHUNK_MIN_TOKENS_RESEARCH", 80),
        embedding_model=_env_model("research", "intfloat/multilingual-e5-small"),
        vector_dim=_vector_dim_for("research"),
        e5_prefix_passage=True,
        qdrant_vector_mode="dense_named",
        chunking_strategy="hierarchical_semantic",
    )


def _ota_profile() -> ChunkingProfile:
    return ChunkingProfile(
        corpus="ota",
        qdrant_collection=(
            os.environ.get("QDRANT_COLLECTION_OTA_INSIGHTS", "").strip()
            or os.environ.get("QDRANT_COLLECTION_OTA", "").strip()
            or "OTA_insights"
        ),
        target_tokens=_env_int("RAG_CHUNK_TARGET_TOKENS_OTA", 500),
        overlap_pct=_env_float("RAG_CHUNK_OVERLAP_PCT_OTA", 0.10),
        max_chunks_per_doc=_env_int("RAG_CHUNK_MAX_CHUNKS_OTA", 100),
        min_tokens=_env_int("RAG_CHUNK_MIN_TOKENS_OTA", 40),
        embedding_model=_env_model("ota", "intfloat/multilingual-e5-base"),
        vector_dim=_vector_dim_for("ota"),
        e5_prefix_passage=True,
        qdrant_vector_mode="ota_triple",
        chunking_strategy="lane_semantic",
    )


PROFILES: dict[CorpusKey, ChunkingProfile] = {
    "news": _news_profile(),
    "research": _research_profile(),
    "ota": _ota_profile(),
}

# Map Qdrant collection names (and legacy aliases) → corpus key
COLLECTION_ALIASES: dict[str, CorpusKey] = {
    "news_data": "news",
    "opentrace_news": "news",
    "academic_papers": "research",
    "policies": "research",
    "public_reports": "research",
    "formation": "research",
    "research_other_papers": "research",
    "opentrace_research_papers": "research",
    "news_public_reports": "research",  # legacy alias only
    "OTA_insights": "ota",
    "opentrace_ota": "ota",
}


def profile_for_corpus(corpus: CorpusKey) -> ChunkingProfile:
    return PROFILES[corpus]


def profile_for_collection(collection_name: str) -> ChunkingProfile:
    name = (collection_name or "").strip()
    key = COLLECTION_ALIASES.get(name)
    if key:
        return PROFILES[key]
    for corpus, prof in PROFILES.items():
        if prof.qdrant_collection == name:
            return prof
    return PROFILES["news"]
