"""
Canonical filesystem paths for RAG ingestion (ml-eng repo root).
"""
from __future__ import annotations

from pathlib import Path

from ml.rag.text_processors.chunking_config import CorpusKey

# ml-eng/ (parents[2] from ml/rag/paths.py)
ML_ENG_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DATA_DIR = ML_ENG_ROOT / "data" / "local" / "preprocessed_data"

# Ingestion CLI kind → JSONL basename
INGEST_KIND_TO_JSONL: dict[str, str] = {
    "news": "news_chunks.jsonl",
    "research": "research_chunks.jsonl",
    "ota": "ota_insights_chunks.jsonl",
}

CORPUS_TO_JSONL: dict[CorpusKey, str] = {
    "news": "news_chunks.jsonl",
    "research": "research_chunks.jsonl",
    "ota": "ota_insights_chunks.jsonl",
}


def preprocessed_data_root() -> Path:
    """Directory for all preprocessed chunk JSONL files."""
    return PREPROCESSED_DATA_DIR.resolve()


def preprocessed_jsonl_for_kind(kind: str) -> Path:
    """JSONL path for an ingestion rebuild kind (news, research, ota)."""
    name = INGEST_KIND_TO_JSONL.get(kind)
    if not name:
        raise ValueError(f"Unknown ingestion kind: {kind!r}")
    return preprocessed_data_root() / name


def preprocessed_jsonl_for_corpus(corpus: CorpusKey) -> Path:
    """JSONL path for a chunking profile corpus key."""
    return preprocessed_data_root() / CORPUS_TO_JSONL[corpus]


def ingest_manifest_path() -> Path:
    return preprocessed_data_root() / "ingest_manifest.json"
