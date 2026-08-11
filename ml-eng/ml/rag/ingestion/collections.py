"""
Ingestion collection registry (Drive → preprocess → Qdrant).

Aligns with:
  - ``chunking_config.PROFILES`` (chunk sizes, embedding model, Qdrant vector mode)
  - ``scripts/qdrant_collection_specs`` (collection schema + payload indexes)
  - ``paths.preprocessed_data`` (JSONL output paths)

CLI ``kind`` keys are used by ``ingestion/rebuild_qdrant.py`` and ``ingestion/cli.py``.
``corpus`` keys match ``chunking_config.CorpusKey`` for loaders and ``profile_for_corpus``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ml.rag.paths import INGEST_KIND_TO_JSONL
from ml.rag.text_processors.chunking_config import PROFILES, CorpusKey


@dataclass(frozen=True)
class CollectionSpec:
    """
    High-level ingestion target for one RAG corpus.

    Attributes:
        kind: Rebuild CLI key (e.g. ``news``, ``research``, ``ota``).
        corpus: ``chunking_config`` profile key.
        collection_env: Env var for Qdrant collection name override.
        gdrive_folder_env: Env var for Google Drive folder ID (optional for OTA/manual).
        default_collection_name: Qdrant collection (from profile, respects env at runtime).
        default_doc_kind: Payload filter value for ``VectorRetriever`` / graph.
        qdrant_vector_mode: How points are stored and queried (dense_named, ota_triple, …).
        allowed_suffixes: File types synced from Drive for this corpus.
        preprocessed_jsonl: Basename under ``data/local/preprocessed_data/``.
    """

    kind: str
    corpus: CorpusKey
    collection_env: str
    gdrive_folder_env: str
    default_collection_name: str
    default_doc_kind: str
    qdrant_vector_mode: str
    allowed_suffixes: tuple[str, ...]
    preprocessed_jsonl: str


def _spec(
    *,
    kind: str,
    corpus: CorpusKey,
    collection_env: str,
    gdrive_folder_env: str,
    default_doc_kind: str,
    allowed_suffixes: tuple[str, ...],
) -> CollectionSpec:
    prof = PROFILES[corpus]
    return CollectionSpec(
        kind=kind,
        corpus=corpus,
        collection_env=collection_env,
        gdrive_folder_env=gdrive_folder_env,
        default_collection_name=prof.qdrant_collection,
        default_doc_kind=default_doc_kind,
        qdrant_vector_mode=prof.qdrant_vector_mode,
        allowed_suffixes=allowed_suffixes,
        preprocessed_jsonl=INGEST_KIND_TO_JSONL.get(kind, f"{kind}_chunks.jsonl"),
    )


NEWS = _spec(
    kind="news",
    corpus="news",
    collection_env="QDRANT_COLLECTION_NEWS",
    gdrive_folder_env="GDRIVE_FOLDER_NEWS_ID",
    default_doc_kind="news_article",
    allowed_suffixes=(".txt", ".docx"),
)

RESEARCH_PAPERS = _spec(
    kind="research",
    corpus="research",
    collection_env="QDRANT_COLLECTION_RESEARCH_PAPERS",
    gdrive_folder_env="GDRIVE_FOLDER_RESEARCH_PAPERS_ID",
    default_doc_kind="academic_article",
    allowed_suffixes=(".pdf",),
)

OTA_INSIGHTS = _spec(
    kind="ota",
    corpus="ota",
    collection_env="QDRANT_COLLECTION_OTA_INSIGHTS",
    gdrive_folder_env="GDRIVE_FOLDER_OTA_INSIGHTS_ID",
    default_doc_kind="ota_insight",
    allowed_suffixes=(".json", ".jsonl", ".docx"),
)

# Kinds with a full Drive → preprocess → upsert path in rebuild_qdrant.py
DRIVE_REBUILD_KINDS: tuple[str, ...] = ("research", "news", "ota")

ALL_SPECS: dict[str, CollectionSpec] = {
    RESEARCH_PAPERS.kind: RESEARCH_PAPERS,
    NEWS.kind: NEWS,
    OTA_INSIGHTS.kind: OTA_INSIGHTS,
}

# Legacy Qdrant names still accepted via chunking_config.COLLECTION_ALIASES:
#   opentrace_news → news_data
#   opentrace_research_papers → research_other_papers
