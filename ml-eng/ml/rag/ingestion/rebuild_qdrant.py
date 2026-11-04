from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ml.rag.ingestion.collections import ALL_SPECS, CollectionSpec, DRIVE_REBUILD_KINDS, LIBRARY_REBUILD_KINDS
from ml.rag.ingestion.formation_library import chunk_library_staging
from ml.rag.ingestion.gdrive.auth import build_drive_service, load_auth_config_from_env
from ml.rag.ingestion.gdrive.sync import SyncStats, sync_drive_folder_to_cache
from ml.rag.ingestion.settings import IngestionSettings, load_ingestion_settings_from_env
from ml.rag.paths import library_academic_jsonl, preprocessed_data_root, preprocessed_jsonl_for_kind
from ml.rag.text_processors.preprocess.write_jsonl import write_chunks_jsonl


@dataclass(frozen=True)
class RebuildResult:
    kind: str
    collection_name: str
    staging_dir: Path
    chunk_jsonl_path: Path
    sync: SyncStats
    upserted: int
    academic_collection_name: str = ""
    academic_chunk_jsonl_path: Path | None = None
    academic_upserted: int = 0


def _merge_sync_stats(*stats: SyncStats) -> SyncStats:
    return SyncStats(
        scanned=sum(s.scanned for s in stats),
        downloaded=sum(s.downloaded for s in stats),
        skipped=sum(s.skipped for s in stats),
    )


def _chunks_dir() -> Path:
    return preprocessed_data_root()


def _run_preprocessor(kind: str, staging_dir: Path, out_jsonl: Path) -> None:
    if kind == "research":
        raise ValueError("Use _run_research_preprocessor for research kind")

    if kind == "news":
        from ml.rag.text_processors.news_collection_preprocessor import preprocess_news_collection

        preprocess_news_collection(
            input_dir=staging_dir,
            output_path=out_jsonl,
            chunk_chars=1200,
            overlap_chars=200,
            max_files=None,
        )
        return

    if kind == "ota":
        from ml.rag.text_processors.ota_insights_preprocessor import consolidate_ota_staging

        consolidate_ota_staging(input_dir=staging_dir, output_path=out_jsonl)
        return

    raise ValueError(f"Unknown kind: {kind}")


def _run_research_preprocessor(
    *,
    research_staging: Path,
    other_staging: Path | None,
    out_jsonl: Path,
) -> None:
    from ml.rag.text_processors.research_papers_preprocessor import preprocess_research_papers

    preprocess_research_papers(
        input_dir=research_staging,
        output_path=out_jsonl,
        doc_kind="academic_article",
        append=False,
    )
    if other_staging is not None and other_staging.exists():
        preprocess_research_papers(
            input_dir=other_staging,
            output_path=out_jsonl,
            doc_kind="policy_report",
            append=True,
        )


def _collection_name(settings: IngestionSettings, kind: str) -> str:
    if kind == "research":
        return settings.qdrant_collection_research_papers
    if kind == "news":
        return settings.qdrant_collection_news
    if kind == "ota":
        return settings.qdrant_collection_ota_insights
    raise ValueError(f"Unknown kind: {kind}")


def _sync_folder(
    *,
    service,
    folder_id: str,
    cache_root: Path,
    allowed_suffixes: tuple[str, ...],
) -> tuple[Path, SyncStats]:
    return sync_drive_folder_to_cache(
        service=service,
        folder_id=folder_id,
        cache_root=cache_root,
        allowed_suffixes=allowed_suffixes,
    )


def _upsert_library_jsonl(*, input_path: Path, collection: str, reset: bool, batch_size: int) -> int:
    from ml.rag.text_processors.load_pdf_chunks_to_vector_db import (
        PAYLOAD_RESEARCH,
        upsert_jsonl_to_qdrant_for_collection,
    )

    return upsert_jsonl_to_qdrant_for_collection(
        input_path=input_path,
        collection=collection,
        reset=reset,
        batch_size=batch_size,
        allowed_payload_keys=PAYLOAD_RESEARCH,
    )


def _rebuild_formation(*, reset: bool, settings: IngestionSettings) -> RebuildResult:
    if not settings.gdrive_folder_formation_id:
        raise RuntimeError("Set GDRIVE_FOLDER_FORMATION_ID for formation rebuild.")

    spec = ALL_SPECS["formation"]
    auth_cfg = load_auth_config_from_env()
    service = build_drive_service(auth_cfg)
    staging_root = settings.staging_root
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir, sync_stats = _sync_folder(
        service=service,
        folder_id=settings.gdrive_folder_formation_id,
        cache_root=staging_root,
        allowed_suffixes=spec.allowed_suffixes,
    )

    formation_chunks, academic_chunks = chunk_library_staging(staging_dir)
    chunks_dir = _chunks_dir()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_jsonl = preprocessed_jsonl_for_kind("formation").resolve()
    academic_jsonl = library_academic_jsonl().resolve()
    write_chunks_jsonl(formation_chunks, chunk_jsonl)
    write_chunks_jsonl(academic_chunks, academic_jsonl)

    batch_size = int(os.environ.get("RAG_INGESTION_UPSERT_BATCH_SIZE", "200") or 200)
    upserted = 0
    if formation_chunks:
        upserted = _upsert_library_jsonl(
            input_path=chunk_jsonl,
            collection=settings.qdrant_collection_formation,
            reset=reset,
            batch_size=batch_size,
        )
    academic_upserted = 0
    if academic_chunks:
        academic_upserted = _upsert_library_jsonl(
            input_path=academic_jsonl,
            collection=settings.qdrant_collection_academic_papers,
            reset=False,
            batch_size=batch_size,
        )
    return RebuildResult(
        kind="formation",
        collection_name=settings.qdrant_collection_formation,
        staging_dir=staging_dir,
        chunk_jsonl_path=chunk_jsonl,
        sync=sync_stats,
        upserted=upserted,
        academic_collection_name=settings.qdrant_collection_academic_papers,
        academic_chunk_jsonl_path=academic_jsonl,
        academic_upserted=academic_upserted,
    )


def rebuild_one(
    *,
    kind: str,
    reset: bool,
    settings: IngestionSettings | None = None,
) -> RebuildResult:
    if kind not in DRIVE_REBUILD_KINDS and kind not in LIBRARY_REBUILD_KINDS:
        raise ValueError(f"Unknown rebuild kind: {kind!r}")

    settings = settings or load_ingestion_settings_from_env(require_core_folders=kind != "formation")
    if kind == "formation":
        return _rebuild_formation(reset=reset, settings=settings)

    spec: CollectionSpec = ALL_SPECS[kind]

    if kind == "ota" and not settings.gdrive_folder_ota_insights_id:
        raise RuntimeError("Set GDRIVE_FOLDER_OTA_INSIGHTS_ID for OTA rebuild.")

    auth_cfg = load_auth_config_from_env()
    service = build_drive_service(auth_cfg)

    staging_root = settings.staging_root
    staging_root.mkdir(parents=True, exist_ok=True)

    sync_stats: SyncStats
    staging_dir: Path

    if kind == "research":
        research_staging, sync_a = _sync_folder(
            service=service,
            folder_id=settings.gdrive_folder_research_papers_id,
            cache_root=staging_root,
            allowed_suffixes=spec.allowed_suffixes,
        )
        other_staging: Path | None = None
        if settings.gdrive_folder_other_papers_id:
            other_staging, sync_b = _sync_folder(
                service=service,
                folder_id=settings.gdrive_folder_other_papers_id,
                cache_root=staging_root,
                allowed_suffixes=spec.allowed_suffixes,
            )
            sync_stats = _merge_sync_stats(sync_a, sync_b)
        else:
            sync_stats = sync_a
        staging_dir = research_staging
    elif kind == "news":
        staging_dir, sync_stats = _sync_folder(
            service=service,
            folder_id=settings.gdrive_folder_news_id,
            cache_root=staging_root,
            allowed_suffixes=spec.allowed_suffixes,
        )
        other_staging = None
    elif kind == "ota":
        staging_dir, sync_stats = _sync_folder(
            service=service,
            folder_id=settings.gdrive_folder_ota_insights_id,
            cache_root=staging_root,
            allowed_suffixes=spec.allowed_suffixes,
        )
        other_staging = None
    else:
        raise ValueError(f"Unknown kind: {kind}")

    chunks_dir = _chunks_dir()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_jsonl = preprocessed_jsonl_for_kind(kind).resolve()

    if kind == "research":
        _run_research_preprocessor(
            research_staging=staging_dir,
            other_staging=other_staging,
            out_jsonl=chunk_jsonl,
        )
    else:
        _run_preprocessor(kind, staging_dir, chunk_jsonl)

    collection = _collection_name(settings, kind)
    batch_size = int(os.environ.get("RAG_INGESTION_UPSERT_BATCH_SIZE", "200") or 200)

    if kind == "research":
        from ml.rag.text_processors.research_papers_load_to_vector_db import load_research_papers_to_qdrant

        upserted = load_research_papers_to_qdrant(
            input_path=chunk_jsonl,
            collection=collection,
            reset=reset,
            batch_size=batch_size,
        )
    elif kind == "news":
        from ml.rag.text_processors.news_load_to_vector_db import load_news_to_qdrant

        upserted = load_news_to_qdrant(
            input_path=chunk_jsonl,
            collection=collection,
            reset=reset,
            batch_size=batch_size,
        )
    elif kind == "ota":
        from ml.rag.text_processors.ota_insights_load_to_vector_db import load_ota_insights_to_qdrant

        upserted = load_ota_insights_to_qdrant(
            input_path=chunk_jsonl,
            collection=collection,
            reset=reset,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return RebuildResult(
        kind=kind,
        collection_name=collection,
        staging_dir=staging_dir,
        chunk_jsonl_path=chunk_jsonl,
        sync=sync_stats,
        upserted=upserted,
    )


def rebuild_many(*, kind: str, reset: bool) -> list[RebuildResult]:
    settings = load_ingestion_settings_from_env(require_core_folders=kind != "formation")
    if kind == "all":
        kinds = list(DRIVE_REBUILD_KINDS)
    else:
        kinds = [kind]
    out: list[RebuildResult] = []
    for k in kinds:
        out.append(rebuild_one(kind=k, reset=reset, settings=settings))
    return out
