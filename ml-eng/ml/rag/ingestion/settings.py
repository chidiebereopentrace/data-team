from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ml.rag.ingestion.collections import ALL_SPECS
from ml.rag.ingestion.gdrive_ids import normalize_drive_folder_id


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ml_eng_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_ml_eng_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_ml_eng_root() / p).resolve()


def _folder_id(env_name: str, *, required: bool = True) -> str:
    raw = _env(env_name)
    if not raw:
        if required:
            raise RuntimeError(f"Missing env var: {env_name}")
        return ""
    return normalize_drive_folder_id(raw)


@dataclass(frozen=True)
class IngestionSettings:
    # Google Drive folder IDs (normalized from bare ID or share URL)
    gdrive_folder_research_papers_id: str
    gdrive_folder_other_papers_id: str
    gdrive_folder_news_id: str
    gdrive_folder_ota_insights_id: str
    staging_root: Path

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_research_papers: str
    qdrant_collection_news: str
    qdrant_collection_ota_insights: str


def _default_staging_root() -> Path:
    return (_ml_eng_root() / "data" / "local" / "gdrive_cache").resolve()


def load_ingestion_settings_from_env() -> IngestionSettings:
    """
    Required env:
      - GDRIVE_FOLDER_RESEARCH_PAPERS_ID
      - GDRIVE_FOLDER_NEWS_ID

    Optional env:
      - GDRIVE_FOLDER_OTHER_PAPERS_ID (policy / public reports; merged into research collection)
      - GDRIVE_FOLDER_OTA_INSIGHTS_ID (required for ``ota`` rebuild)
      - RAG_INGESTION_STAGING_ROOT
      - QDRANT_URL, QDRANT_API_KEY (required for upsert / create_qdrant_collections)
      - QDRANT_COLLECTION_* overrides
    """
    rp = _folder_id("GDRIVE_FOLDER_RESEARCH_PAPERS_ID")
    other = _folder_id("GDRIVE_FOLDER_OTHER_PAPERS_ID", required=False)
    nw = _folder_id("GDRIVE_FOLDER_NEWS_ID")
    ota = _folder_id("GDRIVE_FOLDER_OTA_INSIGHTS_ID", required=False)

    staging = _env("RAG_INGESTION_STAGING_ROOT")
    staging_root = _resolve_ml_eng_path(staging) if staging else _default_staging_root()

    return IngestionSettings(
        gdrive_folder_research_papers_id=rp,
        gdrive_folder_other_papers_id=other,
        gdrive_folder_news_id=nw,
        gdrive_folder_ota_insights_id=ota,
        staging_root=staging_root,
        qdrant_url=_env("QDRANT_URL"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        qdrant_collection_research_papers=_env(
            "QDRANT_COLLECTION_RESEARCH_PAPERS",
            ALL_SPECS["research"].default_collection_name,
        )
        or ALL_SPECS["research"].default_collection_name,
        qdrant_collection_news=_env(
            "QDRANT_COLLECTION_NEWS",
            ALL_SPECS["news"].default_collection_name,
        )
        or ALL_SPECS["news"].default_collection_name,
        qdrant_collection_ota_insights=_env(
            "QDRANT_COLLECTION_OTA_INSIGHTS",
            ALL_SPECS["ota"].default_collection_name,
        )
        or ALL_SPECS["ota"].default_collection_name,
    )
