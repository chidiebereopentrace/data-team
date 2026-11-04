"""Catalogue join and metadata stamp for the OpenTrace Agricultural Library."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ml.rag.chatbot.acf_metadata import enrich_acf_payload_fields
from ml.rag.text_processors.chunk_contract import document_id_from_path, make_chunk_id
from ml.rag.text_processors.preprocess.lineage import section_parent_id
from ml.rag.text_processors.preprocess.models import ChunkOutput

LibraryTarget = Literal["formation", "academic"]

_LIBRARY_ROOT = "opentrace agricultural library"
_YEAR_RE = re.compile(r"^\d{4}$")
_EXTENSION_TITLE_RE = re.compile(
    r"\b(trainer|training\s+of\s+trainers|tot|extension|facilitator)\b",
    re.IGNORECASE,
)
_ACF_PROVENANCE = ("tier", "data_level", "as_of_date", "region", "source_id")
_COUNTRY_ALIASES = {
    "congo, democratic republic of": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
}
_STAGE_FOLDERS = {
    "production": "production",
    "processing": "processing",
    "training": "training",
}
_GROUP_FOLDERS = {
    "crops": "crop",
    "livestock": "livestock",
    "general": "cross_cutting",
}
_TRAINING_TOPICS = frozenset(
    {
        "agribusiness",
        "climate_smart",
        "extension_methods",
        "farm_management",
        "nutrition_gender",
        "soil_water",
    }
)


@dataclass(frozen=True)
class CatalogueRecord:
    path: str
    title: str
    publisher: str
    year: str
    language: str
    url: str
    country: str
    doc_type: str
    is_academic: bool


@dataclass(frozen=True)
class LibraryRoute:
    target: LibraryTarget
    doc_kind: str
    identity_path: str
    payload: dict[str, str]


def library_relative_parts(path: str) -> tuple[str, ...]:
    """Path parts under the library root. The root folder name itself is dropped."""
    raw = (path or "").replace("\\", "/").strip()
    parts = [part.strip() for part in raw.split("/") if part.strip() and part.strip() not in (".", "..")]
    if parts and parts[0].lower() == _LIBRARY_ROOT:
        parts = parts[1:]
    return tuple(parts)


def join_key(path: str) -> str:
    """Case-insensitive key used to match a cache file to a catalogue row."""
    return "/".join(part.lower() for part in library_relative_parts(path))


def split_catalogue_countries(raw: str) -> list[str]:
    """Split a catalogue country cell on semicolons. Commas stay inside one name."""
    countries: list[str] = []
    for part in (raw or "").split(";"):
        name = part.strip()
        if not name:
            continue
        countries.append(_COUNTRY_ALIASES.get(name.lower(), name))
    return countries


def _snake(name: str) -> str:
    text = name.strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _truthy_academic(raw: str) -> bool:
    return raw.strip().lower() in {"y", "yes", "true", "1"}


def _language_code(raw: str) -> str:
    text = raw.strip().lower()
    if re.fullmatch(r"[a-z]{2}", text):
        return text
    return ""


def load_catalogue(csv_path: Path) -> dict[str, CatalogueRecord]:
    """Load catalogue rows keyed by :func:`join_key`."""
    records: dict[str, CatalogueRecord] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            cleaned = {(key or "").strip().lower(): (value or "").strip() for key, value in row.items()}
            path = cleaned.get("path", "")
            if not path:
                continue
            record = CatalogueRecord(
                path=path,
                title=cleaned.get("title", ""),
                publisher=cleaned.get("source_publisher", ""),
                year=cleaned.get("year", ""),
                language=_language_code(cleaned.get("language", "")),
                url=cleaned.get("original_url", ""),
                country=cleaned.get("country", ""),
                doc_type=cleaned.get("doc_type", ""),
                is_academic=_truthy_academic(cleaned.get("is_academic", "")),
            )
            records[join_key(path)] = record
    return records


def find_catalogue_csv(staging_dir: Path) -> Path | None:
    csvs = sorted((path for path in staging_dir.rglob("*.csv") if path.is_file()), key=lambda path: str(path).lower())
    for path in csvs:
        if "catalogue" in path.name.lower():
            return path
    return csvs[0] if csvs else None


def route_library_pdf(relative_path: str, record: CatalogueRecord | None) -> LibraryRoute:
    """Choose collection, doc_kind, and catalogue payload for one library PDF."""
    parts = library_relative_parts(relative_path)
    folders = parts[:-1] if parts else ()
    identity = record.path if record is not None else "/".join(parts)

    payload: dict[str, str] = {}
    stage_folders = folders
    if folders and folders[0].lower() == "academic publications":
        stage_folders = folders[1:]

    stage = _STAGE_FOLDERS.get(stage_folders[0].lower(), "") if stage_folders else ""
    group = _GROUP_FOLDERS.get(stage_folders[1].lower(), "") if len(stage_folders) > 1 else ""
    third = _snake(stage_folders[2]) if len(stage_folders) > 2 else ""

    if stage:
        payload["value_chain_stage"] = stage
    if group:
        payload["commodity_group"] = group

    topic = ""
    if stage == "training" and group == "cross_cutting":
        if third and third != "french" and third in _TRAINING_TOPICS:
            topic = third
        payload["commodity"] = "general"
    elif third:
        payload["commodity"] = third
    elif group == "cross_cutting":
        payload["commodity"] = "general"

    if topic:
        payload["training_topic"] = topic

    education_level = ""
    if folders and folders[0].lower() == "textbooks" and len(folders) > 1:
        level = folders[1].lower()
        if level in {"primary", "secondary", "tertiary"}:
            education_level = level
            payload["education_level"] = education_level
        if len(folders) > 2:
            payload["textbook_subject"] = _snake(folders[2])

    title = record.title if record is not None and record.title else ""
    if title:
        payload["title"] = title
        payload["article_title"] = title

    language = record.language if record is not None else ""
    french_bucket = (
        len(stage_folders) > 2
        and stage == "training"
        and group == "cross_cutting"
        and _snake(stage_folders[2]) == "french"
    )
    if french_bucket:
        language = "fr"
    if language:
        payload["language"] = language

    if record is not None:
        if record.publisher:
            payload["publisher"] = record.publisher
        if record.url:
            payload["url"] = record.url
        if record.doc_type:
            payload["content_form"] = record.doc_type
        year = record.year.strip()
        if _YEAR_RE.match(year):
            payload["publication_year"] = year
        countries = split_catalogue_countries(record.country)
        if countries:
            payload["geo_countries"] = "; ".join(countries)
            payload["geo_country_primary"] = countries[0]
            payload["country"] = countries[0]
            payload["place_of_focus"] = "; ".join(countries)
            if len(countries) >= 5:
                payload["geo_scope"] = "global"
            elif len(countries) > 1:
                payload["geo_scope"] = "multi_country"
            else:
                payload["geo_scope"] = "country"

    audience = _audience(folders, title, education_level)
    payload["audience"] = audience

    academic = (record is not None and record.is_academic) or (
        bool(folders) and folders[0].lower() == "academic publications"
    )
    if academic:
        payload["doc_kind"] = "academic_article"
        return LibraryRoute("academic", "academic_article", identity, payload)

    if folders and folders[0].lower() == "textbooks":
        payload["doc_kind"] = "textbook"
        return LibraryRoute("formation", "textbook", identity, payload)

    payload["doc_kind"] = "agricultural_practise"
    return LibraryRoute("formation", "agricultural_practise", identity, payload)


def _audience(folders: tuple[str, ...], title: str, education_level: str) -> str:
    if education_level in {"primary", "secondary"}:
        return "student"
    if education_level == "tertiary":
        return "trainee_professional"
    folder_names = {part.lower() for part in folders}
    if "training" in folder_names or _EXTENSION_TITLE_RE.search(title):
        return "extension_agent"
    return "farmer"


def stamp_library_chunk(
    chunk: ChunkOutput,
    *,
    relative_path: str,
    record: CatalogueRecord | None,
) -> tuple[LibraryRoute, ChunkOutput]:
    """Overlay catalogue metadata, recompute the chunk id, and refresh ACF provenance."""
    route = route_library_pdf(relative_path, record)
    meta: dict[str, Any] = dict(chunk.metadata)
    meta.update(route.payload)
    document_id = document_id_from_path(route.identity_path)
    meta["document_id"] = document_id
    chunk_index = int(meta.get("chunk_index") or 0)
    hierarchy = str(meta.get("hierarchy_path") or "")
    if hierarchy:
        meta["parent_chunk_id"] = section_parent_id(
            corpus="research",
            document_id=document_id,
            hierarchy_path=hierarchy,
        )
    meta["id"] = make_chunk_id(
        corpus="research",
        document_id=document_id,
        chunk_index=chunk_index,
        text=chunk.text,
    )
    for key in _ACF_PROVENANCE:
        meta.pop(key, None)
    if route.payload.get("publication_year"):
        meta.pop("published_at", None)
    meta = enrich_acf_payload_fields(meta)
    chunk_id = str(meta.pop("id"))
    return route, ChunkOutput(id=chunk_id, text=chunk.text, metadata=meta)
