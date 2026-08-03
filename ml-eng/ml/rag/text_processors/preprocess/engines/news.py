from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ml.rag.text_processors.chunk_contract import content_hash, document_id_from_path, enrich_metadata
from ml.rag.text_processors.chunking_config import profile_for_corpus
from ml.rag.text_processors.domain_taxonomy import infer_domains
from ml.rag.text_processors.ingest_manifest import load_manifest, record_chunk, save_manifest, should_skip_chunk
from ml.rag.text_processors.news_docx_adapter import docx_to_news_txt_content
from ml.rag.text_processors.normalize_dates import normalize_published_at
from ml.rag.text_processors.preprocess.llama_split import TextSlice, cap_slices, split_blocks
from ml.rag.text_processors.preprocess.models import ChunkOutput
from ml.rag.text_processors.preprocess.structure_blocks import paragraphs_to_blocks


def _parse_front_matter_and_body(raw: str) -> tuple[dict[str, Any], str]:
    s = raw.lstrip()
    if not s.startswith("---\n"):
        return {}, s.strip()
    end_marker = "\n---\n"
    end_idx = s.find(end_marker, 4)
    if end_idx < 0:
        return {}, s.strip()
    yaml_block = s[4:end_idx]
    body = s[end_idx + len(end_marker) :]
    try:
        meta = yaml.safe_load(yaml_block) or {}
        return (meta if isinstance(meta, dict) else {}), body.strip()
    except Exception:
        return {}, body.strip()


# Re-export for news_preprocessor / tests that import from this module.
__all__ = [
    "normalize_published_at",
    "list_news_document_files",
    "preprocess_document",
    "preprocess_folder",
]


def _infer_country_from_path(path: Path, input_dir: Path) -> str:
    try:
        rel = path.relative_to(input_dir)
        if rel.parts:
            return rel.parts[0].replace("_", " ").title()
    except Exception:
        pass
    return ""


def list_news_document_files(input_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".txt", ".docx"):
            out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def preprocess_document(
    path: Path,
    input_dir: Path,
    *,
    manifest: dict | None = None,
) -> list[ChunkOutput]:
    profile = profile_for_corpus("news")
    if path.suffix.lower() == ".docx":
        raw = docx_to_news_txt_content(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    fm, body = _parse_front_matter_and_body(raw)
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body or body.lstrip().startswith("%PDF-"):
        return []

    title = str(fm.get("title", "")).strip()
    country = str(fm.get("country", "")).strip() or _infer_country_from_path(path, input_dir)
    doc_id = document_id_from_path(str(path), dedupe_id=str(fm.get("dedupe_id") or "") or None)

    blocks = paragraphs_to_blocks(body, default_section="body")
    tuples = [(b.hierarchy_path, b.section_title, b.text) for b in blocks]
    slices = cap_slices(split_blocks(tuples, profile), profile)

    chunks: list[ChunkOutput] = []
    total = len(slices)
    for i, sl in enumerate(slices):
        if manifest is not None and should_skip_chunk(manifest, content_hash=content_hash(sl.text), document_id=doc_id):
            continue
        domains = infer_domains(sl.text)
        meta = enrich_metadata(
            {
                "doc_kind": "news_article",
                "info_type": "news_article",
                "published_at": normalize_published_at(str(fm.get("published_at", "") or "")),
                "title": title,
                "country": country,
                "url": str(fm.get("url", "")).strip(),
                "source_file": str(path),
                "domains": "; ".join(domains),
            },
            corpus="news",
            document_id=doc_id,
            chunk_index=i,
            total_chunks=total,
            text=sl.text,
            section_path=sl.hierarchy_path,
            section_title=sl.section_title,
            hierarchy_path=sl.hierarchy_path,
            semantic_lane="content",
        )
        chunks.append(ChunkOutput(id=str(meta["id"]), text=sl.text, metadata={k: v for k, v in meta.items() if k != "id"}))
        if manifest is not None:
            record_chunk(manifest, document_id=doc_id, content_hash=content_hash(sl.text), source_file=str(path))
    return chunks


def preprocess_folder(input_dir: Path, *, max_files: int | None = None, use_manifest: bool = True) -> list[ChunkOutput]:
    files = list_news_document_files(input_dir)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    manifest = load_manifest() if use_manifest else None
    out: list[ChunkOutput] = []
    for p in files:
        out.extend(preprocess_document(p, input_dir, manifest=manifest))
    if manifest is not None:
        save_manifest(manifest)
    return out
