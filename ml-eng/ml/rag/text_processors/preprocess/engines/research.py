from __future__ import annotations

from pathlib import Path

from ml.rag.text_processors.chunk_contract import document_id_from_path, enrich_metadata
from ml.rag.text_processors.chunking_config import profile_for_corpus
from ml.rag.text_processors.domain_taxonomy import (
    DEFAULT_MAX_DOMAINS_LONG_DOC,
    infer_domains,
    infer_info_type,
    infer_places_of_focus,
)
from ml.rag.text_processors.preprocess.bibliographic_metadata import resolve_bibliographic_metadata
from ml.rag.text_processors.preprocess.lineage import section_parent_id
from ml.rag.text_processors.preprocess.llama_split import TextSlice, cap_slices
from ml.rag.text_processors.preprocess.models import ChunkOutput
from ml.rag.text_processors.preprocess.section_roles import (
    SectionRole,
    classify_section,
    semantic_lane_for_section,
    should_exclude_section_role,
)
from ml.rag.text_processors.preprocess.split_strategy import split_structure_blocks
from ml.rag.text_processors.preprocess.structure_blocks import StructureBlock, elements_to_blocks
from ml.rag.text_processors.preprocess.unstructured_fast import partition_pdf


def list_pdf_files(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()], key=lambda p: str(p).lower())


def _drop_section(role: str, excluded_roles: frozenset[str] | None) -> bool:
    if excluded_roles is None:
        return should_exclude_section_role(role)
    return (role or "").strip().lower() in excluded_roles


def _indexable_blocks(
    blocks: list[StructureBlock],
    *,
    excluded_roles: frozenset[str] | None,
) -> list[StructureBlock]:
    kept: list[StructureBlock] = []
    for block in blocks:
        role = classify_section(
            block.section_title,
            block.hierarchy_path,
            content_type=block.content_type,
        )
        if _drop_section(role, excluded_roles):
            continue
        kept.append(block)
    return kept


def preprocess_pdf(
    pdf_path: Path,
    *,
    doc_kind: str = "academic_article",
    input_dir: Path | None = None,
    excluded_roles: frozenset[str] | None = None,
) -> list[ChunkOutput]:
    profile = profile_for_corpus("research")
    source_file = str(pdf_path)
    doc_id = document_id_from_path(source_file)

    elements = partition_pdf(pdf_path)
    blocks = _indexable_blocks(elements_to_blocks(elements), excluded_roles=excluded_roles)
    if not blocks:
        return []

    full_text = "\n\n".join(b.text for b in blocks)
    biblio = resolve_bibliographic_metadata(
        pdf_path,
        full_text,
        elements,
        input_dir=input_dir or pdf_path.parent,
    )
    inferred_domains = infer_domains(full_text, max_domains=DEFAULT_MAX_DOMAINS_LONG_DOC)
    info_type = infer_info_type(full_text, source_file)
    places = infer_places_of_focus(full_text)

    slices: list[TextSlice] = cap_slices(split_structure_blocks(blocks, profile), profile)
    if not slices:
        return []

    section_parents: dict[str, str] = {}
    kept: list[tuple[TextSlice, SectionRole]] = []
    for sl in slices:
        role = classify_section(sl.section_title, sl.hierarchy_path, content_type=sl.content_type)
        if _drop_section(role, excluded_roles):
            continue
        kept.append((sl, role))
        if sl.hierarchy_path not in section_parents:
            section_parents[sl.hierarchy_path] = section_parent_id(
                corpus="research",
                document_id=doc_id,
                hierarchy_path=sl.hierarchy_path,
            )

    chunks: list[ChunkOutput] = []
    total = len(kept)
    for i, (sl, role) in enumerate(kept):
        lane = semantic_lane_for_section(role, content_type=sl.content_type)
        parent_id = section_parents.get(sl.hierarchy_path, "")
        meta = enrich_metadata(
            {
                "doc_kind": doc_kind,
                "info_type": info_type,
                "domains": "; ".join(inferred_domains),
                "place_of_focus": "; ".join(places),
                "source_file": source_file,
                "strategy": "structure_token",
                **biblio,
            },
            corpus="research",
            document_id=doc_id,
            chunk_index=i,
            total_chunks=total,
            text=sl.text,
            section_title=sl.section_title,
            hierarchy_path=sl.hierarchy_path,
            parent_chunk_id=parent_id,
            semantic_lane=lane,
            section_role=role,
            content_type=sl.content_type,
        )
        chunks.append(ChunkOutput(id=str(meta["id"]), text=sl.text, metadata={k: v for k, v in meta.items() if k != "id"}))
    return chunks


def preprocess_folder(
    input_dir: Path,
    *,
    doc_kind: str = "academic_article",
    excluded_roles: frozenset[str] | None = None,
) -> list[ChunkOutput]:
    out: list[ChunkOutput] = []
    for pdf in list_pdf_files(input_dir):
        out.extend(
            preprocess_pdf(
                pdf,
                doc_kind=doc_kind,
                input_dir=input_dir,
                excluded_roles=excluded_roles,
            )
        )
    return out
