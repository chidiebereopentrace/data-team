"""
Deprecated: use ``ml.rag.text_processors.preprocess`` engines.

Thin compatibility shim for code that still imports ``chunk_prose``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ml.rag.text_processors.chunking_config import ChunkingProfile
from ml.rag.text_processors.preprocess.llama_split import split_blocks
from ml.rag.text_processors.preprocess.structure_blocks import paragraphs_to_blocks


@dataclass(frozen=True)
class TextChunk:
    text: str
    section_path: str


def chunk_prose(text: str, profile: ChunkingProfile, *, default_section: str = "body") -> list[TextChunk]:
    blocks = paragraphs_to_blocks(text, default_section=default_section)
    tuples = [(b.hierarchy_path, b.section_title, b.text) for b in blocks]
    slices = split_blocks(tuples, profile)
    return [TextChunk(text=s.text, section_path=s.hierarchy_path) for s in slices]
