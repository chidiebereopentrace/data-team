"""Chunk the OpenTrace Agricultural Library staging folder into formation and academic rows."""
from __future__ import annotations

import logging
from pathlib import Path

from ml.rag.ingestion.library_catalogue import (
    CatalogueRecord,
    find_catalogue_csv,
    join_key,
    load_catalogue,
    stamp_library_chunk,
)
from ml.rag.text_processors.preprocess.engines.research import list_pdf_files, preprocess_pdf
from ml.rag.text_processors.preprocess.models import ChunkOutput

logger = logging.getLogger(__name__)

_LIBRARY_EXCLUDED_ROLES = frozenset({"boilerplate"})


def chunk_library_staging(staging_dir: Path) -> tuple[list[ChunkOutput], list[ChunkOutput]]:
    """Preprocess PDFs and split them into formation chunks and academic chunks."""
    catalogue: dict[str, CatalogueRecord] = {}
    csv_path = find_catalogue_csv(staging_dir)
    if csv_path is None:
        logger.warning("No catalogue CSV under %s; routing from folder paths only", staging_dir)
    else:
        catalogue = load_catalogue(csv_path)
        logger.info("Loaded %s catalogue rows from %s", len(catalogue), csv_path.name)

    formation: list[ChunkOutput] = []
    academic: list[ChunkOutput] = []
    for pdf in list_pdf_files(staging_dir):
        relative = pdf.relative_to(staging_dir).as_posix()
        record = catalogue.get(join_key(relative))
        try:
            chunks = preprocess_pdf(
                pdf,
                doc_kind="agricultural_practise",
                input_dir=staging_dir,
                excluded_roles=_LIBRARY_EXCLUDED_ROLES,
            )
        except Exception:
            logger.exception("Skipping unreadable PDF %s", relative)
            continue
        for chunk in chunks:
            route, stamped = stamp_library_chunk(chunk, relative_path=relative, record=record)
            if route.target == "academic":
                academic.append(stamped)
            else:
                formation.append(stamped)
    return formation, academic
