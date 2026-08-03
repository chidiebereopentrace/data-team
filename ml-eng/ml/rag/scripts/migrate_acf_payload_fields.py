"""
Migrate existing Qdrant payloads to include ACF Path B provenance + claim fields.

Derives ``tier``, ``data_level``, ``as_of_date``, ``region``, ``source_id`` from
existing payload spines and re-extracts ``finding`` / ``metric`` / ``direction`` /
``magnitude`` / ``unit`` from payload ``content`` (Hybrid C rules; LLM when
``RAG_ACF_CLAIM_EXTRACT=llm``). No re-embedding. Creates payload indexes when missing.

Covers profile collections **and** research-split collections
(``academic_papers``, ``policies``, ``public_reports`` / ``news_public_reports``,
``formation``) plus the legacy ``research_other_papers`` backup.

Usage (from repo root, with Qdrant env configured):

    PYTHONPATH=ml-eng python -m ml.rag.scripts.migrate_acf_payload_fields
    PYTHONPATH=ml-eng python -m ml.rag.scripts.migrate_acf_payload_fields --corpus research
    PYTHONPATH=ml-eng python -m ml.rag.scripts.migrate_acf_payload_fields --collection academic_papers
    PYTHONPATH=ml-eng python -m ml.rag.scripts.migrate_acf_payload_fields --dry-run
    RAG_ACF_CLAIM_EXTRACT=llm PYTHONPATH=ml-eng python -m ml.rag.scripts.migrate_acf_payload_fields

For a full rebuild with new INGEST_VERSION, prefer ``ingestion/rebuild_qdrant.py``.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from qdrant_client import QdrantClient

from ml.rag.chatbot.acf_metadata import enrich_acf_payload_fields
from ml.rag.scripts.qdrant_collection_specs import PAYLOAD_INDEXES
from ml.rag.text_processors.acf_claim_extract import apply_claim_extract_to_meta
from ml.rag.text_processors.chunk_contract import backfill_geo_from_text
from ml.rag.text_processors.chunking_config import profile_for_corpus

_CLAIM_KEYS = ("finding", "metric", "direction", "magnitude", "unit")

_ACF_PATCH_KEYS = (
    "tier",
    "data_level",
    "as_of_date",
    "region",
    "source_id",
    *_CLAIM_KEYS,
    # Date / geo repairs (set_payload merge; does not overwrite vectors)
    "published_at",
    "geo_countries",
    "geo_country_primary",
    "country",
)

logger = logging.getLogger(__name__)

CORPORA = ("news", "research", "ota", "data_description")

# Research split targets (env override → default), aligned with graph._RESEARCH_SPLIT_COLLECTIONS
# plus migrate_research_collection destination alias news_public_reports.
_RESEARCH_SPLIT_TARGETS: tuple[tuple[str, str], ...] = (
    ("QDRANT_COLLECTION_ACADEMIC_PAPERS", "academic_papers"),
    ("QDRANT_COLLECTION_POLICIES", "policies"),
    ("QDRANT_COLLECTION_PUBLIC_REPORTS", "public_reports"),
    ("QDRANT_COLLECTION_FORMATION", "formation"),
)
_RESEARCH_EXTRA_ALIASES = ("news_public_reports",)


def _client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "").strip()
    key = os.environ.get("QDRANT_API_KEY", "").strip() or None
    if not url:
        raise SystemExit("QDRANT_URL is required")
    return QdrantClient(url=url, api_key=key)


def _collections_for_corpus(corpus: str) -> list[str]:
    """Resolve all Qdrant collection names to migrate for a logical corpus."""
    profile = profile_for_corpus(corpus)  # type: ignore[arg-type]
    names: list[str] = [profile.qdrant_collection]

    if corpus == "research":
        for env_key, default in _RESEARCH_SPLIT_TARGETS:
            name = (os.environ.get(env_key, "") or default).strip() or default
            if name not in names:
                names.append(name)
        for alias in _RESEARCH_EXTRA_ALIASES:
            if alias not in names:
                names.append(alias)

    return names


def _corpus_for_collection(collection: str) -> str:
    """Pick PAYLOAD_INDEXES corpus key for an explicit --collection."""
    for corpus in CORPORA:
        if collection in _collections_for_corpus(corpus):
            return corpus
    # Split research collections share research indexes
    if collection in {
        "academic_papers",
        "policies",
        "public_reports",
        "news_public_reports",
        "formation",
        "research_other_papers",
    }:
        return "research"
    return "news"


def _collection_exists(client: QdrantClient, collection: str) -> bool:
    try:
        client.get_collection(collection)
        return True
    except Exception:
        return False


def _ensure_indexes(client: QdrantClient, corpus: str, collection: str) -> None:
    for name, schema in PAYLOAD_INDEXES.get(corpus, []):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=name,
                field_schema=schema,
            )
            logger.info("Ensured index %s.%s", collection, name)
        except Exception as exc:
            logger.debug("Index %s.%s: %s", collection, name, exc)


def migrate_collection(
    client: QdrantClient,
    corpus: str,
    collection: str,
    *,
    dry_run: bool = False,
    batch_size: int = 64,
    skip_missing: bool = True,
) -> int:
    if not _collection_exists(client, collection):
        msg = f"Collection missing: {collection}"
        if skip_missing:
            logger.warning("%s — skipping", msg)
            return 0
        raise SystemExit(msg)

    _ensure_indexes(client, corpus, collection)

    updated = 0
    next_offset: Any = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for pt in points:
            original = dict(pt.payload or {})
            payload = dict(original)
            content = payload.pop("content", None)
            text = str(content or "")
            had_claim_keys = [k for k in _CLAIM_KEYS if k in original]
            # Force re-extract: clear prior claim fields so quality gate can omit
            if text.strip():
                for k in _CLAIM_KEYS:
                    payload.pop(k, None)
                payload = backfill_geo_from_text(payload, text)
                claimed = apply_claim_extract_to_meta(payload, text, corpus=corpus)
            else:
                claimed = payload
            enriched = enrich_acf_payload_fields(claimed)
            if content is not None:
                enriched["content"] = content
            patch = {
                k: enriched[k]
                for k in _ACF_PATCH_KEYS
                if k in enriched and enriched[k] is not None
            }
            # Stale weak claims / garbage dates: keys present before but omitted after
            drop_keys = [k for k in had_claim_keys if k not in enriched]
            for k in ("published_at", "as_of_date"):
                if k in original and k not in enriched:
                    drop_keys.append(k)
            if not patch and not drop_keys:
                continue
            if dry_run:
                updated += 1
                continue
            if patch:
                client.set_payload(
                    collection_name=collection,
                    payload=patch,
                    points=[pt.id],
                )
            if drop_keys:
                client.delete_payload(
                    collection_name=collection,
                    keys=drop_keys,
                    points=[pt.id],
                )
            updated += 1
        if next_offset is None:
            break
    logger.info(
        "%s (%s): updated %d points%s",
        corpus,
        collection,
        updated,
        " (dry-run)" if dry_run else "",
    )
    return updated


def migrate_corpus(
    client: QdrantClient,
    corpus: str,
    *,
    dry_run: bool = False,
    batch_size: int = 64,
) -> int:
    total = 0
    for collection in _collections_for_corpus(corpus):
        total += migrate_collection(
            client,
            corpus,
            collection,
            dry_run=dry_run,
            batch_size=batch_size,
        )
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=CORPORA, default=None)
    parser.add_argument(
        "--collection",
        default=None,
        help="Migrate a single Qdrant collection (e.g. academic_papers)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    client = _client()
    total = 0
    if args.collection:
        corpus = args.corpus or _corpus_for_collection(args.collection)
        total = migrate_collection(
            client,
            corpus,
            args.collection.strip(),
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            skip_missing=False,
        )
    else:
        corpora = (args.corpus,) if args.corpus else CORPORA
        for corpus in corpora:
            total += migrate_corpus(
                client, corpus, dry_run=args.dry_run, batch_size=args.batch_size
            )
    print(f"Done. Points touched: {total}")


if __name__ == "__main__":
    main()
