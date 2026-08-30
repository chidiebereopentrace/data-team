#!/usr/bin/env python3
"""Backfill period_start, period_end, and year on existing Qdrant payloads (no re-embed)."""
from __future__ import annotations

import argparse
import os
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ml.rag.scripts.qdrant_collection_specs import PAYLOAD_INDEXES, ensure_payload_indexes

_YEAR_RE = re.compile(r"\b(19|20)\d{2})\b")


def _parse_year(text: str) -> str:
    m = _YEAR_RE.search(text or "")
    return m.group(1) if m else ""


def _derive_time_fields(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    pub = str(payload.get("published_at") or payload.get("publication_year") or "").strip()
    title = str(payload.get("title") or payload.get("section_title") or "").strip()
    year = str(payload.get("year") or "").strip()
    if not year:
        year = _parse_year(pub) or _parse_year(title)
    if year:
        out["year"] = year
        if not payload.get("period_start"):
            out["period_start"] = f"{year}-01-01"
        if not payload.get("period_end"):
            out["period_end"] = f"{year}-12-31"
    return out


def backfill_collection(client: QdrantClient, collection: str, *, corpus: str, dry_run: bool) -> int:
    ensure_payload_indexes(client, collection, corpus)
    updated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = dict(point.payload or {})
            patch = _derive_time_fields(payload)
            if not patch:
                continue
            for key, val in patch.items():
                if payload.get(key):
                    patch.pop(key, None)
            if not patch:
                continue
            updated += 1
            if dry_run:
                continue
            client.set_payload(
                collection_name=collection,
                payload=patch,
                points=[point.id],
            )
        if offset is None:
            break
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Qdrant time payload fields")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collection", action="append", default=[])
    args = parser.parse_args()

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    targets = args.collection or [
        os.environ.get("QDRANT_COLLECTION_NEWS", "news_data"),
        os.environ.get("QDRANT_COLLECTION_ACADEMIC_PAPERS", "academic_papers"),
        os.environ.get("QDRANT_COLLECTION_PUBLIC_REPORTS", "public_reports"),
    ]
    corpus_map = {
        os.environ.get("QDRANT_COLLECTION_NEWS", "news_data"): "news",
        os.environ.get("QDRANT_COLLECTION_ACADEMIC_PAPERS", "academic_papers"): "research",
        os.environ.get("QDRANT_COLLECTION_PUBLIC_REPORTS", "public_reports"): "research",
    }
    total = 0
    for coll in targets:
        corpus = corpus_map.get(coll, "news")
        if coll not in PAYLOAD_INDEXES and corpus not in PAYLOAD_INDEXES:
            corpus = "news"
        n = backfill_collection(client, coll, corpus=corpus, dry_run=args.dry_run)
        print(f"{coll}: {'would update' if args.dry_run else 'updated'} {n} points")
        total += n
    print(f"total: {total}")


if __name__ == "__main__":
    main()
