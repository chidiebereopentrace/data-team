"""
Split the mixed ``research_other_papers`` Qdrant collection into three
document-type collections, migrating points by their ``doc_kind`` metadata.

    academic_article  -> academic_papers
    policy_document   -> policies
    public_report     -> news_public_reports

The source collection is left untouched (kept as a backup / fallback). New
collections mirror the source's vector layout exactly (1 dense named "dense" +
1 sparse named "sparse"), so points are re-upserted with their existing vectors
— no re-embedding.

Idempotent: existing target collections are reused (points are upserted by id,
so re-running overwrites rather than duplicates).

Env: QDRANT_URL, QDRANT_API_KEY (loaded from config/.env if present).

Run:
  PYTHONPATH=. python -m ml.rag.scripts.migrate_research_collection --dry-run
  PYTHONPATH=. python -m ml.rag.scripts.migrate_research_collection
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_COLLECTION = "research_other_papers"

# doc_kind payload value -> destination collection name
DOC_KIND_TO_COLLECTION: dict[str, str] = {
    "academic_article": "academic_papers",
    "policy_document": "policies",
    "public_report": "news_public_reports",
}

# research-corpus payload fields worth indexing on the new collections
PAYLOAD_INDEX_FIELDS: list[str] = [
    "doc_kind",
    "geo_country_primary",
    "country",
    "geo_countries",  # TEXT
    "section_role",
    "content_type",
    "semantic_lane",
    "publication_year",
    "journal",
    "doi",
]
TEXT_INDEX_FIELDS = {"geo_countries"}


def _load_env() -> None:
    root = Path(__file__).resolve().parents[3]  # ml-eng/
    for candidate in (root / "config" / ".env", root / "data" / "local" / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
            print(f"[env] loaded {candidate}")
            return
    print("[env] no .env found; relying on process environment")


def _payload_doc_kind(payload: dict[str, Any]) -> str:
    raw_meta = payload.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else payload
    return str(meta.get("doc_kind") or payload.get("doc_kind") or "").strip()


def _create_target_like_source(client: Any, source_name: str, target_name: str) -> None:
    """Create target collection mirroring the source's vector + sparse config."""
    from qdrant_client.http import models

    existing = {c.name for c in client.get_collections().collections}
    if target_name in existing:
        print(f"[skip-create] {target_name} already exists")
        return

    src = client.get_collection(source_name)
    vectors_config = src.config.params.vectors
    sparse_config = getattr(src.config.params, "sparse_vectors", None)

    client.create_collection(
        collection_name=target_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_config,
    )
    print(f"[created] {target_name} (mirrored vector config from {source_name})")

    for field in PAYLOAD_INDEX_FIELDS:
        schema = (
            models.PayloadSchemaType.TEXT
            if field in TEXT_INDEX_FIELDS
            else models.PayloadSchemaType.KEYWORD
        )
        try:
            client.create_payload_index(
                collection_name=target_name, field_name=field, field_schema=schema
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg and "already indexed" not in msg:
                print(f"  [warn] index {field} on {target_name}: {exc}")


def _to_point_struct(point: Any) -> Any:
    """Rebuild a PointStruct from a scrolled record, preserving named + sparse vectors."""
    from qdrant_client.http import models

    return models.PointStruct(
        id=point.id,
        vector=point.vector,  # dict of named dense (+ sparse) vectors as returned by scroll
        payload=point.payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate research_other_papers by doc_kind.")
    parser.add_argument("--dry-run", action="store_true", help="Count only; no writes.")
    parser.add_argument("--batch", type=int, default=256, help="Scroll batch size.")
    parser.add_argument("--upsert-batch", type=int, default=64, help="Points per upsert request.")
    args = parser.parse_args()

    _load_env()
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip().strip('"').strip("'")
    if not url or not api_key:
        print("Missing QDRANT_URL / QDRANT_API_KEY", file=sys.stderr)
        return 1

    from qdrant_client import QdrantClient

    # Long timeout: uploads carry full vectors over the network to Qdrant Cloud.
    client = QdrantClient(url=url, api_key=api_key, check_compatibility=False, timeout=120)

    existing = {c.name for c in client.get_collections().collections}
    if SOURCE_COLLECTION not in existing:
        print(f"Source '{SOURCE_COLLECTION}' not found.", file=sys.stderr)
        return 1

    src_total = client.count(SOURCE_COLLECTION, exact=True).count
    print(f"Source '{SOURCE_COLLECTION}': {src_total} points")

    if not args.dry_run:
        for target in DOC_KIND_TO_COLLECTION.values():
            _create_target_like_source(client, SOURCE_COLLECTION, target)

    routed: Counter = Counter()
    unmapped: Counter = Counter()
    buffers: dict[str, list[Any]] = {t: [] for t in DOC_KIND_TO_COLLECTION.values()}
    migrated = 0
    scanned = 0
    next_page = None

    import time

    def _upsert_with_retry(target: str, pts: list[Any], attempts: int = 4) -> None:
        for i in range(attempts):
            try:
                client.upsert(collection_name=target, points=pts, wait=False)
                return
            except Exception as exc:
                if i == attempts - 1:
                    raise
                wait_s = 2 ** i
                print(f"  [retry {i+1}/{attempts}] {target} upsert failed ({exc}); waiting {wait_s}s", flush=True)
                time.sleep(wait_s)

    def _flush(target: str) -> None:
        nonlocal migrated
        if not buffers[target]:
            return
        if not args.dry_run:
            # Upload in small sub-batches so a single request never exceeds the
            # network write timeout (vectors are heavy over the wire).
            pts = buffers[target]
            for start in range(0, len(pts), args.upsert_batch):
                _upsert_with_retry(target, pts[start : start + args.upsert_batch])
        migrated += len(buffers[target])
        buffers[target] = []

    # Dry-run only needs payloads (fast); real migration needs vectors too.
    fetch_vectors = not args.dry_run
    while True:
        points, next_page = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=args.batch,
            offset=next_page,
            with_payload=True,
            with_vectors=fetch_vectors,
        )
        for p in points:
            scanned += 1
            kind = _payload_doc_kind(p.payload or {})
            target = DOC_KIND_TO_COLLECTION.get(kind)
            if not target:
                unmapped[kind or "(none)"] += 1
                continue
            routed[kind] += 1
            if not args.dry_run:
                buffers[target].append(_to_point_struct(p))
                if len(buffers[target]) >= args.batch:
                    _flush(target)
        if scanned % 2560 == 0:
            print(f"  ...scanned {scanned}/{src_total}", flush=True)
        if next_page is None:
            break

    for target in DOC_KIND_TO_COLLECTION.values():
        _flush(target)

    print("\n=== ROUTING SUMMARY ===")
    for kind, target in DOC_KIND_TO_COLLECTION.items():
        print(f"  {kind:18s} -> {target:22s}: {routed.get(kind, 0)}")
    if unmapped:
        print("\n  UNMAPPED doc_kinds (left in source only):")
        for k, v in unmapped.most_common():
            print(f"    {k}: {v}")

    print(f"\nscanned={scanned}  {'would migrate' if args.dry_run else 'migrated'}={sum(routed.values())}")

    if not args.dry_run:
        print("\n=== TARGET COUNTS ===")
        for target in DOC_KIND_TO_COLLECTION.values():
            try:
                print(f"  {target}: {client.count(target, exact=True).count} points")
            except Exception as exc:
                print(f"  {target}: count failed: {exc}")
        print(f"\nSource '{SOURCE_COLLECTION}' left intact: {client.count(SOURCE_COLLECTION, exact=True).count} points")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
