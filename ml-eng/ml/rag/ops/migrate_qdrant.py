"""One-shot migration: copy Qdrant collections from a source cluster to a target cluster.

Used to move from Qdrant Cloud to the self-managed Qdrant service on Railway (ML-073),
without re-embedding: each collection is snapshotted on the source, and the target
recovers directly from the snapshot's URL over HTTP (Qdrant fetches it itself, so this
process never holds the snapshot bytes in memory or on disk).

See deploy/RAILWAY_QDRANT_MIGRATION.md for the Railway service setup and env var contract.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

_POLL_INTERVAL_S = 2.0
_SNAPSHOT_READY_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class CollectionMigrationResult:
    name: str
    source_points: int
    target_points: int
    ok: bool
    error: str = ""

    @property
    def matched(self) -> bool:
        return self.ok and self.source_points == self.target_points


def _env(name: str, *, required: bool = True, default: str = "") -> str:
    value = (os.environ.get(name) or "").strip()
    if required and not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value or default


def _collections_to_migrate(source: QdrantClient) -> list[str]:
    requested = (os.environ.get("MIGRATE_COLLECTIONS") or "").strip()
    if requested:
        return [name.strip() for name in requested.split(",") if name.strip()]
    return [col.name for col in source.get_collections().collections]


def _point_count(client: QdrantClient, name: str) -> int:
    info = client.get_collection(collection_name=name)
    return int(info.points_count or 0)


def _wait_for_snapshot(source: QdrantClient, collection: str, snapshot_name: str) -> None:
    """create_snapshot(wait=True) already blocks until ready; this is a defensive fallback
    for Qdrant versions/backends where the snapshot briefly 404s right after creation."""
    deadline = time.monotonic() + _SNAPSHOT_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        names = {snap.name for snap in source.list_snapshots(collection_name=collection)}
        if snapshot_name in names:
            return
        time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(f"Snapshot {snapshot_name!r} for {collection!r} never became listable")


def migrate_collection(
    *,
    collection: str,
    source: QdrantClient,
    target: QdrantClient,
    source_url: str,
    source_api_key: str,
    verify_only: bool,
) -> CollectionMigrationResult:
    try:
        source_points = _point_count(source, collection)
    except Exception as exc:  # collection missing on source, auth error, etc.
        return CollectionMigrationResult(collection, 0, 0, ok=False, error=f"source lookup failed: {exc}")

    if verify_only:
        try:
            target_points = _point_count(target, collection)
        except Exception as exc:
            return CollectionMigrationResult(collection, source_points, 0, ok=False, error=f"target lookup failed: {exc}")
        return CollectionMigrationResult(collection, source_points, target_points, ok=True)

    print(f"[{collection}] creating snapshot on source ({source_points} points)...", flush=True)
    try:
        snapshot = source.create_snapshot(collection_name=collection, wait=True)
    except Exception as exc:
        return CollectionMigrationResult(collection, source_points, 0, ok=False, error=f"snapshot creation failed: {exc}")
    if snapshot is None:
        return CollectionMigrationResult(collection, source_points, 0, ok=False, error="snapshot creation returned no result")

    _wait_for_snapshot(source, collection, snapshot.name)
    location = f"{source_url.rstrip('/')}/collections/{collection}/snapshots/{snapshot.name}"
    print(f"[{collection}] recovering from {snapshot.name!r} into target...", flush=True)
    try:
        target.recover_snapshot(
            collection_name=collection,
            location=location,
            api_key=source_api_key,
            priority="snapshot",
            wait=True,
        )
    except Exception as exc:
        return CollectionMigrationResult(collection, source_points, 0, ok=False, error=f"recovery failed: {exc}")

    try:
        target_points = _point_count(target, collection)
    except Exception as exc:
        return CollectionMigrationResult(collection, source_points, 0, ok=False, error=f"post-recovery lookup failed: {exc}")

    result = CollectionMigrationResult(collection, source_points, target_points, ok=True)
    status = "OK" if result.matched else "MISMATCH"
    print(f"[{status}] {collection}: {source_points} -> {target_points} points", flush=True)
    return result


def run() -> int:
    source_url = _env("QDRANT_OLD_URL")
    source_api_key = _env("QDRANT_OLD_API_KEY")
    target_url = _env("QDRANT_URL")
    target_api_key = _env("QDRANT_API_KEY")
    verify_only = (os.environ.get("MIGRATE_VERIFY_ONLY") or "").strip() == "1"

    source = QdrantClient(url=source_url, api_key=source_api_key)
    target = QdrantClient(url=target_url, api_key=target_api_key)

    collections = _collections_to_migrate(source)
    if not collections:
        print("No collections found on source -- nothing to migrate.", flush=True)
        return 1

    mode = "Verifying" if verify_only else "Migrating"
    print(f"{mode} {len(collections)} collection(s): {', '.join(collections)}", flush=True)

    results = [
        migrate_collection(
            collection=name,
            source=source,
            target=target,
            source_url=source_url,
            source_api_key=source_api_key,
            verify_only=verify_only,
        )
        for name in collections
    ]

    print("\n--- Summary ---", flush=True)
    failed = [r for r in results if not r.matched]
    for r in results:
        if r.matched:
            print(f"  OK       {r.name}: {r.source_points} points", flush=True)
        elif r.ok:
            print(f"  MISMATCH {r.name}: source={r.source_points} target={r.target_points}", flush=True)
        else:
            print(f"  FAILED   {r.name}: {r.error}", flush=True)

    if failed:
        print(f"\n{len(failed)} of {len(results)} collection(s) did not match.", flush=True)
        return 1
    print(f"\nAll {len(results)} collection(s) matched.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
