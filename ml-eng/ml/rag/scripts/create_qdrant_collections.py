"""
Create or recreate Qdrant collections for OpenTrace RAG.

Requires:
  QDRANT_URL, QDRANT_API_KEY

Optional:
  RAG_QDRANT_VECTOR_SIZE_NEWS / _RESEARCH / _OTA
  RAG_QDRANT_VECTOR_SIZE (fallback when per-corpus var unset)
  --skip-existing            only create collections that are missing (no delete)
  --indexes-only             skip create/delete; only (re)apply payload indexes from
                             PAYLOAD_INDEXES against existing collections (idempotent)

Run:
  PYTHONPATH=ml-eng python -m ml.rag.scripts.create_qdrant_collections
  PYTHONPATH=ml-eng python -m ml.rag.scripts.create_qdrant_collections --indexes-only
"""

from __future__ import annotations

import argparse
import os
import sys

from qdrant_client import QdrantClient

from ml.rag.local_env import load_data_local_dotenv
from ml.rag.paths import ML_ENG_ROOT
from ml.rag.scripts.qdrant_collection_specs import (
    COLLECTION_BUILDERS,
    ensure_payload_indexes,
    print_capacity_estimates,
)
from ml.rag.text_processors.chunking_config import PROFILES, CorpusKey


def main() -> int:
    parser = argparse.ArgumentParser(description="Create OpenTrace RAG Qdrant collections.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not delete collections that already exist (create missing only).",
    )
    parser.add_argument(
        "--indexes-only",
        action="store_true",
        help="Skip create/delete; only (re)apply payload indexes from PAYLOAD_INDEXES.",
    )
    args = parser.parse_args()

    load_data_local_dotenv(ML_ENG_ROOT)

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")
    if not qdrant_url or not qdrant_api_key:
        print("Set QDRANT_URL and QDRANT_API_KEY", file=sys.stderr)
        return 1

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)

    targets: list[tuple[CorpusKey, str]] = [
        ("news", PROFILES["news"].qdrant_collection),
        ("research", PROFILES["research"].qdrant_collection),
        ("ota", PROFILES["ota"].qdrant_collection),
    ]

    existing = {c.name for c in client.get_collections().collections}

    if args.indexes_only:
        for corpus_key, collection_name in targets:
            if collection_name not in existing:
                print(f"[SKIP] {collection_name} not found in Qdrant")
                continue
            indexed = ensure_payload_indexes(client, collection_name, corpus_key)
            if indexed:
                print(f"[INDEXED] {collection_name}: {', '.join(indexed)}")
            else:
                print(f"[OK] {collection_name}: all indexes already present")
        return 0

    for corpus_key, collection_name in targets:
        if collection_name in existing:
            if args.skip_existing:
                print(f"[SKIP] {collection_name} already exists")
                continue
            client.delete_collection(collection_name=collection_name)
            print(f"[DELETED] {collection_name}")

        kwargs = COLLECTION_BUILDERS[corpus_key]()
        client.create_collection(collection_name=collection_name, **kwargs)
        indexed = ensure_payload_indexes(client, collection_name, corpus_key)
        idx_msg = f", indexes: {', '.join(indexed)}" if indexed else ""
        dim = PROFILES[corpus_key].vector_dim
        print(f"[CREATED] {collection_name} ({corpus_key}, dim={dim}{idx_msg})")

    print("\n==============================")
    print("COLLECTIONS IN QDRANT")
    print("==============================")
    for c in client.get_collections().collections:
        print("-", c.name)

    print_capacity_estimates()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
