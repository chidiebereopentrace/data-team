"""
Summarize Qdrant contents for debugging retrieval (doc_kind counts, sample rows).

From ``ml-eng`` (PowerShell: ``$env:PYTHONPATH='.'``):

  python -m ml.rag.inspect_vector_db
  python -m ml.rag.inspect_vector_db --collection news_data --limit 5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect Qdrant collection payload metadata.")
    p.add_argument("--collection", default="opentrace_rag", help="Qdrant collection name")
    p.add_argument("--limit", type=int, default=8, help="Max documents to sample")
    args = p.parse_args()

    from ml.rag.local_env import load_rag_dotenv
    from ml.rag.retrievers.vector_retriever import _get_qdrant_config
    from qdrant_client import QdrantClient

    repo_root = Path(__file__).resolve().parents[2]
    load_rag_dotenv(repo_root)

    url, api_key, _, timeout_s = _get_qdrant_config()
    client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout_s))
    try:
        info = client.get_collection(collection_name=args.collection)
    except Exception as e:
        print(f"Collection {args.collection!r}: {e}")
        return 1

    points_count = getattr(info, "points_count", None)
    print(f"collection: {args.collection}  points: {points_count}")

    # Scroll payloads to sample metadata
    metas: list[dict[str, Any]] = []
    try:
        records, _ = client.scroll(
            collection_name=args.collection,
            limit=min(max(200, args.limit * 50), 1000),
            with_payload=True,
            with_vectors=False,
        )
        for r in records or []:
            payload = getattr(r, "payload", None) or {}
            if isinstance(payload, dict):
                metas.append(payload)
    except Exception:
        pass

    kinds = Counter()
    for m in metas:
        if not isinstance(m, dict):
            continue
        dk = str(m.get("doc_kind") or "").strip() or "(missing)"
        kinds[dk] += 1
    print("doc_kind (sampled):")
    for k, c in kinds.most_common(20):
        print(f"  {k}: {c}")

    print(f"\nfirst {min(args.limit, len(metas))} payload metadata keys (truncated):")
    for md in metas[: args.limit]:
        if not isinstance(md, dict):
            continue
        slim = {k: md[k] for k in list(md.keys())[:12]}
        print(json.dumps(slim, default=str)[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
