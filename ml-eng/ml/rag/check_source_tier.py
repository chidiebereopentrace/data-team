"""Check whether any Qdrant collection has a source_tier (or related) field.

Pillar 1 (OFIA) gap analysis — run from data-team/ml-eng:
    python -m ml.rag.check_source_tier
"""
from __future__ import annotations
import os
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / "config" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from qdrant_client import QdrantClient

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY", ""), timeout=20)

TIER_FIELDS = ("source_tier", "tier", "data_tier", "source_level", "source_type", "doc_tier", "data_source_tier")
COLLECTIONS = (
    "news_data",
    "academic_papers",
    "policies",
    "public_reports",
    "formation",
    "OTA_insights",
)

for coll in COLLECTIONS:
    try:
        pts, _ = client.scroll(coll, limit=50, with_payload=True)
    except Exception as e:
        print(f"\n{coll}: ERROR — {e}")
        continue

    found: dict[str, set] = {}
    all_keys: set[str] = set()

    for p in pts:
        pay = p.payload or {}
        all_keys.update(pay.keys())
        for f in TIER_FIELDS:
            if f in pay:
                found.setdefault(f, set()).add(str(pay[f])[:50])

    print(f"\n{'=' * 60}")
    print(f"COLLECTION: {coll}  (sampled {len(pts)} points)")
    print(f"{'=' * 60}")

    if found:
        for f, vals in found.items():
            print(f"  FOUND tier field [{f}]: {sorted(vals)[:8]}")
    else:
        print(f"  NO tier field found. Checked: {TIER_FIELDS}")

    print(f"  All payload keys ({len(all_keys)}): {sorted(all_keys)}")
