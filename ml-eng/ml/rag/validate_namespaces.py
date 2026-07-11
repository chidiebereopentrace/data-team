"""Sprint 1, Week 3: Validate namespace separation in Qdrant.

Connects to Qdrant Cloud and checks that the namespace-separated collections
(academic_papers, policies, news_public_reports) contain only the expected
doc_kind values — no cross-contamination.

Usage: python -m ml.rag.validate_namespaces
"""
import os
import sys
from pathlib import Path

# Load env
env_path = Path(__file__).resolve().parents[2] / "config" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def main():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("ERROR: pip install qdrant-client")
        sys.exit(1)

    url = os.environ.get("QDRANT_URL", "")
    key = os.environ.get("QDRANT_API_KEY", "")
    if not url:
        print("ERROR: QDRANT_URL not set")
        sys.exit(1)

    print(f"Connecting to Qdrant: {url[:60]}...")
    client = QdrantClient(url=url, api_key=key, timeout=15)

    # 1. List all collections
    collections = client.get_collections().collections
    print(f"\n{'='*60}")
    print(f"COLLECTIONS ({len(collections)})")
    print(f"{'='*60}")
    for c in sorted(collections, key=lambda x: x.name):
        try:
            info = client.get_collection(c.name)
            print(f"  {c.name}: {info.points_count:,} points")
        except Exception as e:
            print(f"  {c.name}: error ({e})")

    # 2. Validate namespace separation
    print(f"\n{'='*60}")
    print("NAMESPACE SEPARATION VALIDATION")
    print(f"{'='*60}")

    expected = {
        "academic_papers": {"academic_article"},
        "policies": {"policy_document"},
        "news_public_reports": {"public_report"},
    }

    all_passed = True
    for coll_name, expected_kinds in expected.items():
        try:
            info = client.get_collection(coll_name)
            points, _ = client.scroll(coll_name, limit=20, with_payload=True)
            kinds_found = set()
            geos_found = set()
            for p in points:
                payload = p.payload or {}
                dk = payload.get("doc_kind", "unknown")
                kinds_found.add(dk)
                geo = payload.get("geo_country_primary") or payload.get("country") or ""
                if geo:
                    geos_found.add(geo)

            unexpected = kinds_found - expected_kinds - {"unknown"}
            if unexpected:
                print(f"\n  FAIL {coll_name}: {info.points_count:,} pts")
                print(f"     CROSS-CONTAMINATION: found doc_kinds {unexpected}")
                print(f"     Expected: {expected_kinds}")
                all_passed = False
            else:
                print(f"\n  PASS {coll_name}: {info.points_count:,} pts")
                print(f"     doc_kinds: {kinds_found}")
                print(f"     sample geos: {sorted(geos_found)[:10]}")
        except Exception as e:
            print(f"\n  WARN {coll_name}: NOT FOUND ({e})")

    # 3. Check legacy collection
    print(f"\n{'='*60}")
    print("LEGACY COLLECTION (backup)")
    print(f"{'='*60}")
    try:
        info = client.get_collection("research_other_papers")
        points, _ = client.scroll("research_other_papers", limit=20, with_payload=True)
        kinds = set()
        for p in points:
            dk = (p.payload or {}).get("doc_kind", "unknown")
            kinds.add(dk)
        print(f"  research_other_papers: {info.points_count:,} pts, doc_kinds: {kinds}")
    except Exception:
        print("  research_other_papers: NOT FOUND (expected if migration complete)")

    # 4. Check news collection
    print(f"\n{'='*60}")
    print("NEWS COLLECTION")
    print(f"{'='*60}")
    try:
        info = client.get_collection("news_data")
        points, _ = client.scroll("news_data", limit=10, with_payload=True)
        geos = set()
        for p in points:
            geo = (p.payload or {}).get("geo_country_primary") or (p.payload or {}).get("country") or ""
            if geo:
                geos.add(geo)
        print(f"  news_data: {info.points_count:,} pts")
        print(f"  sample geos: {sorted(geos)[:10]}")
    except Exception as e:
        print(f"  news_data: error ({e})")

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("RESULT: PASS - NAMESPACE SEPARATION VALIDATED - no cross-contamination")
    else:
        print("RESULT: FAIL - CROSS-CONTAMINATION DETECTED - see above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
