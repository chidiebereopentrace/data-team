"""Sprint 1, Week 3: Identify research papers missing bibliographic metadata.

Supports TASKS.md §4 — "Identify the highest-impact papers among the 200–600
records missing bibliographic metadata" — by scanning the namespace-separated
research collections in Qdrant and reporting, per collection:
  - how many points are missing a clickable link (DOI or stable URL), and
  - how many are missing core citation fields (authors / title / year),
plus a sample of the worst offenders so the manual enrichment can be targeted.

This automates the *identification* half of the task. The actual DOI/URI lookup
and metadata update remains a manual data-curation step.

Usage: python -m ml.rag.find_missing_bibliography [--limit N] [--samples M]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load env (same pattern as validate_namespaces.py)
env_path = Path(__file__).resolve().parents[2] / "config" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

RESEARCH_COLLECTIONS = ("academic_papers", "policies", "public_reports")

# A record has a usable clickable link if it has any of these (DOI or URL).
_LINK_FIELDS = ("doi", "url", "link", "source_url")
# Core citation fields for a clean fallback render (title + author + year).
_TITLE_FIELDS = ("article_title", "title", "section_title", "label")
_AUTHOR_FIELDS = ("authors", "author")
_YEAR_FIELDS = ("publication_year", "year")


def _has_any(payload: dict, fields: tuple[str, ...]) -> bool:
    for f in fields:
        val = payload.get(f)
        if val is not None and str(val).strip():
            return True
    return False


def _title_of(payload: dict) -> str:
    for f in _TITLE_FIELDS:
        val = payload.get(f)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "(untitled)"


def _scan_collection(client, name: str, limit: int, samples: int) -> dict:
    from qdrant_client import models as qmodels  # noqa: F401  (imported lazily)

    total = 0
    missing_link = 0
    missing_title = 0
    missing_author = 0
    missing_year = 0
    worst_samples: list[str] = []

    offset = None
    while True:
        points, offset = client.scroll(
            name,
            limit=min(256, limit - total) if limit else 256,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            total += 1
            has_link = _has_any(payload, _LINK_FIELDS)
            has_title = _has_any(payload, _TITLE_FIELDS)
            has_author = _has_any(payload, _AUTHOR_FIELDS)
            has_year = _has_any(payload, _YEAR_FIELDS)

            if not has_link:
                missing_link += 1
            if not has_title:
                missing_title += 1
            if not has_author:
                missing_author += 1
            if not has_year:
                missing_year += 1

            # "Worst" = missing link AND at least one core field: prime enrichment target.
            if not has_link and (not has_author or not has_year) and len(worst_samples) < samples:
                worst_samples.append(
                    f"      - {_title_of(payload)[:90]} "
                    f"(link={'no'}, author={'yes' if has_author else 'no'}, "
                    f"year={'yes' if has_year else 'no'})"
                )
        if offset is None or (limit and total >= limit):
            break

    return {
        "total": total,
        "missing_link": missing_link,
        "missing_title": missing_title,
        "missing_author": missing_author,
        "missing_year": missing_year,
        "worst_samples": worst_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Max points to scan per collection (0 = all).")
    parser.add_argument("--samples", type=int, default=15, help="Sample worst offenders to print per collection.")
    args = parser.parse_args()

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
    client = QdrantClient(url=url, api_key=key, timeout=30)

    grand_missing_link = 0
    grand_total = 0

    for name in RESEARCH_COLLECTIONS:
        print(f"\n{'='*60}")
        print(f"COLLECTION: {name}")
        print(f"{'='*60}")
        try:
            stats = _scan_collection(client, name, args.limit, args.samples)
        except Exception as e:
            print(f"  WARN: could not scan ({e})")
            continue

        total = stats["total"]
        grand_total += total
        grand_missing_link += stats["missing_link"]
        if total == 0:
            print("  (empty)")
            continue

        def _pct(n: int) -> str:
            return f"{n:,} ({100.0 * n / total:.1f}%)"

        print(f"  Scanned points:        {total:,}")
        print(f"  Missing clickable link: {_pct(stats['missing_link'])}")
        print(f"  Missing title:          {_pct(stats['missing_title'])}")
        print(f"  Missing authors:        {_pct(stats['missing_author'])}")
        print(f"  Missing year:           {_pct(stats['missing_year'])}")
        if stats["worst_samples"]:
            print("  Priority enrichment targets (no link + missing author/year):")
            print("\n".join(stats["worst_samples"]))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if grand_total:
        print(
            f"  Across research collections: {grand_missing_link:,} of {grand_total:,} points "
            f"({100.0 * grand_missing_link / grand_total:.1f}%) lack a clickable link."
        )
    print("  Next step: manual DOI/URI lookup for the priority targets above, then")
    print("  update the corresponding Qdrant payloads (doi / url).")


if __name__ == "__main__":
    main()
