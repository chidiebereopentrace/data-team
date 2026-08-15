"""
Automated bibliography enrichment for Qdrant research collections (ML-046).

Scrolls academic_papers, policies, and public_reports collections for points
missing DOI/URL, queries the CrossRef public API by title, and updates Qdrant
payload when a high-confidence match is found.

Design:
  - Dry-run by default (--apply to write to Qdrant)
  - File checkpoint so runs are resumable
  - CrossRef polite pool: mailto header + 1 req/s default
  - Title fuzzy-match threshold: 0.85 (configurable)
  - Batch Qdrant payload updates (100 points per batch)

Usage:
    # Dry run — see what would be updated (safe):
    python -m ml.rag.enrich_bibliography --dry-run

    # Apply updates to Qdrant:
    python -m ml.rag.enrich_bibliography --apply

    # Single collection, limit to first 500 points:
    python -m ml.rag.enrich_bibliography --apply --collections academic_papers --limit 500

    # Resume from checkpoint:
    python -m ml.rag.enrich_bibliography --apply --checkpoint data/local/enrich_checkpoint.json

Requires: QDRANT_URL, QDRANT_API_KEY in config/.env
CrossRef API is public — no key needed. Set ENRICH_MAILTO for polite pool.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).resolve().parents[2] / "config" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

RESEARCH_COLLECTIONS = ("academic_papers", "policies", "public_reports")
_LINK_FIELDS = ("doi", "url", "link", "source_url")
_TITLE_FIELDS = ("article_title", "title", "section_title", "label")
_AUTHOR_FIELDS = ("authors", "author")
_YEAR_FIELDS = ("publication_year", "year")

CROSSREF_URL = "https://api.crossref.org/works"
_DEFAULT_DELAY_S = 1.0
_DEFAULT_THRESHOLD = 0.85
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_any(payload: dict, fields: tuple) -> bool:
    for f in fields:
        v = payload.get(f)
        if v is not None and str(v).strip():
            return True
    return False


def _title_of(payload: dict) -> str:
    for f in _TITLE_FIELDS:
        v = payload.get(f)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _normalize_title(t: str) -> str:
    """Lowercase, strip accents, collapse whitespace for fuzzy compare."""
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _title_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity (Jaccard on word sets)."""
    sa = set(_normalize_title(a).split())
    sb = set(_normalize_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _crossref_lookup(
    title: str,
    mailto: str,
    delay_s: float,
) -> dict[str, Any] | None:
    """
    Query CrossRef works API by title. Returns the top item dict or None.
    Respects rate limit with a sleep after each call.
    """
    params = urllib.parse.urlencode({
        "query.title": title[:200],
        "rows": "1",
        "select": "DOI,title,author,published,container-title",
    })
    url = f"{CROSSREF_URL}?{params}"
    headers = {
        "User-Agent": f"OpenTrace-BibEnrich/1.0 (mailto:{mailto})",
        "Accept": "application/json",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        time.sleep(delay_s)
        items = (data.get("message") or {}).get("items") or []
        return items[0] if items else None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("CrossRef rate limit hit — sleeping 10s")
            time.sleep(10)
        else:
            logger.debug("CrossRef HTTP %s for title=%r", e.code, title[:60])
        return None
    except Exception as exc:
        logger.debug("CrossRef error for title=%r: %s", title[:60], exc)
        return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _extract_crossref_fields(item: dict) -> dict[str, Any]:
    """Extract doi, url, authors, year from a CrossRef works item."""
    out: dict[str, Any] = {}
    doi = (item.get("DOI") or "").strip()
    if doi:
        out["doi"] = doi
        out["url"] = f"https://doi.org/{doi}"
    raw_authors = item.get("author") or []
    if raw_authors:
        names = []
        for a in raw_authors[:10]:
            family = (a.get("family") or "").strip()
            given = (a.get("given") or "").strip()
            if family:
                names.append(f"{given} {family}".strip() if given else family)
        if names:
            out["authors"] = "; ".join(names)
    pub = item.get("published") or item.get("published-print") or item.get("published-online")
    if isinstance(pub, dict):
        dp = pub.get("date-parts")
        if dp and isinstance(dp, list) and dp[0]:
            out["publication_year"] = int(dp[0][0])
    return out


def _crossref_title(item: dict) -> str:
    titles = item.get("title") or []
    if isinstance(titles, list) and titles:
        return str(titles[0])
    return str(titles) if titles else ""


def _load_checkpoint(path: str) -> set[str]:
    p = Path(path)
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()


def _save_checkpoint(path: str, seen: set[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sorted(seen)), encoding="utf-8")


# ---------------------------------------------------------------------------
# Qdrant batch update
# ---------------------------------------------------------------------------

def _flush_updates(client: Any, collection: str, updates: list[tuple[Any, dict]]) -> None:
    for point_id, fields in updates:
        try:
            client.set_payload(
                collection_name=collection,
                payload=fields,
                points=[point_id],
            )
        except Exception as exc:
            logger.warning("Payload update failed for %s: %s", point_id, exc)


# ---------------------------------------------------------------------------
# Core enrichment loop
# ---------------------------------------------------------------------------

def _enrich_collection(
    client: Any,
    collection: str,
    *,
    dry_run: bool,
    limit: int,
    threshold: float,
    delay_s: float,
    mailto: str,
    checkpoint: set[str],
    checkpoint_path: str,
    checkpoint_every: int = 50,
) -> dict[str, int]:
    stats = {
        "scanned": 0, "skipped_has_link": 0, "skipped_checkpoint": 0,
        "no_title": 0, "no_match": 0, "low_confidence": 0,
        "updated": 0, "would_update": 0, "errors": 0,
    }
    pending: list[tuple[Any, dict[str, Any]]] = []
    offset = None

    while True:
        batch_limit = min(256, limit - stats["scanned"]) if limit else 256
        points, offset = client.scroll(
            collection, limit=batch_limit, with_payload=True, offset=offset,
        )
        if not points:
            break

        for p in points:
            stats["scanned"] += 1
            pid = str(p.id)
            payload = p.payload or {}

            if _has_any(payload, _LINK_FIELDS):
                stats["skipped_has_link"] += 1
                continue
            if pid in checkpoint:
                stats["skipped_checkpoint"] += 1
                continue

            title = _title_of(payload)
            if not title or len(title) < 10:
                stats["no_title"] += 1
                checkpoint.add(pid)
                continue

            cr_item = _crossref_lookup(title, mailto=mailto, delay_s=delay_s)
            checkpoint.add(pid)

            if not cr_item:
                stats["no_match"] += 1
                continue

            cr_title = _crossref_title(cr_item)
            sim = _title_similarity(title, cr_title) if cr_title else 0.0
            if sim < threshold:
                stats["low_confidence"] += 1
                continue

            fields = _extract_crossref_fields(cr_item)
            if not fields.get("doi"):
                stats["no_match"] += 1
                continue

            if dry_run:
                stats["would_update"] += 1
                logger.info("[DRY RUN] %s | sim=%.2f | doi=%s | %r",
                            pid, sim, fields.get("doi"), title[:70])
            else:
                pending.append((p.id, fields))
                stats["updated"] += 1
                logger.info("Queued %s | sim=%.2f | doi=%s | %r",
                            pid, sim, fields.get("doi"), title[:60])
                if len(pending) >= _BATCH_SIZE:
                    _flush_updates(client, collection, pending)
                    pending.clear()

            if stats["scanned"] % checkpoint_every == 0:
                _save_checkpoint(checkpoint_path, checkpoint)
                logger.info("Progress: %d scanned, %d updated/%d would",
                            stats["scanned"], stats["updated"], stats["would_update"])

        if offset is None or (limit and stats["scanned"] >= limit):
            break

    if not dry_run and pending:
        _flush_updates(client, collection, pending)
    _save_checkpoint(checkpoint_path, checkpoint)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated bibliography enrichment via CrossRef")
    parser.add_argument("--apply", action="store_true",
                        help="Write updates to Qdrant (default: dry-run)")
    parser.add_argument("--collections", nargs="+", default=list(RESEARCH_COLLECTIONS),
                        choices=list(RESEARCH_COLLECTIONS))
    parser.add_argument("--limit", type=int, default=0,
                        help="Max points per collection (0=all)")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                        help="Title similarity threshold 0-1 (default 0.85)")
    parser.add_argument("--delay", type=float, default=_DEFAULT_DELAY_S,
                        help="Seconds between CrossRef requests (default 1.0)")
    parser.add_argument("--checkpoint", default="data/local/enrich_checkpoint.json")
    parser.add_argument("--mailto", default="",
                        help="Email for CrossRef polite pool (or ENRICH_MAILTO env)")
    args = parser.parse_args()

    dry_run = not args.apply
    mailto = args.mailto or os.environ.get("ENRICH_MAILTO", "team@opentrace.africa")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("ERROR: pip install qdrant-client")
        sys.exit(1)

    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    if not qdrant_url:
        print("ERROR: QDRANT_URL not set in config/.env")
        sys.exit(1)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
    checkpoint = _load_checkpoint(args.checkpoint)

    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"\n{'='*58}")
    print(f"Bibliography Enrichment via CrossRef -- {mode}")
    print(f"{'='*58}")
    print(f"Collections: {args.collections}")
    print(f"Threshold:   {args.threshold}  |  Delay: {args.delay}s  |  Limit: {args.limit or 'all'}")
    print(f"Checkpoint:  {args.checkpoint} ({len(checkpoint)} already seen)")
    if dry_run:
        print("\nDRY RUN -- no Qdrant writes. Pass --apply to update.")
    print("")

    grand: dict[str, int] = {}
    for col in args.collections:
        print(f"\n--- {col} ---")
        try:
            stats = _enrich_collection(
                client, col,
                dry_run=dry_run, limit=args.limit, threshold=args.threshold,
                delay_s=args.delay, mailto=mailto,
                checkpoint=checkpoint, checkpoint_path=args.checkpoint,
            )
        except Exception as exc:
            logger.error("Collection %s failed: %s", col, exc)
            continue
        for k, v in stats.items():
            grand[k] = grand.get(k, 0) + v
        key = "would_update" if dry_run else "updated"
        print(f"  Scanned:        {stats['scanned']:,}")
        print(f"  Already linked: {stats['skipped_has_link']:,}")
        print(f"  Checkpoint skip:{stats['skipped_checkpoint']:,}")
        print(f"  No title:       {stats['no_title']:,}")
        print(f"  No CR match:    {stats['no_match']:,}")
        print(f"  Low confidence: {stats['low_confidence']:,}")
        print(f"  {key.upper()}: {stats[key]:,}")

    key = "would_update" if dry_run else "updated"
    print(f"\n{'='*58}")
    print(f"TOTAL -- Scanned: {grand.get('scanned',0):,}  |  {key.upper()}: {grand.get(key,0):,}")
    print(f"Checkpoint saved: {args.checkpoint}")
    if dry_run:
        print("\nRe-run with --apply to write to Qdrant.")
    print("")


if __name__ == "__main__":
    main()

