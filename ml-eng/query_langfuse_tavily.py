"""
Tavily web fallback hit-rate analysis from Langfuse traces (ML-045).

Pulls all traces from the last N days and prints a distribution of
web_fallback_status values, answering: "How often does Tavily actually
fire?" — directly validates the $30/mo Tavily Project budget in v7/v10.

Usage:
    cd data-team/ml-eng
    python query_langfuse_tavily.py              # last 30 days
    python query_langfuse_tavily.py --days 7
    python query_langfuse_tavily.py --days 90

Requires: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
in config/.env or as environment variables.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / "config" / ".env"
    if not env_path.exists():
        return
    import os
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _creds() -> tuple[str, str, str]:
    import os
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    base = (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "").strip().rstrip("/")
    if not pk or not sk:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required.")
        sys.exit(1)
    return pk, sk, base or "https://cloud.langfuse.com"


def _get_traces(pk: str, sk: str, base: str, *, from_dt: datetime) -> list[dict]:
    auth = "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()
    from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    all_traces: list[dict] = []
    page = 1
    while True:
        url = f"{base}/api/public/traces?page={page}&limit=100&fromTimestamp={from_str}"
        req = urllib.request.Request(url, headers={"Authorization": auth, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"ERROR: Langfuse API {e.code}: {e.reason}")
            if e.code == 401:
                print("       Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.")
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: Cannot reach {base}: {exc}")
            sys.exit(1)
        items = data.get("data") or []
        if not items:
            break
        all_traces.extend(items)
        meta = data.get("meta") or {}
        total_pages = int(meta.get("totalPages") or 1)
        print(f"  Page {page}/{total_pages} — {len(all_traces)} traces...", end="\r")
        if page >= total_pages:
            break
        page += 1
    return all_traces


def _web_status(trace: dict) -> str | None:
    """Extract web_fallback_status from trace metadata or nested summary."""
    meta = trace.get("metadata") or {}
    if isinstance(meta, dict):
        s = meta.get("web_fallback_status")
        if s:
            return str(s).lower().strip()
        summary = meta.get("summary") or {}
        if isinstance(summary, dict):
            s = summary.get("web_fallback_status")
            if s:
                return str(s).lower().strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Tavily hit-rate from Langfuse traces")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    _load_env()
    pk, sk, base = _creds()
    from_dt = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f"\nQuerying Langfuse: {base}")
    print(f"Period: last {args.days} days (from {from_dt.strftime('%Y-%m-%d')})")
    print("Fetching traces...")

    traces = _get_traces(pk, sk, base, from_dt=from_dt)
    print(f"\nTotal traces fetched: {len(traces)}")

    if not traces:
        print("No traces found. Check keys and date range.")
        return

    shortcut_n = 0
    status_ctr: Counter[str] = Counter()
    no_status_n = 0

    for t in traces:
        if _is_shortcut(t):
            shortcut_n += 1
            continue
        s = _web_status(t)
        if s:
            status_ctr[s] += 1
        else:
            no_status_n += 1

    retrieval_n = len(traces) - shortcut_n
    tavily_fired = status_ctr.get("ok", 0) + status_ctr.get("empty", 0)
    web_fired = tavily_fired + status_ctr.get("rate_limited", 0) + status_ctr.get("error", 0)
    skipped = status_ctr.get("skipped", 0)

    print("\n" + "=" * 58)
    print("TAVILY WEB FALLBACK HIT-RATE REPORT")
    print("=" * 58)
    print(f"Period:              last {args.days} days")
    print(f"Total traces:        {len(traces)}")
    print(f"  Shortcuts (meta/product): {shortcut_n}")
    print(f"  Retrieval traces:  {retrieval_n}")
    print(f"  No web_status tag: {no_status_n}  (pre-ML-027 or no-op traces)")
    print("")
    print("web_fallback_status distribution (retrieval traces):")
    for status, count in sorted(status_ctr.items(), key=lambda x: -x[1]):
        pct = (count / retrieval_n * 100) if retrieval_n else 0
        bar = "#" * int(pct / 2)
        print(f"  {status:<15} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("")
    if retrieval_n > 0:
        web_rate   = web_fired   / retrieval_n * 100
        tavily_pct = tavily_fired / retrieval_n * 100
        skip_pct   = skipped      / retrieval_n * 100
        print(f"Web fallback invoked:       {web_fired}/{retrieval_n}  ({web_rate:.1f}%)")
        print(f"  Tavily fired (ok+empty):  {tavily_fired}  ({tavily_pct:.1f}%)")
        print(f"  Reranker sufficient:      {skipped}  ({skip_pct:.1f}%)")
        print("")
        daily  = tavily_fired / args.days
        monthly = daily * 30
        print("COST MODEL CHECK (Tavily Project ~4,000 credits/mo):")
        print(f"  Est. Tavily calls/day:    {daily:.1f}")
        print(f"  Est. Tavily calls/month:  {monthly:.0f}")
        if monthly <= 4000:
            print("  STATUS: OK  — fits Tavily Project (~$30/mo, 4k credits)")
        elif monthly <= 10000:
            print("  STATUS: WARNING — exceeds Project; Bootstrap ~$100/mo needed")
        else:
            print("  STATUS: ALERT   — exceeds Bootstrap; needs Growth-class plan")
    else:
        print("Not enough retrieval traces to compute a rate.")

    print("=" * 58)
    print("\nNote: 'skipped' = reranker gave enough context, Tavily NOT called.")
    print("Higher skip% = better reranker = lower Tavily cost.")
    print("")


if __name__ == "__main__":
    main()
