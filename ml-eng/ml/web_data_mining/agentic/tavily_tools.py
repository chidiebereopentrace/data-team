"""
Thin wrappers around langchain-tavily tools (Tavily Extract + Search).

See: https://docs.tavily.com/documentation/integrations/langchain
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # ml/web_data_mining/agentic/tavily_tools.py -> repo root
    return Path(__file__).resolve().parents[3]


def _load_data_local_env() -> None:
    """Load data/local/.env when keys are not already set (same pattern as ml/rag)."""
    env_file = _repo_root() / "data" / "local" / ".env"
    if not env_file.is_file():
        return
    try:
        raw = env_file.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        root = _repo_root()
        load_dotenv(root / "data" / "local" / ".env", override=False)
        load_dotenv(root / ".env", override=False)
        load_dotenv(override=False)
    except ImportError:
        pass


def tavily_api_key() -> str | None:
    _load_data_local_env()
    _load_dotenv_if_present()
    # Official name + common typo (TRAVILY)
    key = (
        os.environ.get("TAVILY_API_KEY")
        or os.environ.get("TRAVILY_API_KEY")
        or ""
    ).strip()
    return key or None


def is_tavily_configured() -> bool:
    return tavily_api_key() is not None


def _plain(text: str, max_len: int = 120_000) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:max_len] if len(t) > max_len else t


# Rate-limit error sentinel.
# Callers can test ``err.startswith(TAVILY_RATE_LIMIT_PREFIX)`` to detect a 429
# / usage-limit response and skip retries (retrying just burns quota).
TAVILY_RATE_LIMIT_PREFIX = "RATE_LIMIT:"


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Best-effort detector for Tavily / langchain-tavily rate-limit signals.

    langchain-tavily raises generic exceptions whose ``str`` form embeds the
    underlying HTTP status and body — that is currently the only stable handle
    we have, so we match on textual markers.
    """
    msg = str(exc).lower()
    markers = (
        "429",
        "too many requests",
        "rate limit",
        "rate-limit",
        "rate_limit",
        "usage limit",
        "usage_limit",
        "quota",
    )
    return any(m in msg for m in markers)


def _wrap_error(exc: BaseException) -> str:
    if _is_rate_limit_error(exc):
        return f"{TAVILY_RATE_LIMIT_PREFIX} {exc}"
    return str(exc)


def tavily_extract_urls(urls: list[str], *, extract_depth: str = "basic") -> tuple[str | None, str | None]:
    """
    Extract main text from publisher URLs via Tavily Extract API.
    Returns (raw_content, error_message).
    """
    key = tavily_api_key()
    if not key:
        return None, "TAVILY_API_KEY not set"
    clean = [u.strip() for u in urls if u.startswith("http") and "news.google.com/rss/articles" not in u.lower()]
    if not clean:
        return None, "no suitable URLs for Tavily extract"
    try:
        from langchain_tavily import TavilyExtract  # type: ignore[import-untyped]
    except ImportError as exc:
        return None, f"langchain-tavily not installed: {exc}"

    os.environ["TAVILY_API_KEY"] = key
    try:
        tool: Any = TavilyExtract(extract_depth=extract_depth)
        out = tool.invoke({"urls": clean[:3]})
    except Exception as exc:
        return None, _wrap_error(exc)

    results = (out or {}).get("results") or []
    for row in results:
        raw = (row.get("raw_content") or "").strip()
        if len(raw) >= 200:
            return raw, None
    return None, "tavily extract returned no usable raw_content"


def tavily_search_news(
    query: str,
    *,
    max_results: int = 3,
    search_depth: str = "basic",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """
    News-biased search. Returns (concatenated_snippet_text, raw_results, error).
    Uses topic='news' per Tavily docs.
    """
    key = tavily_api_key()
    if not key:
        return None, [], "TAVILY_API_KEY not set"
    q = _plain(query, 500)
    if len(q) < 10:
        return None, [], "query too short"
    try:
        from langchain_tavily import TavilySearch  # type: ignore[import-untyped]
    except ImportError as exc:
        return None, [], f"langchain-tavily not installed: {exc}"

    os.environ["TAVILY_API_KEY"] = key
    try:
        tool: Any = TavilySearch(
            max_results=max_results,
            topic="news",
            search_depth=search_depth,
            include_raw_content=False,
        )
        payload: dict[str, Any] = {"query": q}
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        out = tool.invoke(payload)
    except Exception as exc:
        return None, [], _wrap_error(exc)

    results = (out or {}).get("results") or []
    chunks: list[str] = []
    for row in results:
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        u = (row.get("url") or "").strip()
        line = " ".join(x for x in (title, content, u) if x)
        if line:
            chunks.append(line)
    text = "\n\n".join(chunks) if chunks else None
    return text, results, None
