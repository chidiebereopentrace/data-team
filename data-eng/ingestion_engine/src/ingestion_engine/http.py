"""Bounded HTTPS GET. Redirects stay on HTTPS. Response bodies are not logged."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from urllib.parse import urljoin

import requests

RETRY_STATUSES = frozenset({202, 429, 500, 502, 503, 504})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
USER_AGENT = "OpenTrace-ingestion-engine/0.1"


class HttpError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _read_limited(response: requests.Response, max_bytes: int) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=1_048_576) as handle:
        size = 0
        for chunk in response.iter_content(65_536):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                raise HttpError("response exceeds the size limit")
            handle.write(chunk)
        handle.seek(0)
        return handle.read()


def _waf_challenge(response: requests.Response) -> bool:
    return (response.headers.get("x-amzn-waf-action") or "").strip().lower() == "challenge"


def max_response_bytes() -> int:
    raw = os.environ.get("INGEST_MAX_BYTES", "").strip()
    if not raw:
        return 512_000_000
    return int(raw)


def request_bytes(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    max_bytes: int | None = None,
    max_retries: int = 3,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET url, retry network errors and 429/5xx, and follow HTTPS redirects."""
    if max_bytes is None:
        max_bytes = max_response_bytes()
    sess = session or requests.Session()
    if session is None:
        sess.headers["User-Agent"] = USER_AGENT
    current = url
    sent_params = params
    for _hop in range(6):
        if not current.startswith("https://"):
            raise HttpError("only https URLs are fetched")
        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(max_retries + 1):
            try:
                response = sess.get(
                    current,
                    params=sent_params,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                last_error = exc
                response = None
                if attempt >= max_retries:
                    raise HttpError("request failed after retries") from exc
                sleep_fn(min(8.0, 0.5 * (2**attempt)))
                continue
            if response.status_code in RETRY_STATUSES and not _waf_challenge(response) and attempt < max_retries:
                response.close()
                sleep_fn(min(5.0, 0.5 * (2**attempt)))
                response = None
                continue
            break
        if response is None:
            raise HttpError(f"request failed: {last_error}")
        try:
            if _waf_challenge(response):
                raise HttpError(
                    "HTTP 202 WAF challenge; response is not the file",
                    status_code=202,
                )
            if response.status_code == 202:
                raise HttpError("HTTP 202", status_code=202)
            if response.status_code in REDIRECTS:
                location = response.headers.get("Location") or ""
                current = urljoin(current, location)
                sent_params = None
                continue
            if response.status_code >= 400:
                raise HttpError(f"HTTP {response.status_code}", status_code=response.status_code)
            return _read_limited(response, max_bytes)
        finally:
            response.close()
    raise HttpError("too many redirects")
