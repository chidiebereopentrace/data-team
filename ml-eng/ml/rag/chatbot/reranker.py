"""
Reranker node: takes merged retrieval results and reranks them before passing
to the generator.

Modes, selected via ``RAG_RERANKER_MODE``:

- ``openrouter`` (recommended when ``RAG_LLM_BASE_URL`` is OpenRouter):
    One ``POST /rerank`` via OpenRouter (default model ``cohere/rerank-4-pro``).
    Reuses ``RAG_LLM_API_KEY``. Model via ``RAG_RERANK_MODEL_ID`` (or legacy
    ``RAG_RERANKER_COHERE_MODEL`` if it looks like an OpenRouter slug).

- ``cohere``:
    Cohere's managed SDK API (``rerank-v3.5``). Requires ``COHERE_API_KEY``.

- ``cross_encoder`` (default when no OpenRouter/Cohere key):
    Local fastembed / sentence-transformers batch pass.

- ``llm`` / ``off``: legacy per-chunk LLM scoring / boost-only pass-through.

Auto-select order when mode is unset: openrouter (if OpenRouter + API key) →
cohere (if Cohere key) → cross_encoder.

Failure handling: never raises. Degrade:
    openrouter → cohere → cross_encoder → llm (if configured) → off
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_model_id
from ml.rag.observability import get_observe_decorator, trace_elapsed_ms, update_current_span_metadata
from ml.rag.rerank_client import (
    openrouter_rerank,
    openrouter_rerank_configured,
    rerank_model_id as _openrouter_rerank_model_id,
)

logger = logging.getLogger(__name__)

_observe_span = get_observe_decorator()

_LAST_RERANK_MODE = "off"

# Static source priority — applied additively on top of any model score so a
# BigQuery row that scores near a news chunk still wins the tie-break (BQ
# rows are higher-trust ground truth than free-text news).
_SOURCE_BOOST: dict[str, float] = {
    "bigquery": 0.12,
    "academic": 0.06,
    "policy": 0.06,
    "public_report": 0.06,
    "formation": 0.05,
    "ota_insight": 0.05,
    "news": 0.04,
    "web_wikipedia": 0.0,
    "web_search": 0.0,
}


def _item_source_boost(item: dict[str, Any]) -> float:
    kind = str(item.get("_context_kind") or item.get("source") or "").lower()
    boost = float(_SOURCE_BOOST.get(kind, 0.0))
    raw_meta = item.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    extra = meta.get("corpus_boost")
    if extra is not None:
        try:
            # corpus_boost already includes default catalog boost; use delta over static
            # only when it exceeds the static map (router raised it).
            extra_f = float(extra)
            if extra_f > boost:
                boost = extra_f
            elif extra_f > 0 and kind not in _SOURCE_BOOST:
                boost = extra_f
            else:
                # Router may add preference on top of default; prefer max of static and stamped.
                boost = max(boost, extra_f)
        except (TypeError, ValueError):
            pass
    if str(meta.get("constraint_relaxed") or "").strip() in ("", "none"):
        boost += 0.05
    return boost


def _passage_with_metadata(item: dict[str, Any], content_key: str, max_chars: int) -> str:
    """Prepend a compact geo/year/domains header so rerankers see constraints."""
    raw = str(item.get(content_key) or item.get("text") or item)
    raw_meta = item.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    geo = str(
        meta.get("geo_country_primary") or meta.get("country") or meta.get("geo_countries") or ""
    ).strip()
    year = str(meta.get("published_at") or meta.get("publication_year") or "").strip()
    if len(year) >= 4 and year[:4].isdigit():
        year = year[:4]
    domains = str(meta.get("domains") or meta.get("domain") or "").strip()
    bits: list[str] = []
    if geo:
        bits.append(f"geo={geo[:80]}")
    if year:
        bits.append(f"year={year}")
    if domains:
        bits.append(f"domains={domains[:120]}")
    body = raw[: max(0, max_chars)]
    if not bits:
        return body
    header = "[" + "; ".join(bits) + "]\n"
    remain = max(0, max_chars - len(header))
    return header + body[:remain]


# --- env helpers --------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _llm_configured() -> bool:
    return bool(os.environ.get("HF_API_TOKEN") or os.environ.get("RAG_LLM_BASE_URL", "").strip())


def _reranker_mode() -> str:
    """Resolve the configured reranker mode with back-compat handling.

    Order of precedence:
    1. ``RAG_RERANKER_MODE`` if explicitly set
       (openrouter | cohere | cross_encoder | llm | off)
    2. Auto ``openrouter`` when OpenRouter base URL + ``RAG_LLM_API_KEY``
    3. Auto ``cohere`` when a Cohere API key is available
    4. Legacy ``RAG_LLM_RERANK`` (truthy -> ``llm``, falsy -> ``off``)
    5. Default: ``cross_encoder``
    """
    explicit = os.environ.get("RAG_RERANKER_MODE", "").strip().lower()
    if explicit in {"openrouter", "cohere", "cross_encoder", "llm", "off"}:
        return explicit
    if openrouter_rerank_configured():
        return "openrouter"
    if _cohere_api_key():
        return "cohere"
    legacy = os.environ.get("RAG_LLM_RERANK", "").strip().lower()
    if legacy in {"on", "1", "true", "yes"}:
        return "llm"
    if legacy in {"off", "0", "false", "no"}:
        logger.info(
            "Legacy RAG_LLM_RERANK=off detected; using mode='off'. "
            "Set RAG_RERANKER_MODE=openrouter or cross_encoder to enable reranking."
        )
        return "off"
    return "cross_encoder"


def _reranker_model_id() -> str:
    return (
        os.environ.get("RAG_RERANKER_MODEL", "").strip()
        or "BAAI/bge-reranker-base"
    )


def _cohere_api_key() -> str | None:
    key = (
        os.environ.get("RAG_RERANKER_COHERE_API_KEY")
        or os.environ.get("COHERE_API_KEY")
        or ""
    ).strip()
    return key or None


def _cohere_model() -> str:
    return (
        os.environ.get("RAG_RERANKER_COHERE_MODEL", "").strip()
        or "rerank-v3.5"
    )


def _openrouter_rerank_model() -> str:
    """Model slug for OpenRouter POST /rerank (e.g. cohere/rerank-4-pro)."""
    explicit = os.environ.get("RAG_RERANK_MODEL_ID", "").strip()
    if explicit:
        return explicit
    # Allow reusing the Cohere model env when it looks like an OpenRouter slug.
    cohere_like = os.environ.get("RAG_RERANKER_COHERE_MODEL", "").strip()
    if "/" in cohere_like:
        return cohere_like
    return _openrouter_rerank_model_id()


def _max_text_chars() -> int:
    """Cap per-chunk characters fed to the cross-encoder.

    Cross-encoders have small context windows (typically 512 tokens for
    MiniLM). 2000 chars is a safe upper bound after the tokenizer truncates
    internally; we set it explicitly so very long news/academic chunks do not
    dominate the model's truncation behavior.
    """
    return _env_int("RAG_RERANKER_MAX_TEXT_CHARS", 2000)


# --- cross-encoder backend ---------------------------------------------------


_ce_lock = threading.Lock()
_ce_cache: dict[str, Any] = {}


def _load_cross_encoder(model_id: str) -> Any | None:
    """Lazy-load and cache a cross-encoder. Tries fastembed first, then
    sentence-transformers. Returns None if neither is available so callers
    can fall back gracefully."""
    with _ce_lock:
        if model_id in _ce_cache:
            return _ce_cache[model_id]

        # 1. fastembed (already used by the project for ONNX embeddings on Railway)
        try:
            from fastembed.rerank.cross_encoder import (  # type: ignore[import-not-found]
                TextCrossEncoder,
            )
        except Exception:  # pragma: no cover - exercised when fastembed missing
            TextCrossEncoder = None  # type: ignore[assignment]

        if TextCrossEncoder is not None:
            try:
                encoder = TextCrossEncoder(model_id)
                _ce_cache[model_id] = ("fastembed", encoder)
                logger.info("Cross-encoder loaded via fastembed: %s", model_id)
                return _ce_cache[model_id]
            except Exception as exc:
                logger.warning(
                    "fastembed cross-encoder %s failed to load: %s", model_id, exc
                )

        # 2. sentence-transformers fallback (dev machines that already have it)
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

            encoder = CrossEncoder(model_id)
            _ce_cache[model_id] = ("sentence_transformers", encoder)
            logger.info(
                "Cross-encoder loaded via sentence-transformers: %s", model_id
            )
            return _ce_cache[model_id]
        except Exception as exc:
            logger.warning(
                "sentence-transformers cross-encoder %s unavailable: %s",
                model_id,
                exc,
            )

        _ce_cache[model_id] = None
        return None


def _ce_score(encoder: tuple[str, Any], query: str, passages: list[str]) -> list[float] | None:
    """Score a batch of (query, passage) pairs.

    Returns ``None`` on backend error so the caller can fall back.
    """
    backend, model = encoder
    try:
        if backend == "fastembed":
            # fastembed TextCrossEncoder.rerank returns a generator of floats
            return [float(s) for s in model.rerank(query, passages)]
        # sentence-transformers CrossEncoder.predict accepts list[tuple]
        scored = model.predict([(query, p) for p in passages])
        return [float(s) for s in scored]
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed: %s", exc)
        return None


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1] so the source boost has a comparable scale.

    Cross-encoders emit logits that can be wildly different ranges between
    models (ms-marco MiniLM emits roughly [-10, 10]; BGE rerankers emit [0, 1]
    already). Without normalization the static source boost would either be
    drowned out or completely dominate.
    """
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _rerank_cross_encoder(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int,
    content_key: str,
) -> list[dict[str, Any]] | None:
    """Cross-encoder reranking. Returns ``None`` if the backend is unavailable
    so the public ``rerank()`` can degrade to LLM or off."""
    model_id = _reranker_model_id()
    encoder = _load_cross_encoder(model_id)
    if encoder is None:
        return None

    max_chars = _max_text_chars()
    passages = [
        _passage_with_metadata(item, content_key, max_chars)
        for item in context_items
    ]
    raw_scores = _ce_score(encoder, query, passages)
    if raw_scores is None or len(raw_scores) != len(context_items):
        return None

    norm = _normalize_scores(raw_scores)
    scored = []
    for i, (item, raw, ns) in enumerate(zip(context_items, raw_scores, norm)):
        boost = _item_source_boost(item)
        scored.append(
            {
                **item,
                "content": passages[i],
                "_order": i,
                "_ce_score_raw": raw,
                "_ce_score": ns,
                "_source_boost": boost,
                "_rerank_score": ns + boost,
            }
        )
    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return scored[:top_k]


# --- llm backend (legacy, kept for back-compat) -------------------------------


def _score_with_llama(query: str, text: str) -> float:
    """Ask the configured LLM to score one (query, text) pair on [0, 1]."""
    prompt = (
        "You are a ranking model. Given a user question and a context chunk, "
        "return a single floating point number between 0 and 1 indicating how "
        "relevant the context is to answering the question. Respond with ONLY "
        "the number.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{text}\n\n"
        "Relevance score (0-1):"
    )
    raw = llm_chat_complete(
        [{"role": "user", "content": prompt}],
        model=llm_model_id(),
        max_tokens=8,
        temperature=0.0,
        timeout_s=30,
    )
    if not raw:
        return -1.0
    try:
        return float(raw.split()[0])
    except Exception:
        return -1.0


def _rerank_llm(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int,
    content_key: str,
) -> list[dict[str, Any]]:
    max_chars = _max_text_chars()
    scored = []
    for i, item in enumerate(context_items):
        text = _passage_with_metadata(item, content_key, max_chars)
        raw = _score_with_llama(query, text)
        boost = _item_source_boost(item)
        adjusted = (raw + boost) if raw >= 0 else raw
        scored.append(
            {
                **item,
                "content": text,
                "_order": i,
                "_llm_score": raw,
                "_source_boost": boost,
                "_rerank_score": adjusted,
            }
        )
    scored.sort(
        key=lambda x: x.get("_rerank_score", x.get("_llm_score", -1.0)),
        reverse=True,
    )
    return scored[:top_k]


# --- off (pass-through) -------------------------------------------------------


def _rerank_off(
    context_items: list[dict[str, Any]],
    top_k: int,
    content_key: str,
) -> list[dict[str, Any]]:
    """Pass-through ordering using only the static source boost.

    Kept for dev / debugging — the failure mode internal testing surfaced was
    that no real reranking was happening at all, so operators should not run
    production on this mode.
    """
    scored = []
    for i, item in enumerate(context_items):
        text = _passage_with_metadata(item, content_key, _max_text_chars())
        boost = _item_source_boost(item)
        scored.append(
            {
                **item,
                "content": text,
                "_order": i,
                "_llm_score": -1.0,
                "_source_boost": boost,
                "_rerank_score": boost,
            }
        )
    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return scored[:top_k]


# --- OpenRouter batch reranker ------------------------------------------------


def _rerank_openrouter(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int,
    content_key: str,
) -> list[dict[str, Any]] | None:
    """Rerank via OpenRouter ``POST /rerank`` (e.g. ``cohere/rerank-4-pro``).

    Returns ``None`` when unconfigured or the API returns nothing so the caller
    can degrade to Cohere SDK / cross_encoder.
    """
    if not openrouter_rerank_configured():
        return None

    max_chars = _max_text_chars()
    passages = [
        _passage_with_metadata(item, content_key, max_chars)
        for item in context_items
    ]
    model_name = _openrouter_rerank_model()
    hits = openrouter_rerank(query, passages, top_n=top_k, model=model_name)
    if not hits:
        return None

    scored: list[dict[str, Any]] = []
    for idx, relevance in hits:
        if idx < 0 or idx >= len(context_items):
            continue
        item = context_items[idx]
        boost = _item_source_boost(item)
        scored.append(
            {
                **item,
                "content": passages[idx],
                "_order": idx,
                "_openrouter_score": relevance,
                "_source_boost": boost,
                "_rerank_score": relevance + boost,
            }
        )
    if not scored:
        return None
    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    logger.info(
        "OpenRouter rerank ok: model=%s query_len=%d candidates=%d top_k=%d",
        model_name,
        len(query),
        len(context_items),
        len(scored),
    )
    return scored


# --- Cohere managed reranker --------------------------------------------------


def _rerank_cohere(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int,
    content_key: str,
) -> list[dict[str, Any]] | None:
    """Rerank via the Cohere managed API.

    Returns ``None`` when Cohere is unavailable (no key, import error, or
    API failure) so the caller can degrade to the next backend.

    Cohere's relevance_score is already in [0, 1] and encodes both position
    and semantic relevance, so we skip the min-max normalisation step and add
    the static source boost directly on top.
    """
    key = _cohere_api_key()
    if not key:
        return None

    try:
        import cohere as _cohere  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "Cohere package not installed. "
            "Run: pip install cohere   (or add it to requirements.txt)"
        )
        return None

    max_chars = _max_text_chars()
    passages = [
        _passage_with_metadata(item, content_key, max_chars)
        for item in context_items
    ]

    model_name = _cohere_model()
    try:
        co = _cohere.ClientV2(api_key=key)
        response = co.rerank(
            model=model_name,
            query=query,
            documents=passages,
            top_n=top_k,
            return_documents=False,
        )
    except Exception as exc:
        logger.warning("Cohere rerank API call failed: %s", exc)
        return None

    results = getattr(response, "results", None) or []
    if not results:
        logger.warning("Cohere returned empty results for query=%r", query[:80])
        return None

    scored = []
    for hit in results:
        idx = int(hit.index)
        relevance = float(getattr(hit, "relevance_score", 0.0))
        item = context_items[idx]
        boost = _item_source_boost(item)
        scored.append(
            {
                **item,
                "content": passages[idx],
                "_order": idx,
                "_cohere_score": relevance,
                "_source_boost": boost,
                "_rerank_score": relevance + boost,
            }
        )

    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    logger.info(
        "Cohere rerank ok: model=%s query_len=%d candidates=%d top_k=%d",
        model_name,
        len(query),
        len(context_items),
        len(scored),
    )
    return scored


# --- public entry point -------------------------------------------------------


def _finalize_rerank_trace(
    items: list[dict[str, Any]],
    *,
    mode: str,
    input_count: int,
    top_k: int,
    start: float,
    model: str | None = None,
) -> list[dict[str, Any]]:
    global _LAST_RERANK_MODE
    _LAST_RERANK_MODE = mode
    meta: dict[str, Any] = {
        "mode": mode,
        "input_count": input_count,
        "output_count": len(items),
        "top_k": top_k,
        "latency_ms": trace_elapsed_ms(start),
    }
    if model:
        meta["model"] = model
    if items:
        top = items[0].get("_rerank_score")
        if top is not None:
            try:
                meta["top_rerank_score"] = float(top)
            except (TypeError, ValueError):
                pass
    update_current_span_metadata(meta)
    return items


def last_rerank_mode() -> str:
    """Effective rerank backend used on the most recent ``rerank()`` call."""
    return _LAST_RERANK_MODE


@_observe_span(as_type="span", name="rerank", capture_input=False, capture_output=False)
def rerank(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Rerank ``context_items`` by relevance to ``query``.

    Each item should expose ``content`` (or ``text``) and optionally
    ``_context_kind`` / ``source`` so the static source boost applies.

    Mode is selected by :func:`_reranker_mode`. When the chosen backend is
    unavailable at runtime the function degrades safely:
        openrouter (error) -> cohere -> cross_encoder -> llm if configured -> off
        cohere (no key / error) -> cross_encoder -> llm if configured -> off
        cross_encoder unavailable -> llm if configured -> off
        llm unavailable           -> off

    Top-k cap respected from the ``top_k`` argument and (when set) the
    ``RAG_RERANKER_TOP_K`` env override.
    """
    t0 = time.perf_counter()
    if not context_items:
        return _finalize_rerank_trace([], mode="off", input_count=0, top_k=top_k, start=t0)

    env_top_k = _env_int("RAG_RERANKER_TOP_K", 0)
    if env_top_k > 0:
        top_k = min(top_k, env_top_k)

    content_key = "content" if any("content" in c for c in context_items) else "text"
    mode = _reranker_mode()
    input_count = len(context_items)
    logger.debug("Reranker mode=%s top_k=%d candidates=%d", mode, top_k, input_count)

    if mode == "openrouter":
        result = _rerank_openrouter(query, context_items, top_k, content_key)
        if result is not None:
            return _finalize_rerank_trace(
                result,
                mode="openrouter",
                input_count=input_count,
                top_k=top_k,
                start=t0,
                model=_openrouter_rerank_model(),
            )
        logger.warning(
            "OpenRouter reranker unavailable; degrading to cohere/cross_encoder."
        )
        mode = "cohere" if _cohere_api_key() else "cross_encoder"

    if mode == "cohere":
        result = _rerank_cohere(query, context_items, top_k, content_key)
        if result is not None:
            return _finalize_rerank_trace(
                result,
                mode="cohere",
                input_count=input_count,
                top_k=top_k,
                start=t0,
                model=_cohere_model(),
            )
        logger.warning(
            "Cohere reranker unavailable; degrading to cross_encoder."
        )
        mode = "cross_encoder"

    if mode == "cross_encoder":
        result = _rerank_cross_encoder(query, context_items, top_k, content_key)
        if result is not None:
            return _finalize_rerank_trace(
                result,
                mode="cross_encoder",
                input_count=input_count,
                top_k=top_k,
                start=t0,
                model=_env("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            )
        logger.warning(
            "Cross-encoder unavailable; degrading. Install fastembed or "
            "sentence-transformers and set RAG_RERANKER_MODEL accordingly."
        )
        mode = "llm" if _llm_configured() else "off"

    if mode == "llm":
        if not _llm_configured():
            logger.warning(
                "RAG_RERANKER_MODE=llm but no LLM backend configured; "
                "degrading to off."
            )
            return _finalize_rerank_trace(
                _rerank_off(context_items, top_k, content_key),
                mode="off",
                input_count=input_count,
                top_k=top_k,
                start=t0,
            )
        return _finalize_rerank_trace(
            _rerank_llm(query, context_items, top_k, content_key),
            mode="llm",
            input_count=input_count,
            top_k=top_k,
            start=t0,
            model=llm_model_id(),
        )

    return _finalize_rerank_trace(
        _rerank_off(context_items, top_k, content_key),
        mode="off",
        input_count=input_count,
        top_k=top_k,
        start=t0,
    )
