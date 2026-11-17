"""
Dense text embeddings via fastembed (ONNX, no torch).

Used on Railway and other slim deployments where ``sentence_transformers`` is absent.
Supports ``intfloat/multilingual-e5-small`` (384-dim, matches Qdrant ingest).
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from ml.rag.observability import observed_span, trace_elapsed_ms, update_current_span_metadata

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"

_EMBED_CACHE_TTL_S = float(os.environ.get("RAG_EMBED_CACHE_TTL_S", "60") or 60)
_EMBED_CACHE_MAX = max(16, int(os.environ.get("RAG_EMBED_CACHE_MAX", "256") or 256))
_embed_cache: OrderedDict[tuple[str, str], tuple[float, list[float]]] = OrderedDict()


def _embed_cache_get(key: tuple[str, str]) -> list[float] | None:
    entry = _embed_cache.get(key)
    if entry is None:
        return None
    ts, vec = entry
    if time.monotonic() - ts > _EMBED_CACHE_TTL_S:
        _embed_cache.pop(key, None)
        return None
    _embed_cache.move_to_end(key)
    return vec


def _embed_cache_put(key: tuple[str, str], vec: list[float]) -> None:
    _embed_cache[key] = (time.monotonic(), vec)
    _embed_cache.move_to_end(key)
    while len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)

_E5_MODEL_DIMS = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
}


def _register_e5_model(model_id: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=model_id,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_id),
        dim=_E5_MODEL_DIMS[model_id],
        model_file="onnx/model.onnx",
    )


@lru_cache(maxsize=4)
def _text_embedding(model_id: str) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "Dense fastembed embeddings require fastembed. Install: pip install fastembed"
        ) from exc

    mid = (model_id or DEFAULT_DENSE_MODEL).strip() or DEFAULT_DENSE_MODEL
    logger.info("Loading fastembed TextEmbedding %r (one-time)", mid)
    if mid in _E5_MODEL_DIMS and not any(info["model"] == mid for info in TextEmbedding.list_supported_models()):
        _register_e5_model(mid)
    return TextEmbedding(model_name=mid)


def embed_dense_texts(texts: list[str], *, model_id: str) -> list[list[float]]:
    """Return L2-normalized dense vectors for a batch of texts."""
    t0 = time.perf_counter()
    if not texts:
        return []
    mid = (model_id or DEFAULT_DENSE_MODEL).strip() or DEFAULT_DENSE_MODEL
    cache_key: tuple[str, str] | None = None
    if len(texts) == 1:
        cache_key = (mid, (texts[0] or "").strip())
        if cache_key[1]:
            cached = _embed_cache_get(cache_key)
            if cached is not None:
                update_current_span_metadata(
                    {
                        "model_id": mid,
                        "mode": "fastembed",
                        "batch_size": 1,
                        "embed_cache_hit": True,
                        "latency_ms": trace_elapsed_ms(t0),
                    }
                )
                return [cached]
    span_ctx = observed_span(
        "embedding.query",
        input_data={"model_id": mid, "mode": "fastembed", "batch_size": len(texts)},
    )
    with span_ctx:
        model = _text_embedding(mid)
        cleaned = [(t or "").strip() or " " for t in texts]
        out: list[list[float]] = []
        for emb in model.embed(cleaned):
            out.append([float(x) for x in emb])
        if len(texts) == 1 and out and cache_key and cache_key[1]:
            _embed_cache_put(cache_key, out[0])
        update_current_span_metadata(
            {
                "model_id": mid,
                "mode": "fastembed",
                "batch_size": len(texts),
                "latency_ms": trace_elapsed_ms(t0),
            }
        )
    return out
