"""
Dense text embeddings via fastembed (ONNX, no torch).

Used on Railway and other slim deployments where ``sentence_transformers`` is absent.
Supports ``intfloat/multilingual-e5-small`` (384-dim, matches Qdrant ingest).
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from ml.rag.observability import observed_span, trace_elapsed_ms, update_current_span_metadata

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"

_E5_SMALL = "intfloat/multilingual-e5-small"
_E5_REGISTERED = False


def _ensure_e5_small_registered() -> None:
    global _E5_REGISTERED
    if _E5_REGISTERED:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=_E5_SMALL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=_E5_SMALL),
        dim=384,
        model_file="onnx/model.onnx",
    )
    _E5_REGISTERED = True


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
    try:
        return TextEmbedding(model_name=mid)
    except Exception:
        if mid == _E5_SMALL or "multilingual-e5-small" in mid.lower():
            _ensure_e5_small_registered()
            return TextEmbedding(model_name=mid)
        raise


def embed_dense_texts(texts: list[str], *, model_id: str) -> list[list[float]]:
    """Return L2-normalized dense vectors for a batch of texts."""
    t0 = time.perf_counter()
    if not texts:
        return []
    span_ctx = observed_span(
        "embedding.query",
        input_data={"model_id": model_id, "mode": "fastembed", "batch_size": len(texts)},
    )
    with span_ctx:
        model = _text_embedding(model_id)
        cleaned = [(t or "").strip() or " " for t in texts]
        out: list[list[float]] = []
        for emb in model.embed(cleaned):
            out.append([float(x) for x in emb])
        update_current_span_metadata(
            {
                "model_id": model_id,
                "mode": "fastembed",
                "batch_size": len(texts),
                "latency_ms": trace_elapsed_ms(t0),
            }
        )
    return out


def warmup_dense_model(model_id: str | None = None) -> None:
    """Download/load ONNX weights into FASTEMBED_CACHE_PATH (Docker build or startup)."""
    mid = (model_id or DEFAULT_DENSE_MODEL).strip() or DEFAULT_DENSE_MODEL
    vecs = embed_dense_texts(["warmup"], model_id=mid)
    if not vecs or len(vecs[0]) != 384:
        raise RuntimeError(f"fastembed warmup failed for {mid!r}")
    logger.info("fastembed dense model ready: %r (dim=%d)", mid, len(vecs[0]))
