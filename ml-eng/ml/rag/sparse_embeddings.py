"""
BM25 sparse vectors for Qdrant hybrid search (via fastembed).

Env:
  RAG_SPARSE_EMBEDDINGS=on|off     (default on) — populate sparse vectors on upsert
  RAG_QDRANT_HYBRID_SEARCH=on|off  (default on) — dense+sparse RRF at query time
  RAG_SPARSE_MODEL=Qdrant/bm25     (default)
  RAG_HYBRID_DENSE_PREFETCH=20     — dense prefetch limit per vector (news/research/OTA)
  RAG_HYBRID_SPARSE_PREFETCH=20    — sparse prefetch limit per vector
  RAG_HYBRID_FUSION_LIMIT=20       — post-RRF result cap (before client-side trim to top_k)
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ml.rag.observability import get_observe_decorator, trace_elapsed_ms, update_current_span_metadata

if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

_observe_span = get_observe_decorator()


def _flag(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name, "on" if default else "off").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def sparse_embeddings_enabled() -> bool:
    return _flag("RAG_SPARSE_EMBEDDINGS", default=True)


def hybrid_search_enabled() -> bool:
    return _flag("RAG_QDRANT_HYBRID_SEARCH", default=True)


def sparse_model_name() -> str:
    return os.environ.get("RAG_SPARSE_MODEL", DEFAULT_SPARSE_MODEL).strip() or DEFAULT_SPARSE_MODEL


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def hybrid_dense_prefetch_limit() -> int:
    return _env_int("RAG_HYBRID_DENSE_PREFETCH", 20)


def hybrid_sparse_prefetch_limit() -> int:
    return _env_int("RAG_HYBRID_SPARSE_PREFETCH", 20)


def hybrid_fusion_limit(*, top_k: int = 20) -> int:
    """Post-RRF cap; at least top_k so callers always receive enough candidates."""
    return max(top_k, _env_int("RAG_HYBRID_FUSION_LIMIT", 20))


@lru_cache(maxsize=1)
def _sparse_model() -> Any:
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise ImportError(
            "Sparse embeddings require fastembed. Install: pip install fastembed"
        ) from exc
    return SparseTextEmbedding(model_name=sparse_model_name())


def _to_sparse_vector(emb: Any) -> SparseVector:
    from qdrant_client.http.models import SparseVector

    indices = getattr(emb, "indices", None)
    values = getattr(emb, "values", None)
    if indices is None or values is None:
        raise RuntimeError("Unexpected sparse embedding payload from fastembed")
    return SparseVector(
        indices=[int(i) for i in indices.tolist()],
        values=[float(v) for v in values.tolist()],
    )


def embed_sparse_documents(texts: list[str]) -> list[SparseVector]:
    """BM25 sparse vectors for document/passage indexing."""
    if not texts:
        return []
    model = _sparse_model()
    cleaned = [(t or "").strip() or " " for t in texts]
    return [_to_sparse_vector(e) for e in model.embed(cleaned)]


@_observe_span(as_type="span", name="embedding.query", capture_input=False, capture_output=False)
def embed_sparse_query(text: str) -> SparseVector:
    """BM25 sparse vector for a search query."""
    t0 = time.perf_counter()
    model = _sparse_model()
    q = (text or "").strip() or " "
    emb = next(iter(model.query_embed(q)))
    result = _to_sparse_vector(emb)
    update_current_span_metadata(
        {
            "model_id": sparse_model_name(),
            "mode": "fastembed_sparse",
            "batch_size": 1,
            "latency_ms": trace_elapsed_ms(t0),
        }
    )
    return result


def warmup_sparse_model() -> None:
    """Download/load BM25 model into FASTEMBED_CACHE_PATH (Docker build or startup)."""
    embed_sparse_query("warmup")
