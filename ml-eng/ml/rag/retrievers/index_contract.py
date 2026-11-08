"""Vector index contract: a collection is queried only when its stored layout matches the query embedder.

The expected layout comes from the same spec builders that create collections
(``scripts/qdrant_collection_specs.COLLECTION_BUILDERS``) and the corpus chunking profile.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from ml.rag.dense_embeddings import DEFAULT_DENSE_MODEL
from ml.rag.scripts.qdrant_collection_specs import COLLECTION_BUILDERS
from ml.rag.text_processors.chunking_config import INGEST_VERSION, profile_for_collection

try:
    from fastembed import TextEmbedding
except ImportError:  # slim images without the ONNX backend
    TextEmbedding = None

logger = logging.getLogger(__name__)

IndexStatus = Literal["healthy", "unhealthy", "incompatible"]

# Output dimension of the dense models the profiles may name. A model missing here cannot be verified.
MODEL_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

_HEALTH_TTL_S = 300.0
_health_cache: dict[tuple[str, str, str], tuple[float, IndexHealth]] = {}


@dataclass(frozen=True)
class IndexHealth:
    collection: str
    status: IndexStatus
    reasons: tuple[str, ...] = ()

    @property
    def queryable(self) -> bool:
        return self.status != "incompatible"

    def to_dict(self) -> dict[str, Any]:
        return {"collection": self.collection, "status": self.status, "reasons": list(self.reasons)}


def _expected(collection: str) -> tuple[dict[str, Any], set[str], str]:
    profile = profile_for_collection(collection)
    spec = COLLECTION_BUILDERS[profile.corpus]()
    return dict(spec.get("vectors_config") or {}), set((spec.get("sparse_vectors_config") or {}).keys()), profile.embedding_model


def _fastembed_models() -> set[str]:
    if TextEmbedding is None:
        raise ImportError("fastembed")
    return {str(m.get("model")) for m in TextEmbedding.list_supported_models()} | {DEFAULT_DENSE_MODEL}


def _distance_name(raw: Any) -> str:
    return str(getattr(raw, "value", raw) or "").lower()


def check_collection(client: Any, collection: str, *, backend: str) -> IndexHealth:
    """Compare the live collection with the spec layout and the query embedder on ``backend``."""
    expected_dense, expected_sparse, model_id = _expected(collection)
    incompatible: list[str] = []
    degraded: list[str] = []

    query_dim = MODEL_DIMS.get(model_id)
    if query_dim is None:
        degraded.append(f"embedder {model_id} has no recorded dimension")
    if backend == "fastembed":
        try:
            if model_id not in _fastembed_models():
                incompatible.append(f"embedder {model_id} is not available on the fastembed backend")
        except ImportError:
            incompatible.append("fastembed backend selected but fastembed is not installed")

    try:
        info = client.get_collection(collection_name=collection)
    except Exception as exc:
        return IndexHealth(collection, "incompatible", (f"collection unavailable: {str(exc)[:160]}",))
    params = info.config.params
    live_dense = params.vectors if isinstance(params.vectors, dict) else {"": params.vectors}
    live_sparse = set((params.sparse_vectors or {}).keys())

    for name, spec in expected_dense.items():
        live = live_dense.get(name)
        if live is None:
            incompatible.append(f"dense vector '{name}' missing")
            continue
        if query_dim is not None and int(live.size) != query_dim:
            incompatible.append(f"dense vector '{name}' has dim {live.size}; {model_id} embeds {query_dim}")
        elif int(live.size) != int(spec.size):
            degraded.append(f"dense vector '{name}' has dim {live.size}; spec says {spec.size}")
        if _distance_name(live.distance) != _distance_name(spec.distance):
            incompatible.append(f"dense vector '{name}' uses {live.distance}; spec says {spec.distance}")
    missing_sparse = sorted(expected_sparse - live_sparse)
    if missing_sparse:
        degraded.append(f"sparse vectors missing: {', '.join(missing_sparse)} (hybrid unavailable)")

    try:
        points, _next = client.scroll(collection_name=collection, limit=1, with_payload=["ingest_version"])
    except Exception as exc:
        points = []
        degraded.append(f"payload sample failed: {str(exc)[:120]}")
    if points:
        stored = str((points[0].payload or {}).get("ingest_version") or "")
        if stored != INGEST_VERSION:
            degraded.append(f"ingest_version {stored or 'missing'}; code expects {INGEST_VERSION}")

    if incompatible:
        return IndexHealth(collection, "incompatible", tuple(incompatible + degraded))
    return IndexHealth(collection, "unhealthy" if degraded else "healthy", tuple(degraded))


def collection_health(client: Any, collection: str, *, backend: str) -> IndexHealth:
    """Cached ``check_collection`` (per process, a few minutes) so queries do not re-probe the index."""
    key = (str(id(client)), collection, backend)
    now = time.monotonic()
    cached = _health_cache.get(key)
    if cached is not None and now - cached[0] < _HEALTH_TTL_S:
        return cached[1]
    health = check_collection(client, collection, backend=backend)
    if health.status != "healthy":
        logger.warning("vector index %s is %s: %s", collection, health.status, "; ".join(health.reasons))
    _health_cache[key] = (now, health)
    return health


def clear_health_cache() -> None:
    _health_cache.clear()


__all__ = ["IndexHealth", "IndexStatus", "MODEL_DIMS", "check_collection", "clear_health_cache", "collection_health"]
