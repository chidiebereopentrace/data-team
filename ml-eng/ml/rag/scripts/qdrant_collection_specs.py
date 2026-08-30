"""
Qdrant collection vector layouts (dense HNSW + INT8 quant + optional sparse).

Used by create_qdrant_collections and loaders on --reset.

Per-corpus dense dimensions come from ``chunking_config.PROFILES`` (384 news/research,
768 OTA by default).
"""
from __future__ import annotations

from typing import Any

from qdrant_client.http import models

from ml.rag.text_processors.chunking_config import PROFILES, ChunkingProfile, CorpusKey

HNSW_M = 4

# Dense + sparse vector counts per point (must match collection builders below).
CORPUS_VECTOR_LAYOUT: dict[CorpusKey, dict[str, int]] = {
    "news": {"dense_vectors": 1, "sparse_vectors": 1},
    "research": {"dense_vectors": 1, "sparse_vectors": 1},
    "ota": {"dense_vectors": 3, "sparse_vectors": 2},
}

# Sparse vector name → text source for BM25 ("doc" = chunk text; else metadata key).
CORPUS_SPARSE_FIELDS: dict[CorpusKey, list[tuple[str, str]]] = {
    "news": [("sparse", "doc")],
    "research": [("sparse", "doc")],
    "ota": [
        ("sparse_insight", "insight_text"),
        ("sparse_recommendation", "recommendation_text"),
    ],
}


def sparse_vector_names(corpus: CorpusKey) -> tuple[str, ...]:
    return tuple(name for name, _ in CORPUS_SPARSE_FIELDS.get(corpus, []))


def corpus_has_sparse(corpus: CorpusKey) -> bool:
    return bool(CORPUS_SPARSE_FIELDS.get(corpus))

# Payload size scales with chunk token targets (content + metadata in Qdrant payload).
PAYLOAD_BASE_BYTES = 220
PAYLOAD_BYTES_PER_TOKEN = 4
SPARSE_VECTOR_BYTES = 200

# Illustrative chunks/doc when profile has no max_chunks_per_doc cap.
_ILLUSTRATIVE_CHUNKS_PER_DOC: dict[CorpusKey, int] = {
    "news": 8,
}


def _int8_quant(*, quantile: float | None = None, always_ram: bool = False) -> models.ScalarQuantization:
    cfg: dict[str, Any] = {"type": models.ScalarType.INT8}
    if quantile is not None:
        cfg["quantile"] = quantile
    if always_ram:
        cfg["always_ram"] = True
    return models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(**cfg))


def dense_vector_params(
    *,
    dim: int,
    ef_construct: int = 100,
    quantile: float | None = None,
    always_ram: bool = False,
    on_disk: bool = False,
) -> models.VectorParams:
    size = max(32, dim)
    return models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        on_disk=on_disk,
        hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=ef_construct),
        quantization_config=_int8_quant(quantile=quantile, always_ram=always_ram),
    )


def _dim(corpus: CorpusKey) -> int:
    return PROFILES[corpus].vector_dim


def news_collection_kwargs() -> dict[str, Any]:
    dim = _dim("news")
    return {
        "vectors_config": {"dense": dense_vector_params(dim=dim, ef_construct=100, quantile=0.99, always_ram=True)},
        "sparse_vectors_config": {"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    }


def research_collection_kwargs() -> dict[str, Any]:
    vp = dense_vector_params(dim=_dim("research"), ef_construct=100, on_disk=True)
    return {
        "vectors_config": {"dense": vp},
        "sparse_vectors_config": {"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    }


def ota_collection_kwargs() -> dict[str, Any]:
    vp = dense_vector_params(dim=_dim("ota"), ef_construct=100)
    return {
        "vectors_config": {
            "insight_vector": vp,
            "metric_vector": vp,
            "recommendation_vector": vp,
        },
        "sparse_vectors_config": {
            "sparse_insight": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "sparse_recommendation": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    }


COLLECTION_BUILDERS: dict[str, Any] = {
    "news": news_collection_kwargs,
    "research": research_collection_kwargs,
    "ota": ota_collection_kwargs,
}

# Payload fields indexed for server-side Filter / Range (must match loader payloads).
_ACF_PAYLOAD_INDEXES: list[tuple[str, models.PayloadSchemaType]] = [
    ("tier", models.PayloadSchemaType.INTEGER),
    ("data_level", models.PayloadSchemaType.KEYWORD),
    ("as_of_date", models.PayloadSchemaType.KEYWORD),
    ("region", models.PayloadSchemaType.KEYWORD),
    ("source_id", models.PayloadSchemaType.KEYWORD),
]

PAYLOAD_INDEXES: dict[str, list[tuple[str, models.PayloadSchemaType]]] = {
    "news": [
        ("doc_kind", models.PayloadSchemaType.KEYWORD),
        ("published_at", models.PayloadSchemaType.KEYWORD),
        ("period_start", models.PayloadSchemaType.KEYWORD),
        ("period_end", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.KEYWORD),
        ("geo_country_primary", models.PayloadSchemaType.KEYWORD),
        ("country", models.PayloadSchemaType.KEYWORD),
        # TEXT index enables MatchText substring filter on the semicolon-list payload.
        ("geo_countries", models.PayloadSchemaType.TEXT),
        ("geo_scope", models.PayloadSchemaType.KEYWORD),
        ("domains", models.PayloadSchemaType.TEXT),
        *_ACF_PAYLOAD_INDEXES,
    ],
    "research": [
        ("doc_kind", models.PayloadSchemaType.KEYWORD),
        ("geo_country_primary", models.PayloadSchemaType.KEYWORD),
        ("geo_countries", models.PayloadSchemaType.TEXT),
        # KEYWORD index lets the geo filter's ``country`` MatchValue fallback work.
        ("country", models.PayloadSchemaType.KEYWORD),
        ("geo_scope", models.PayloadSchemaType.KEYWORD),
        ("section_role", models.PayloadSchemaType.KEYWORD),
        ("content_type", models.PayloadSchemaType.KEYWORD),
        ("semantic_lane", models.PayloadSchemaType.KEYWORD),
        ("publication_year", models.PayloadSchemaType.KEYWORD),
        ("period_start", models.PayloadSchemaType.KEYWORD),
        ("period_end", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.KEYWORD),
        ("journal", models.PayloadSchemaType.KEYWORD),
        ("doi", models.PayloadSchemaType.KEYWORD),
        *_ACF_PAYLOAD_INDEXES,
    ],
    "ota": [
        ("doc_kind", models.PayloadSchemaType.KEYWORD),
        ("geo_country_primary", models.PayloadSchemaType.KEYWORD),
        ("geo_scope", models.PayloadSchemaType.KEYWORD),
        ("geo_countries", models.PayloadSchemaType.TEXT),
        *_ACF_PAYLOAD_INDEXES,
    ],
}


def indexed_fields_for_corpus(corpus: CorpusKey) -> frozenset[str]:
    """Names of payload fields with a server-side index for the given corpus.

    Used at query time by ``build_qdrant_filter`` to drop ``FieldCondition``s
    that would otherwise raise ``400 Index required but not found`` on Qdrant.
    """
    return frozenset(name for name, _ in PAYLOAD_INDEXES.get(corpus, []))


def indexed_fields_on_collection(client: Any, collection_name: str) -> frozenset[str] | None:
    """Payload field names that actually have indexes on this Qdrant collection.

    Intersect with ``indexed_fields_for_corpus`` at query time so older collections
    (created before indexes were added) do not trigger 400 errors.
    """
    try:
        info = client.get_collection(collection_name=collection_name)
    except Exception:
        return None
    schema = getattr(info.config, "payload_schema", None) or getattr(info, "payload_schema", None)
    if not isinstance(schema, dict) or not schema:
        return frozenset()
    return frozenset(str(k) for k in schema.keys())


def ensure_payload_indexes(client: Any, collection_name: str, corpus: str) -> list[str]:
    """Create payload indexes for a collection (idempotent). Returns field names created."""
    created: list[str] = []
    for field_name, schema in PAYLOAD_INDEXES.get(corpus, []):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
            created.append(field_name)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "already indexed" in msg:
                continue
            raise
    return created


def estimate_bytes_per_point(*, profile: ChunkingProfile, layout: dict[str, int]) -> int:
    """Rough RAM per Qdrant point including multi-vector layout and chunk-sized payload."""
    dim = profile.vector_dim
    num_dense = layout["dense_vectors"]
    num_sparse = layout["sparse_vectors"]
    dense_bytes = dim * num_dense  # INT8 per dimension per dense vector
    hnsw_overhead = HNSW_M * dim * 4 * num_dense
    payload_bytes = PAYLOAD_BASE_BYTES + profile.target_tokens * PAYLOAD_BYTES_PER_TOKEN
    sparse_bytes = SPARSE_VECTOR_BYTES * num_sparse
    return dense_bytes + hnsw_overhead + payload_bytes + sparse_bytes


def estimate_points_per_gib(*, profile: ChunkingProfile, layout: dict[str, int]) -> int:
    """Rough capacity for a 1 GiB RAM single-node cluster."""
    per_point = estimate_bytes_per_point(profile=profile, layout=layout)
    gib = 1024**3
    usable = int(gib * 0.65)  # headroom for OS / Qdrant process
    return max(1, usable // max(per_point, 1))


def _doc_capacity_hint(corpus: CorpusKey, profile: ChunkingProfile, points: int) -> str:
    if profile.max_chunks_per_doc:
        chunks = profile.max_chunks_per_doc
        docs = points // max(chunks, 1)
        return f"~{docs:,} docs @ max {chunks} chunks/doc"
    chunks = _ILLUSTRATIVE_CHUNKS_PER_DOC.get(corpus, 5)
    docs = points // max(chunks, 1)
    return f"~{docs:,} docs @ ~{chunks} chunks/doc (illustrative)"


def print_capacity_estimates() -> None:
    print("\n==============================")
    print("CAPACITY ESTIMATES (1 GiB RAM node)")
    print("==============================")
    print(f"  HNSW m={HNSW_M}, INT8 dense quant, ~65% RAM usable")
    print("  Payload estimate scales with per-corpus chunk token targets.")
    print()
    for corpus in ("news", "research", "ota"):
        profile = PROFILES[corpus]
        layout = CORPUS_VECTOR_LAYOUT[corpus]
        points = estimate_points_per_gib(profile=profile, layout=layout)
        per_point = estimate_bytes_per_point(profile=profile, layout=layout)
        dim = profile.vector_dim
        n_dense = layout["dense_vectors"]
        n_sparse = layout["sparse_vectors"]
        doc_hint = _doc_capacity_hint(corpus, profile, points)
        print(f"  {profile.qdrant_collection} ({corpus}):")
        print(
            f"    dim={dim}, {n_dense} dense + {n_sparse} sparse, "
            f"~{profile.target_tokens} tok/chunk, ~{per_point:,} B/point"
        )
        print(f"    ~{points:,} points -> {doc_hint}")
    print("  Reindex after model/dim/HNSW changes (--reset on loaders).")
