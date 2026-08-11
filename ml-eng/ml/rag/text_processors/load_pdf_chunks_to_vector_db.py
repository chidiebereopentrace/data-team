"""
Load preprocessed chunk JSONL into the Qdrant collection.

Expected input format:
    One JSON object per line with keys:
      - id (str)
      - text (str)
      - metadata (dict)

Defaults:
    --input  data/local/pdf_chunks.jsonl
    --collection opentrace_rag

Usage:
    PYTHONPATH=ml-eng python -m ml.rag.text_processors.load_pdf_chunks_to_vector_db
    PYTHONPATH=ml-eng python -m ml.rag.text_processors.load_pdf_chunks_to_vector_db --reset
    PYTHONPATH=ml-eng python -m ml.rag.text_processors.load_pdf_chunks_to_vector_db --batch-size 200
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Final, TypedDict

from ml.rag.chatbot.acf_metadata import enrich_acf_payload_fields

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "local" / "pdf_chunks.jsonl"


class _ProfileUpsertKwargs(TypedDict):
    input_path: Path
    collection: str
    reset: bool
    batch_size: int
    allowed_payload_keys: frozenset[str] | None
    model_id: str
    vector_dim: int
    corpus: str


def _sentence_model_id() -> str:
    s = os.environ.get("RAG_EMBEDDING_MODEL_SENTENCE", "").strip()
    if s:
        return s
    return os.environ.get("RAG_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2").strip()


def _semantic_model_id() -> str:
    s = os.environ.get("RAG_EMBEDDING_MODEL_SEMANTIC", "").strip()
    if s:
        return s
    return _sentence_model_id()


def _embed_mode() -> str:
    return os.environ.get("RAG_EMBEDDINGS_MODE", "local")


def _resolve_sparse_text(source: str, doc: str, meta: dict[str, Any]) -> str:
    if source == "doc":
        return doc
    val = meta.get(source)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return " "


def _batch_sparse_vectors(
    corpus: str,
    docs: list[str],
    metas: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Return sparse_name -> list of SparseVector (one per doc in batch)."""
    from ml.rag.scripts.qdrant_collection_specs import CORPUS_SPARSE_FIELDS

    fields = CORPUS_SPARSE_FIELDS.get(corpus)  # type: ignore[arg-type]
    if not fields:
        return {}
    try:
        from ml.rag.sparse_embeddings import embed_sparse_documents, sparse_embeddings_enabled
    except ImportError as exc:
        raise RuntimeError(
            "Sparse embeddings requested (RAG_SPARSE_EMBEDDINGS on by default) but "
            "fastembed is not installed. Install fastembed or set RAG_SPARSE_EMBEDDINGS=off."
        ) from exc
    if not sparse_embeddings_enabled():
        return {}

    out: dict[str, list[Any]] = {}
    for sparse_name, source in fields:
        texts = [_resolve_sparse_text(source, d, m) for d, m in zip(docs, metas)]
        out[sparse_name] = embed_sparse_documents(texts)
    return out


def _attach_sparse_to_vector(
    dense_vectors: dict[str, Any],
    sparse_batch: dict[str, list[Any]],
    idx: int,
) -> dict[str, Any]:
    merged = dict(dense_vectors)
    for name, vectors in sparse_batch.items():
        if idx < len(vectors):
            merged[name] = vectors[idx]
    return merged


def _clean_semicolon_list(value: str) -> str:
    parts = [p.strip() for p in (value or "").split(";")]
    parts = [p for p in parts if p]
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
    return "; ".join(out)


def _normalize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Add a small consistent metadata spine across sources without changing preprocessors.

    Adds:
      - doc_kind: news_article | academic_article | other
      - label: optional human label
      - geo_scope: country | multi_country | regional | global | unknown
      - geo_countries: '; '-separated country list (string)
      - geo_country_primary: first country in geo_countries (string)
      - ACF Path B fields via enrich_acf_payload_fields
    """
    m = dict(meta or {})

    # --- doc_kind / label ---
    if isinstance(m.get("doc_kind"), str) and m.get("doc_kind"):
        doc_kind = m["doc_kind"].strip()
    else:
        info_type = m.get("info_type")
        if isinstance(info_type, str) and info_type.strip():
            doc_kind = info_type.strip()
        else:
            doc_kind = "other"
    m["doc_kind"] = doc_kind

    if doc_kind == "news_article":
        m.setdefault("source_kind", "web_news_rss_txt")
    elif doc_kind == "academic_article":
        m.setdefault("source_kind", "pdf_text_document")
    elif doc_kind == "policy_report":
        m.setdefault("source_kind", "pdf_policy_report")

    # --- geo normalization ---
    geo_countries = ""
    if isinstance(m.get("geo_countries"), str) and m.get("geo_countries"):
        geo_countries = _clean_semicolon_list(m["geo_countries"])
    else:
        # News: single country
        country = m.get("country")
        if isinstance(country, str) and country.strip():
            geo_countries = _clean_semicolon_list(country)
        # PDFs: place_of_focus is typically a semicolon-separated list
        else:
            pof = m.get("place_of_focus")
            if isinstance(pof, str) and pof.strip():
                geo_countries = _clean_semicolon_list(pof)

    if geo_countries:
        m["geo_countries"] = geo_countries
        primary = geo_countries.split(";")[0].strip()
        if primary:
            m.setdefault("geo_country_primary", primary)
        if ";" in geo_countries:
            m.setdefault("geo_scope", "multi_country")
        else:
            m.setdefault("geo_scope", "country")
    else:
        m.setdefault("geo_scope", "unknown")

    if "domain" in m and "domains" not in m:
        m["domains"] = m.pop("domain")

    # ACF Path B provenance (best-effort at load time if preprocess missed it)
    return enrich_acf_payload_fields(m)


def _safe_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Qdrant payload supports scalar types. Coerce unsupported values to compact strings.
    """
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            # Skip null metadata values for broader Qdrant client compatibility.
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)[:1000]
    return out


# ---------------------------------------------------------------------------
# Per-collection payload allow-lists (fields kept in each Qdrant point).
# ---------------------------------------------------------------------------

# Shared ACF Path B provenance + claim/D/M fields (geo-scale tier, not OFIA authority).
_PAYLOAD_ACF: Final[frozenset[str]] = frozenset({
    "tier", "data_level", "as_of_date", "region", "source_id",
    "finding", "metric", "direction", "magnitude", "unit",
})

PAYLOAD_NEWS: Final[frozenset[str]] = frozenset({
    "content", "doc_kind", "geo_country_primary", "geo_countries", "geo_scope",
    "domains", "published_at", "title", "source", "url",
    "chunk_index", "total_chunks", "document_id", "content_hash",
    "section_path", "ingest_version",
}) | _PAYLOAD_ACF

PAYLOAD_RESEARCH: Final[frozenset[str]] = frozenset({
    "content", "doc_kind", "strategy", "geo_country_primary", "geo_countries",
    "geo_scope", "place_of_focus", "region",
    "domains", "info_type",
    "section_title", "chunk_index", "total_chunks",
    "document_id", "content_hash", "ingest_version",
    "hierarchy_path", "parent_chunk_id", "semantic_lane",
    "section_role", "content_type",
    "article_title", "authors", "publication_year", "journal", "doi",
    "volume", "issue", "pages", "bibliography_source",
}) | _PAYLOAD_ACF

PAYLOAD_OTA: Final[frozenset[str]] = frozenset({
    "content", "insight_text", "metric_text", "recommendation_text",
    "doc_kind", "geo_country_primary", "geo_countries", "geo_scope", "domains",
    "chunk_index", "total_chunks", "document_id", "content_hash",
    "ingest_version", "ota_record_id",
}) | _PAYLOAD_ACF


def _filter_payload(
    meta: dict[str, Any],
    doc: str,
    allowed: frozenset[str] | None,
) -> dict[str, str | int | float | bool]:
    """Build a point payload keeping only *allowed* keys plus ``content``."""
    payload: dict[str, str | int | float | bool] = {"content": doc}
    if allowed is None:
        payload.update(meta)
        return payload
    for k, v in meta.items():
        if k in allowed and v is not None:
            if isinstance(v, (str, int, float, bool)):
                payload[k] = v
            else:
                payload[k] = str(v)[:1000]
    return payload


def load_jsonl_chunks(path: Path) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]]]:
    """
    Parse chunk JSONL and return ids, documents, metadatas.
    Skips malformed lines and empty texts.
    """
    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            chunk_id = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            metadata = row.get("metadata", {})

            if not chunk_id or not text:
                continue
            if not isinstance(metadata, dict):
                metadata = {}

            ids.append(chunk_id)
            docs.append(text)
            metadatas.append(_safe_metadata(_normalize_metadata(metadata)))

    return ids, docs, metadatas


def _upsert_in_batches(
    client: Any,
    collection: str,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    batch_size: int,
    allowed_keys: frozenset[str] | None = None,
) -> int:
    """Legacy: single unnamed vector (default embedding model)."""
    total = len(ids)
    inserted = 0
    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_docs = docs[i:i + batch_size]
        b_meta = metadatas[i:i + batch_size]
        from ml.rag.retrievers.vector_retriever import _embed_texts
        from qdrant_client.http.models import PointStruct

        embed_mode = _embed_mode()
        model_id = os.environ.get("RAG_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
        vectors = _embed_texts(b_docs, model_id=model_id, mode=embed_mode)
        points = []
        for pid, doc, meta, vec in zip(b_ids, b_docs, b_meta, vectors):
            payload = _filter_payload(meta, doc, allowed_keys)
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        client.upsert(collection_name=collection, points=points)
        inserted += len(b_ids)
    return inserted


def _upsert_dual_vectors_batches(
    client: Any,
    collection: str,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    batch_size: int,
    allowed_keys: frozenset[str] | None = None,
) -> int:
    """Named vectors: sentence + semantic (same chunk text, two models)."""
    total = len(ids)
    inserted = 0
    mode = _embed_mode()
    mid_s = _sentence_model_id()
    mid_m = _semantic_model_id()
    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_docs = docs[i:i + batch_size]
        b_meta = metadatas[i:i + batch_size]
        from ml.rag.retrievers.vector_retriever import _embed_texts
        from qdrant_client.http.models import PointStruct

        v_sentence = _embed_texts(b_docs, model_id=mid_s, mode=mode)
        v_semantic = _embed_texts(b_docs, model_id=mid_m, mode=mode)
        points = []
        for pid, doc, meta, vs, vm in zip(b_ids, b_docs, b_meta, v_sentence, v_semantic):
            payload = _filter_payload(meta, doc, allowed_keys)
            points.append(
                PointStruct(
                    id=pid,
                    vector={"sentence": vs, "semantic": vm},
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points)
        inserted += len(b_ids)
    return inserted


def _upsert_sentence_named_batches(
    client: Any,
    collection: str,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    batch_size: int,
    allowed_keys: frozenset[str] | None = None,
) -> int:
    """Single named vector ``sentence`` (data descriptions collection)."""
    total = len(ids)
    inserted = 0
    mode = _embed_mode()
    mid_s = _sentence_model_id()
    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_docs = docs[i:i + batch_size]
        b_meta = metadatas[i:i + batch_size]
        from ml.rag.retrievers.vector_retriever import _embed_texts
        from qdrant_client.http.models import PointStruct

        vecs = _embed_texts(b_docs, model_id=mid_s, mode=mode)
        points = []
        for pid, doc, meta, vec in zip(b_ids, b_docs, b_meta, vecs):
            payload = _filter_payload(meta, doc, allowed_keys)
            points.append(PointStruct(id=pid, vector={"sentence": vec}, payload=payload))
        client.upsert(collection_name=collection, points=points)
        inserted += len(b_ids)
    return inserted


def _ensure_collection_exists(*, client: Any, collection: str, dim: int, reset: bool) -> None:
    from qdrant_client.http.models import Distance, VectorParams

    if reset:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def ensure_collection_dual_vectors(*, client: Any, collection: str, reset: bool) -> None:
    from ml.rag.retrievers.vector_retriever import _embed_texts
    from qdrant_client.http.models import Distance, VectorParams

    if reset:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass

    mode = _embed_mode()
    d_s = len(_embed_texts(["ping"], model_id=_sentence_model_id(), mode=mode)[0])
    d_m = len(_embed_texts(["ping"], model_id=_semantic_model_id(), mode=mode)[0])

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "sentence": VectorParams(size=d_s, distance=Distance.COSINE),
            "semantic": VectorParams(size=d_m, distance=Distance.COSINE),
        },
    )


def ensure_collection_sentence_named(*, client: Any, collection: str, reset: bool) -> None:
    from ml.rag.retrievers.vector_retriever import _embed_texts
    from qdrant_client.http.models import Distance, VectorParams

    if reset:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass

    mode = _embed_mode()
    d_s = len(_embed_texts(["ping"], model_id=_sentence_model_id(), mode=mode)[0])

    client.create_collection(
        collection_name=collection,
        vectors_config={"sentence": VectorParams(size=d_s, distance=Distance.COSINE)},
    )


def upsert_jsonl_to_qdrant_for_collection(
    *,
    input_path: Path,
    collection: str,
    reset: bool = False,
    batch_size: int = 100,
    allowed_payload_keys: frozenset[str] | None = None,
) -> int:
    """Route upsert to the vector layout implied by the collection's embedding profile."""
    from ml.rag.local_env import load_data_local_dotenv
    from ml.rag.paths import ML_ENG_ROOT
    from ml.rag.text_processors.chunking_config import profile_for_collection

    load_data_local_dotenv(ML_ENG_ROOT)
    prof = profile_for_collection(collection)
    mode = prof.qdrant_vector_mode
    common: _ProfileUpsertKwargs = {
        "input_path": input_path,
        "collection": collection,
        "reset": reset,
        "batch_size": batch_size,
        "allowed_payload_keys": allowed_payload_keys,
        "model_id": prof.embedding_model,
        "vector_dim": prof.vector_dim,
        "corpus": prof.corpus,
    }
    if mode == "dense_named":
        return upsert_jsonl_to_qdrant_dense_named_profile(**common)
    if mode == "legacy":
        return upsert_jsonl_to_qdrant_profile(**common)
    if mode == "sentence_named":
        return upsert_jsonl_to_qdrant_sentence_named_profile(**common)
    if mode == "research_dual":
        return upsert_jsonl_to_qdrant_research_dual_profile(**common)
    if mode == "ota_triple":
        return upsert_jsonl_to_qdrant_ota_triple(
            input_path=input_path,
            collection=collection,
            reset=reset,
            batch_size=batch_size,
            model_id=prof.embedding_model,
            corpus=prof.corpus,
        )
    return upsert_jsonl_to_qdrant_dual(
        input_path=input_path,
        collection=collection,
        reset=reset,
        batch_size=batch_size,
        allowed_payload_keys=allowed_payload_keys,
    )


def _ensure_collection_from_spec(*, client: Any, collection: str, corpus: str, reset: bool) -> None:
    from ml.rag.scripts.qdrant_collection_specs import COLLECTION_BUILDERS

    if reset:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass
    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass
    from ml.rag.scripts.qdrant_collection_specs import ensure_payload_indexes

    builder = COLLECTION_BUILDERS.get(corpus)
    if not builder:
        raise ValueError(f"No Qdrant collection spec for corpus {corpus!r}")
    client.create_collection(collection_name=collection, **builder())
    ensure_payload_indexes(client, collection, corpus)


def _research_lane_texts(doc: str, meta: dict[str, Any]) -> tuple[str, str]:
    lane = str(meta.get("semantic_lane") or "").strip().lower()
    if lane == "abstract":
        section = str(meta.get("section_title") or "").strip()
        abstract = f"{section}\n\n{doc}".strip() if section else doc
        return abstract, doc
    abstract = str(meta.get("abstract_text") or "").strip()
    section = str(meta.get("section_title") or meta.get("hierarchy_path") or meta.get("section_path") or "").strip()
    if not abstract:
        lead = doc[:700].strip()
        abstract = f"{section}\n\n{lead}".strip() if section else lead
    return abstract, doc


def upsert_jsonl_to_qdrant_profile(
    *,
    input_path: Path,
    collection: str,
    reset: bool,
    batch_size: int,
    allowed_payload_keys: frozenset[str] | None,
    model_id: str,
    vector_dim: int,
    corpus: str = "news",
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0
    from ml.rag.retrievers.vector_retriever import _get_qdrant_config, make_qdrant_client

    url, api_key, _, timeout_s = _get_qdrant_config()
    _ = url, api_key
    client = make_qdrant_client(timeout_s=timeout_s)
    _ensure_collection_from_spec(client=client, collection=collection, corpus=corpus, reset=reset)
    _ = vector_dim
    return _upsert_legacy_batches(
        client=client,
        collection=collection,
        ids=ids,
        docs=docs,
        metadatas=metadatas,
        batch_size=batch_size,
        allowed_keys=allowed_payload_keys,
        model_id=model_id,
    )


def _upsert_legacy_batches(
    *,
    client: Any,
    collection: str,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int,
    allowed_keys: frozenset[str] | None,
    model_id: str,
) -> int:
    from ml.rag.retrievers.vector_retriever import _embed_texts_for_indexing
    from qdrant_client.http.models import PointStruct

    total = 0
    for i in range(0, len(docs), batch_size):
        b_ids = ids[i : i + batch_size]
        b_docs = docs[i : i + batch_size]
        b_meta = metadatas[i : i + batch_size]
        vectors = _embed_texts_for_indexing(b_docs, model_id=model_id, mode=_embed_mode(), is_query=False)
        points = []
        for pid, doc, meta, vec in zip(b_ids, b_docs, b_meta, vectors):
            payload = _filter_payload(meta, doc, allowed_keys)
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        if points:
            client.upsert(collection_name=collection, points=points)
            total += len(points)
    return total


def upsert_jsonl_to_qdrant_dense_named_profile(
    *,
    input_path: Path,
    collection: str,
    reset: bool,
    batch_size: int,
    allowed_payload_keys: frozenset[str] | None,
    model_id: str,
    vector_dim: int,
    corpus: str,
) -> int:
    return _upsert_single_named_vector_profile(
        input_path=input_path,
        collection=collection,
        reset=reset,
        batch_size=batch_size,
        allowed_payload_keys=allowed_payload_keys,
        model_id=model_id,
        vector_dim=vector_dim,
        corpus=corpus,
        vector_name="dense",
    )


def upsert_jsonl_to_qdrant_sentence_named_profile(
    *,
    input_path: Path,
    collection: str,
    reset: bool,
    batch_size: int,
    allowed_payload_keys: frozenset[str] | None,
    model_id: str,
    vector_dim: int,
    corpus: str,
) -> int:
    return _upsert_single_named_vector_profile(
        input_path=input_path,
        collection=collection,
        reset=reset,
        batch_size=batch_size,
        allowed_payload_keys=allowed_payload_keys,
        model_id=model_id,
        vector_dim=vector_dim,
        corpus=corpus,
        vector_name="sentence",
    )


def _upsert_single_named_vector_profile(
    *,
    input_path: Path,
    collection: str,
    reset: bool,
    batch_size: int,
    allowed_payload_keys: frozenset[str] | None,
    model_id: str,
    vector_dim: int,
    corpus: str,
    vector_name: str,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0
    from ml.rag.retrievers.vector_retriever import _embed_texts_for_indexing, make_qdrant_client
    from qdrant_client.http.models import PointStruct

    client = make_qdrant_client()
    _ensure_collection_from_spec(client=client, collection=collection, corpus=corpus, reset=reset)
    _ = vector_dim

    total = 0
    for i in range(0, len(docs), batch_size):
        b_ids = ids[i : i + batch_size]
        b_docs = docs[i : i + batch_size]
        b_meta = metadatas[i : i + batch_size]
        vecs = _embed_texts_for_indexing(b_docs, model_id=model_id, mode=_embed_mode(), is_query=False)
        sparse_batch = _batch_sparse_vectors(corpus=corpus, docs=b_docs, metas=b_meta)
        points = []
        for j, (pid, doc, meta, vec) in enumerate(zip(b_ids, b_docs, b_meta, vecs)):
            payload = _filter_payload(meta, doc, allowed_payload_keys)
            point_vectors: dict[str, Any] = _attach_sparse_to_vector({vector_name: vec}, sparse_batch, j)
            points.append(PointStruct(id=pid, vector=point_vectors, payload=payload))
        if points:
            client.upsert(collection_name=collection, points=points)
            total += len(points)
    return total


def upsert_jsonl_to_qdrant_research_dual_profile(
    *,
    input_path: Path,
    collection: str,
    reset: bool,
    batch_size: int,
    allowed_payload_keys: frozenset[str] | None,
    model_id: str,
    vector_dim: int,
    corpus: str,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0
    from ml.rag.retrievers.vector_retriever import _embed_texts_for_indexing, make_qdrant_client
    from qdrant_client.http.models import PointStruct

    client = make_qdrant_client()
    _ensure_collection_from_spec(client=client, collection=collection, corpus=corpus, reset=reset)
    _ = vector_dim

    total = 0
    for i in range(0, len(docs), batch_size):
        b_ids = ids[i : i + batch_size]
        b_docs = docs[i : i + batch_size]
        b_meta = metadatas[i : i + batch_size]
        abs_docs = [_research_lane_texts(d, m)[0] for d, m in zip(b_docs, b_meta)]
        v_abs = _embed_texts_for_indexing(abs_docs, model_id=model_id, mode=_embed_mode(), is_query=False)
        v_content = _embed_texts_for_indexing(b_docs, model_id=model_id, mode=_embed_mode(), is_query=False)
        sparse_batch = _batch_sparse_vectors(corpus=corpus, docs=b_docs, metas=b_meta)
        points = []
        for j, (pid, doc, meta, va, vc) in enumerate(zip(b_ids, b_docs, b_meta, v_abs, v_content)):
            payload = _filter_payload(meta, doc, allowed_payload_keys)
            point_vectors = _attach_sparse_to_vector(
                {"abstract_vector": va, "content_vector": vc},
                sparse_batch,
                j,
            )
            points.append(PointStruct(id=pid, vector=point_vectors, payload=payload))
        if points:
            client.upsert(collection_name=collection, points=points)
            total += len(points)
    return total


def upsert_jsonl_to_qdrant(
    *,
    input_path: Path,
    collection: str,
    reset: bool = False,
    batch_size: int = 100,
    allowed_payload_keys: frozenset[str] | None = None,
) -> int:
    """
    Programmatic API for loading chunk JSONL into Qdrant.

    Returns number of upserted points.
    """
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0

    from ml.rag.retrievers.vector_retriever import _embed_texts, make_qdrant_client

    client = make_qdrant_client()

    embed_mode = _embed_mode()
    model_id = os.environ.get("RAG_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    dim = len(_embed_texts(["ping"], model_id=model_id, mode=embed_mode)[0])

    _ensure_collection_exists(client=client, collection=collection, dim=dim, reset=reset)
    return _upsert_in_batches(
        client=client,
        collection=collection,
        ids=ids,
        docs=docs,
        metadatas=metadatas,
        batch_size=max(1, int(batch_size)),
        allowed_keys=allowed_payload_keys,
    )


def upsert_jsonl_to_qdrant_dual(
    *,
    input_path: Path,
    collection: str,
    reset: bool = False,
    batch_size: int = 100,
    allowed_payload_keys: frozenset[str] | None = None,
) -> int:
    """Research/news: named vectors ``sentence`` + ``semantic``."""
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0

    from ml.rag.retrievers.vector_retriever import _get_qdrant_config, make_qdrant_client

    url, api_key, _, timeout_s = _get_qdrant_config()
    _ = url, api_key
    client = make_qdrant_client(timeout_s=timeout_s)

    ensure_collection_dual_vectors(client=client, collection=collection, reset=reset)
    return _upsert_dual_vectors_batches(
        client=client,
        collection=collection,
        ids=ids,
        docs=docs,
        metadatas=metadatas,
        batch_size=max(1, int(batch_size)),
        allowed_keys=allowed_payload_keys,
    )


def upsert_jsonl_to_qdrant_sentence_named(
    *,
    input_path: Path,
    collection: str,
    reset: bool = False,
    batch_size: int = 100,
    allowed_payload_keys: frozenset[str] | None = None,
) -> int:
    """Data descriptions: single named vector ``sentence``."""
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    ids, docs, metadatas = load_jsonl_chunks(input_path)
    if not docs:
        return 0

    from ml.rag.retrievers.vector_retriever import make_qdrant_client

    client = make_qdrant_client()

    ensure_collection_sentence_named(client=client, collection=collection, reset=reset)
    return _upsert_sentence_named_batches(
        client=client,
        collection=collection,
        ids=ids,
        docs=docs,
        metadatas=metadatas,
        batch_size=max(1, int(batch_size)),
        allowed_keys=allowed_payload_keys,
    )


# ---------------------------------------------------------------------------
# OTA Insights: triple named vectors (insight / metric / recommendation)
# ---------------------------------------------------------------------------

_OTA_TEXT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "insight_text": ("insight_text", "text_insight"),
    "metric_text": ("metric_text", "text_metric"),
    "recommendation_text": ("recommendation_text", "text_recommendation"),
}


def _pick_text(row: dict[str, Any], canonical: str, aliases: tuple[str, ...], fallback: str) -> str:
    """Resolve a text field from top-level row or nested metadata."""
    for alias in aliases:
        v = row.get(alias) or (row.get("metadata") or {}).get(alias)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback


def load_ota_jsonl_chunks(
    path: Path,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[dict[str, str | int | float | bool]],
]:
    """Parse OTA insight JSONL.

    Returns (ids, insight_texts, metric_texts, recommendation_texts,
             combined_texts, metadatas).
    """
    ids: list[str] = []
    insight_texts: list[str] = []
    metric_texts: list[str] = []
    recommendation_texts: list[str] = []
    combined_texts: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            chunk_id = str(row.get("id", "")).strip()
            if not chunk_id:
                continue

            fallback = str(row.get("text", "")).strip()
            i_text = _pick_text(row, "insight_text", _OTA_TEXT_ALIASES["insight_text"], fallback)
            m_text = _pick_text(row, "metric_text", _OTA_TEXT_ALIASES["metric_text"], fallback)
            r_text = _pick_text(row, "recommendation_text", _OTA_TEXT_ALIASES["recommendation_text"], fallback)

            if not (i_text or m_text or r_text):
                continue

            meta = row.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("info_type", "ota_insight")

            from ml.rag.text_processors.domain_taxonomy import infer_domains
            combined = " ".join(filter(None, [i_text, m_text, r_text]))
            if "domains" not in meta and "domain" not in meta:
                meta["domains"] = "; ".join(infer_domains(combined))

            normed = _normalize_metadata(meta)
            safe = _safe_metadata(normed)

            safe["insight_text"] = i_text
            safe["metric_text"] = m_text
            safe["recommendation_text"] = r_text

            ids.append(chunk_id)
            insight_texts.append(i_text)
            metric_texts.append(m_text)
            recommendation_texts.append(r_text)
            combined_texts.append(combined)
            metadatas.append(safe)

    return ids, insight_texts, metric_texts, recommendation_texts, combined_texts, metadatas


def _upsert_ota_triple_batches(
    client: Any,
    collection: str,
    ids: list[str],
    insight_texts: list[str],
    metric_texts: list[str],
    recommendation_texts: list[str],
    combined_texts: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    batch_size: int,
    allowed_keys: frozenset[str] | None = PAYLOAD_OTA,
    model_id: str | None = None,
) -> int:
    """Embed three text lanes into three named vectors per point."""
    total = len(ids)
    inserted = 0
    mode = _embed_mode()
    mid = model_id or _sentence_model_id()
    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_it = insight_texts[i:i + batch_size]
        b_mt = metric_texts[i:i + batch_size]
        b_rt = recommendation_texts[i:i + batch_size]
        b_ct = combined_texts[i:i + batch_size]
        b_meta = metadatas[i:i + batch_size]

        from ml.rag.retrievers.vector_retriever import _embed_texts_for_indexing
        from qdrant_client.http.models import PointStruct

        v_insight = _embed_texts_for_indexing(b_it, model_id=mid, mode=mode, is_query=False)
        v_metric = _embed_texts_for_indexing(b_mt, model_id=mid, mode=mode, is_query=False)
        v_recommendation = _embed_texts_for_indexing(b_rt, model_id=mid, mode=mode, is_query=False)

        sparse_batch = _batch_sparse_vectors(corpus="ota", docs=b_ct, metas=b_meta)
        points = []
        for j, (pid, ct, meta, vi, vm, vr) in enumerate(zip(b_ids, b_ct, b_meta, v_insight, v_metric, v_recommendation)):
            payload = _filter_payload(meta, ct, allowed_keys)
            point_vectors = _attach_sparse_to_vector(
                {
                    "insight_vector": vi,
                    "metric_vector": vm,
                    "recommendation_vector": vr,
                },
                sparse_batch,
                j,
            )
            points.append(PointStruct(id=pid, vector=point_vectors, payload=payload))
        client.upsert(collection_name=collection, points=points)
        inserted += len(b_ids)
    return inserted


def ensure_collection_ota_triple(*, client: Any, collection: str, reset: bool) -> None:
    _ensure_collection_from_spec(client=client, collection=collection, corpus="ota", reset=reset)


def upsert_jsonl_to_qdrant_ota_triple(
    *,
    input_path: Path,
    collection: str,
    reset: bool = False,
    batch_size: int = 100,
    model_id: str | None = None,
    corpus: str = "ota",
) -> int:
    """OTA insights: three named vectors (insight / metric / recommendation)."""
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    ids, i_texts, m_texts, r_texts, c_texts, metadatas = load_ota_jsonl_chunks(input_path)
    if not ids:
        return 0

    from ml.rag.retrievers.vector_retriever import _get_qdrant_config, make_qdrant_client

    url, api_key, _, timeout_s = _get_qdrant_config()
    _ = url, api_key
    client = make_qdrant_client(timeout_s=timeout_s)

    _ensure_collection_from_spec(client=client, collection=collection, corpus=corpus, reset=reset)
    mid = model_id or _sentence_model_id()
    return _upsert_ota_triple_batches(
        client=client,
        collection=collection,
        ids=ids,
        insight_texts=i_texts,
        metric_texts=m_texts,
        recommendation_texts=r_texts,
        combined_texts=c_texts,
        metadatas=metadatas,
        batch_size=max(1, int(batch_size)),
        model_id=mid,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load chunk JSONL into Qdrant collection."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to chunk JSONL (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=os.environ.get("QDRANT_COLLECTION", "opentrace_rag"),
        help="Qdrant collection name (default: opentrace_rag)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Upsert batch size (default: 100)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before loading.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path: Path = args.input

    try:
        inserted = upsert_jsonl_to_qdrant_for_collection(
            input_path=input_path,
            collection=args.collection,
            reset=bool(args.reset),
            batch_size=int(args.batch_size),
        )
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {input_path}")

    print(
        f"Upserted {inserted} chunks into collection '{args.collection}' "
        f"in Qdrant"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

