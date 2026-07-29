"""
Vector DB retriever: Qdrant Cloud.

This module keeps the retriever interface stable (`BaseRetriever`) while allowing
the vector store implementation to be swapped. In this repo, RAG uses Qdrant Cloud.

Typical collection layouts (embedding model must match how points were indexed):

  news_data / research_other_papers        – single named ``dense`` vector (mode: **dense_named**; e5-small)
  legacy dual-vector collections           – named ``sentence`` + ``semantic`` (mode: **dual**)
  research_other_papers (legacy schema)    – ``abstract_vector`` + ``content_vector`` (mode: **research_dual**)
  data descriptions (DOCX loader)          – named ``sentence`` only (mode: **sentence_named**)
  OTA_insights                             – insight / metric / recommendation (mode: **ota_triple**)
  BQ_table_descriptions (triple schema)    – table / schema / business (mode: **bq_triple**)
  single-vector collections                – unnamed vector (mode: **legacy**)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.rag.retrievers.base import BaseRetriever
from ml.rag.observability import get_observe_decorator, trace_elapsed_ms, update_current_span_metadata
from ml.rag.text_processors.chunking_config import profile_for_collection

logger = logging.getLogger(__name__)

_observe_span = get_observe_decorator()

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

# Loading SentenceTransformer is slow (model files + torch) and not thread-safe on Windows
# (file locks, DLL races). We load each model once per process and reuse the instance.
_SENTENCE_MODEL_LOCK = threading.Lock()


# #region agent log
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # data-team workspace root


def _agent_debug_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    # Gated for production containers (RAG_DEBUG_SESSIONS=1 to enable).
    if os.environ.get("RAG_DEBUG_SESSIONS", "").strip().lower() not in ("1", "true", "on"):
        return
    try:
        payload = {
            "sessionId": "6c8b2f",
            "id": f"log_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with (_WORKSPACE_ROOT / "debug-6c8b2f.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _research_exclude_roles() -> frozenset[str]:
    from ml.rag.text_processors.preprocess.section_roles import (
        exclude_boilerplate_enabled,
        research_excluded_roles,
    )

    if not exclude_boilerplate_enabled():
        return frozenset()
    return research_excluded_roles()


def _is_research_collection(collection_name: str) -> bool:
    try:
        return profile_for_collection(collection_name).corpus == "research"
    except Exception:
        return False


def embedding_model_id(collection_name: str | None = None) -> str:
    """Embedding model for a collection (per-corpus profile) or global fallback."""
    if collection_name:
        return profile_for_collection(collection_name).embedding_model
    return _env("RAG_EMBEDDING_MODEL_ID", DEFAULT_MODEL) or DEFAULT_MODEL


def _is_e5_model(model_id: str) -> bool:
    return "e5" in (model_id or "").lower()


def _prefix_texts(texts: list[str], *, model_id: str, is_query: bool) -> list[str]:
    if not _is_e5_model(model_id):
        return texts
    prefix = "query: " if is_query else "passage: "
    return [prefix + (t or "") for t in texts]


def _embed_texts_for_indexing(
    texts: list[str],
    *,
    model_id: str,
    mode: str,
    is_query: bool = False,
) -> list[list[float]]:
    return _embed_texts(_prefix_texts(texts, model_id=model_id, is_query=is_query), model_id=model_id, mode=mode)


def _get_qdrant_config() -> tuple[str, str, str, float]:
    url = _env("QDRANT_URL").strip('"').strip("'")
    api_key = _env("QDRANT_API_KEY").strip('"').strip("'")
    collection = _env("QDRANT_COLLECTION", "opentrace_rag") or "opentrace_rag"
    try:
        timeout_s = float(_env("QDRANT_TIMEOUT_S", "30") or 30)
    except Exception:
        timeout_s = 30.0
    # #region agent log
    _agent_debug_log(
        "vector_retriever.py:_get_qdrant_config",
        "qdrant env probe",
        {
            "url_present": bool(url),
            "api_key_present": bool(api_key),
            "url_len": len(url),
            "api_key_len": len(api_key),
            "collection": collection,
            "timeout_s": timeout_s,
        },
        "C",
    )
    # #endregion
    if not url or not api_key:
        raise RuntimeError(
            "Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY "
            "(and optionally QDRANT_COLLECTION, QDRANT_TIMEOUT_S)."
        )
    return url, api_key, collection, timeout_s


def make_qdrant_client(*, timeout_s: float | None = None):
    """Qdrant client using env config; skips client/server version compatibility check."""
    from qdrant_client import QdrantClient

    url, api_key, _, default_timeout = _get_qdrant_config()
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=int(timeout_s if timeout_s is not None else default_timeout or 30),
        check_compatibility=False,
    )


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_id: str) -> Any:
    """Process-wide singleton SentenceTransformer per model_id.

    `lru_cache` makes lookups thread-safe; the explicit lock serializes the
    initial load so concurrent threads do not race on file/DLL access.
    """
    with _SENTENCE_MODEL_LOCK:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer %r (one-time)", model_id)
        return SentenceTransformer(model_id)


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_embed_backend(mode: str) -> str:
    """
    Map RAG_EMBEDDINGS_MODE to an in-container implementation (no HF inference API).

    - fastembed: ONNX dense via fastembed (Railway / slim images)
    - local: sentence_transformers when torch is installed; else fastembed
    - hf_api (legacy): redirected to fastembed with a warning
    """
    raw = (mode or "local").strip().lower()
    if raw == "hf_api":
        logger.warning(
            "RAG_EMBEDDINGS_MODE=hf_api is deprecated; using in-container fastembed instead"
        )
        return "fastembed"
    if raw == "fastembed":
        return "fastembed"
    if raw == "local":
        return "local" if _sentence_transformers_available() else "fastembed"
    logger.warning("Unknown RAG_EMBEDDINGS_MODE=%r; using fastembed", raw)
    return "fastembed"


def _embed_texts(texts: list[str], *, model_id: str, mode: str) -> list[list[float]]:
    backend = _resolve_embed_backend(mode)
    if backend == "fastembed":
        from ml.rag.dense_embeddings import embed_dense_texts

        return embed_dense_texts(texts, model_id=model_id)

    m = _load_sentence_transformer(model_id)
    vecs = m.encode(texts, normalize_embeddings=True)
    return [[float(x) for x in row] for row in vecs]


def _publication_years_in_range(
    published_at_from: str | None,
    published_at_to: str | None,
    *,
    max_span: int = 60,
) -> list[str]:
    """Year strings for MatchAny on KEYWORD-indexed ``publication_year`` (avoids integer Range)."""
    from datetime import date

    today_y = date.today().year
    y0 = int(published_at_from[:4]) if published_at_from and len(published_at_from) >= 4 else 1900
    y1 = int(published_at_to[:4]) if published_at_to and len(published_at_to) >= 4 else today_y
    if y1 < y0:
        y0, y1 = y1, y0
    if y1 - y0 > max_span:
        y1 = y0 + max_span
    return [str(y) for y in range(y0, y1 + 1)]


def build_qdrant_filter(
    *,
    doc_kind: str | None = None,
    doc_kinds: list[str] | None = None,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
    published_at_from: str | None = None,
    published_at_to: str | None = None,
    domains_substring: str | None = None,
    exclude_section_roles: frozenset[str] | None = None,
    indexed_fields: frozenset[str] | None = None,
) -> Any | None:
    """Build a Qdrant Filter for indexed payload fields (requires payload indexes).

    ``indexed_fields`` (optional): when provided, any ``FieldCondition`` whose key
    is not in the set is silently dropped. This prevents Qdrant from returning
    ``400 Bad request: Index required but not found for "<field>"`` when a
    collection happens to not index every field referenced here. Pass the result
    of ``indexed_fields_for_corpus`` from ``qdrant_collection_specs``.
    """
    try:
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchText,
            MatchValue,
            Range,
        )
    except ImportError:
        return None

    def _allow(field: str) -> bool:
        return indexed_fields is None or field in indexed_fields

    def _geo_should_for_country(gc: str) -> list[Any]:
        geo_should: list[Any] = []
        if _allow("geo_country_primary"):
            geo_should.append(FieldCondition(key="geo_country_primary", match=MatchValue(value=gc)))
        if _allow("country"):
            geo_should.append(FieldCondition(key="country", match=MatchValue(value=gc)))
        if _allow("geo_countries"):
            geo_should.append(FieldCondition(key="geo_countries", match=MatchText(text=gc)))
        return geo_should

    must: list[Any] = []
    must_not: list[Any] = []

    kinds: list[str] = []
    if doc_kinds:
        kinds = [str(k).strip() for k in doc_kinds if str(k).strip()]
    elif doc_kind:
        kinds = [doc_kind.strip()]
    if kinds and _allow("doc_kind"):
        if len(kinds) == 1:
            must.append(FieldCondition(key="doc_kind", match=MatchValue(value=kinds[0])))
        else:
            must.append(FieldCondition(key="doc_kind", match=MatchAny(any=kinds)))

    countries: list[str] = []
    if geo_countries:
        countries = [str(c).strip() for c in geo_countries if str(c).strip()]
    elif geo_country and geo_country.strip():
        countries = [geo_country.strip()]

    if len(countries) >= 2:
        country_filters: list[Any] = []
        for gc in countries:
            geo_should = _geo_should_for_country(gc)
            if geo_should:
                country_filters.append(Filter(should=geo_should))
        if country_filters:
            must.append(Filter(should=country_filters))
    elif len(countries) == 1:
        geo_should = _geo_should_for_country(countries[0])
        if geo_should:
            must.append(Filter(should=geo_should))

    if (published_at_from or published_at_to) and _allow("published_at"):
        range_args: dict[str, str] = {}
        if published_at_from:
            range_args["gte"] = published_at_from
        if published_at_to:
            range_args["lte"] = published_at_to
        # KEYWORD index on ISO date strings (YYYY-MM-DD); lexicographic range is valid.
        must.append(
            FieldCondition(key="published_at", range=Range(**range_args))  # type: ignore[arg-type]
        )
    elif (published_at_from or published_at_to) and _allow("publication_year"):
        # Research: publication_year is KEYWORD-indexed — use MatchAny, not numeric Range.
        years = _publication_years_in_range(published_at_from, published_at_to)
        if years:
            must.append(FieldCondition(key="publication_year", match=MatchAny(any=years)))

    if domains_substring and _allow("domains"):
        ds = domains_substring.strip()
        if ds:
            must.append(FieldCondition(key="domains", match=MatchText(text=ds)))

    if exclude_section_roles and _allow("section_role"):
        for role in sorted(exclude_section_roles):
            must_not.append(FieldCondition(key="section_role", match=MatchValue(value=role)))

    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _safe_payload(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)[:1000]
    return out


def _merge_scored_hits(hits_lists: list[list[Any]], limit: int) -> list[Any]:
    """Merge multiple search results by point id, keeping the higher score."""
    best: dict[Any, tuple[float, Any]] = {}
    for hits in hits_lists:
        for h in hits or []:
            pid = getattr(h, "id", None)
            if pid is None:
                continue
            sc = float(getattr(h, "score", 0.0) or 0.0)
            if pid not in best or sc > best[pid][0]:
                best[pid] = (sc, h)
    merged = sorted(best.values(), key=lambda x: x[0], reverse=True)
    return [t[1] for t in merged[:limit]]


# -- Named vector definitions per collection schema --------------------------

RESEARCH_VECTORS = ("abstract_vector", "content_vector")
DUAL_SENTENCE_SEMANTIC = ("sentence", "semantic")
OTA_VECTORS = ("insight_vector", "metric_vector", "recommendation_vector")
BQ_VECTORS = ("table_vector", "schema_vector", "business_vector")

# Maps a query_using shorthand → actual named vector(s) to search.
_RESEARCH_USING: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract_vector",),
    "content": ("content_vector",),
    "both": RESEARCH_VECTORS,
}
_OTA_USING: dict[str, tuple[str, ...]] = {
    "insight": ("insight_vector",),
    "metric": ("metric_vector",),
    "recommendation": ("recommendation_vector",),
    "merge": OTA_VECTORS,
}
_BQ_USING: dict[str, tuple[str, ...]] = {
    "table": ("table_vector",),
    "schema": ("schema_vector",),
    "business": ("business_vector",),
    "merge": BQ_VECTORS,
}


def _sparse_names_for_collection(collection_name: str) -> tuple[str, ...]:
    """Sparse vector names for hybrid RRF when enabled for this collection."""
    try:
        from ml.rag.sparse_embeddings import hybrid_search_enabled, sparse_embeddings_enabled
        from ml.rag.scripts.qdrant_collection_specs import sparse_vector_names
    except ImportError:
        return ()
    if not hybrid_search_enabled() or not sparse_embeddings_enabled():
        return ()
    prof = profile_for_collection(collection_name)
    return sparse_vector_names(prof.corpus)


def _collection_has_hybrid_sparse(collection_name: str) -> bool:
    return bool(_sparse_names_for_collection(collection_name))


class VectorRetriever(BaseRetriever):
    """
    Qdrant Cloud retriever.

    Env:
      - QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION
      - RAG_EMBEDDINGS_MODE=local|fastembed (in-container only; fastembed on Railway)
      - RAG_EMBEDDING_MODEL_ID (default BAAI/bge-m3)
      - RAG_QDRANT_VECTOR_SEARCH_MODE=legacy|dual|sentence_named|research_dual|ota_triple|bq_triple
      - RAG_QDRANT_DUAL_QUERY_USING=sentence|semantic|both (only for mode dual; default both)
      - RAG_QDRANT_RESEARCH_QUERY_USING=abstract|content|both
      - RAG_QDRANT_OTA_QUERY_USING=insight|metric|recommendation|merge
      - RAG_QDRANT_BQ_QUERY_USING=table|schema|business|merge
      - RAG_SPARSE_EMBEDDINGS=on|off (BM25 sparse vectors on upsert; default on)
      - RAG_QDRANT_HYBRID_SEARCH=on|off (dense+sparse RRF at query; default on)
      - RAG_HYBRID_DENSE_PREFETCH / RAG_HYBRID_SPARSE_PREFETCH / RAG_HYBRID_FUSION_LIMIT (default 20 each)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embed_model: str = DEFAULT_MODEL,
    ):
        self.collection_name = (
            (collection_name or _env("QDRANT_COLLECTION", "opentrace_rag")) or "opentrace_rag"
        )
        self.embed_model = embed_model
        self._client = None
        self._embed_mode = _env("RAG_EMBEDDINGS_MODE", "local") or "local"
        self._embed_model_id = embedding_model_id(self.collection_name)

    def _get_client(self):
        if self._client is not None:
            return self._client
        url, api_key, _, timeout_s = _get_qdrant_config()
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("Install qdrant-client: pip install qdrant-client") from e
        self._client = make_qdrant_client(timeout_s=timeout_s)
        return self._client

    def _ensure_collection(self) -> None:
        """Auto-create only legacy single-vector collections; named-vector collections are created by the create script."""
        client = self._get_client()
        collection = self.collection_name
        try:
            client.get_collection(collection_name=collection)
            return
        except Exception:
            pass

        dim = len(_embed_texts(["ping"], model_id=self._embed_model_id, mode=self._embed_mode)[0])
        from qdrant_client.http.models import Distance, VectorParams

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def _metadata_passes_filters(
        self,
        meta: dict[str, Any],
        *,
        doc_kind: str | None,
        doc_kinds: list[str] | None = None,
        geo_country: str | None,
        geo_countries: list[str] | None = None,
        published_at_from: str | None,
        published_at_to: str | None,
        domains_substring: str | None,
        exclude_section_roles: frozenset[str] | None = None,
    ) -> bool:
        allowed_kinds: list[str] = []
        if doc_kinds:
            allowed_kinds = [str(k).strip() for k in doc_kinds if str(k).strip()]
        elif doc_kind:
            allowed_kinds = [doc_kind.strip()]

        if allowed_kinds:
            dk = str(meta.get("doc_kind") or "").strip()
            it = str(meta.get("info_type") or "").strip()
            matched = dk in allowed_kinds or it in allowed_kinds
            if not matched and "bq_table_description" in allowed_kinds:
                matched = dk == "bq_table_description" or str(meta.get("type") or "").strip().lower().startswith(
                    "bq "
                )
            if not matched:
                return False

        geo_list: list[str] = []
        if geo_countries:
            geo_list = [str(c).strip().lower() for c in geo_countries if str(c).strip()]
        elif geo_country:
            geo_list = [geo_country.strip().lower()]

        if geo_list:
            primary = str(meta.get("geo_country_primary") or meta.get("country") or "").lower()
            blob = str(meta.get("geo_countries") or "").lower()
            if not any(gc in primary or gc in blob for gc in geo_list):
                return False

        pub = str(meta.get("published_at") or "").strip()[:10]
        if pub and re.match(r"^\d{4}-\d{2}-\d{2}$", pub):
            if published_at_from and pub < published_at_from:
                return False
            if published_at_to and pub > published_at_to:
                return False
        elif published_at_from or published_at_to:
            py_raw = meta.get("publication_year")
            if py_raw is not None and str(py_raw).strip():
                py = str(py_raw).strip()[:4]
                if re.match(r"^\d{4}$", py):
                    if published_at_from and py < published_at_from[:4]:
                        return False
                    if published_at_to and py > published_at_to[:4]:
                        return False

        if domains_substring:
            ds = (meta.get("domains") or meta.get("domain") or "")
            if domains_substring.lower() not in str(ds).lower():
                return False

        if exclude_section_roles:
            role = str(meta.get("section_role") or "").strip().lower()
            if role in exclude_section_roles:
                return False

        return True

    # -- internal: query by named vectors ------------------------------------

    def _query_hybrid(
        self,
        query: str,
        dense_names: tuple[str, ...],
        sparse_names: tuple[str, ...],
        *,
        dense_prefetch: int,
        sparse_prefetch: int,
        fusion_limit: int,
        q_filter: Any,
    ) -> list[Any]:
        """Dense + sparse prefetch fused with reciprocal rank fusion (RRF).

        The geo/temporal/doc_kind filter is applied **inside each ``Prefetch``** so
        each dense/sparse candidate stream is pre-filtered before fusion. This is the
        Qdrant-recommended pattern for dynamic, query-aware retrieval — top-level
        ``query_filter`` would only post-filter the fused results.
        """
        from ml.rag.sparse_embeddings import embed_sparse_query
        from qdrant_client.http.models import Fusion, FusionQuery, Prefetch

        client = self._get_client()
        collection = self.collection_name
        qv = _embed_texts_for_indexing(
            [query], model_id=self._embed_model_id, mode=self._embed_mode, is_query=True
        )[0]
        sqv = embed_sparse_query(query)

        if q_filter is not None:
            logger.debug(
                "Hybrid pre-filter applied per Prefetch for %s: %s",
                collection,
                q_filter,
            )

        prefetch: list[Any] = []
        for vname in dense_names:
            prefetch.append(
                Prefetch(query=qv, using=vname, limit=dense_prefetch, filter=q_filter)
            )
        for sname in sparse_names:
            prefetch.append(
                Prefetch(query=sqv, using=sname, limit=sparse_prefetch, filter=q_filter)
            )

        resp = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fusion_limit,
            with_payload=True,
        )
        return resp.points or []

    def _query_named_vectors(
        self,
        query: str,
        vector_names: tuple[str, ...],
        fetch_n: int,
        q_filter: Any,
        *,
        top_k: int,
        sparse_names: tuple[str, ...] | None = None,
    ) -> list[Any]:
        """Embed query once, search dense named vectors; optionally hybrid with sparse RRF."""
        if sparse_names is None:
            sparse_names = _sparse_names_for_collection(self.collection_name)
        if sparse_names:
            try:
                from ml.rag.sparse_embeddings import (
                    hybrid_dense_prefetch_limit,
                    hybrid_fusion_limit,
                    hybrid_sparse_prefetch_limit,
                )

                return self._query_hybrid(
                    query,
                    vector_names,
                    sparse_names,
                    dense_prefetch=hybrid_dense_prefetch_limit(),
                    sparse_prefetch=hybrid_sparse_prefetch_limit(),
                    fusion_limit=hybrid_fusion_limit(top_k=top_k),
                    q_filter=q_filter,
                )
            except ImportError as exc:
                logger.warning(
                    "Hybrid search disabled (missing dependency) for %s: %s",
                    self.collection_name,
                    exc,
                )
            except Exception:
                # Fall back to dense-only if hybrid fails (e.g. empty sparse index).
                logger.exception(
                    "Hybrid (dense+sparse) search failed for %s; falling back to dense-only",
                    self.collection_name,
                )

        client = self._get_client()
        collection = self.collection_name
        qv = _embed_texts_for_indexing(
            [query], model_id=self._embed_model_id, mode=self._embed_mode, is_query=True
        )[0]

        query_kwargs: dict[str, Any] = {"limit": fetch_n, "with_payload": True}
        if q_filter is not None:
            query_kwargs["query_filter"] = q_filter

        if len(vector_names) == 1:
            resp = client.query_points(
                collection_name=collection,
                query=qv,
                using=vector_names[0],
                **query_kwargs,
            )
            return resp.points or []

        all_hits: list[list[Any]] = []
        for vname in vector_names:
            resp = client.query_points(
                collection_name=collection,
                query=qv,
                using=vname,
                **query_kwargs,
            )
            all_hits.append(resp.points or [])
        return _merge_scored_hits(all_hits, fetch_n)

    # -- public retrieve -----------------------------------------------------

    @_observe_span(as_type="span", name="retrieval.qdrant", capture_input=False, capture_output=False)
    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Return top_k similar chunks from Qdrant.

        Each item: { "content", "score", "metadata", "source": "vector" }.

        Kwargs:
          vector_search_mode: legacy | dual | sentence_named | research_dual | ota_triple | bq_triple
          doc_kind / doc_kinds / geo_country / published_at_from / published_at_to / domains_substring
        """
        doc_kind = kwargs.get("doc_kind")
        t0 = time.perf_counter()
        if isinstance(doc_kind, str):
            doc_kind = doc_kind.strip() or None
        else:
            doc_kind = None

        raw_doc_kinds = kwargs.get("doc_kinds")
        doc_kinds: list[str] | None = None
        if isinstance(raw_doc_kinds, (list, tuple)):
            doc_kinds = [str(k).strip() for k in raw_doc_kinds if str(k).strip()] or None

        geo_country = kwargs.get("geo_country")
        if isinstance(geo_country, str):
            geo_country = geo_country.strip() or None
        else:
            geo_country = None

        raw_geo_countries = kwargs.get("geo_countries")
        geo_countries: list[str] | None = None
        if isinstance(raw_geo_countries, (list, tuple)):
            geo_countries = [str(c).strip() for c in raw_geo_countries if str(c).strip()] or None

        published_at_from = kwargs.get("published_at_from")
        if not isinstance(published_at_from, str) or not published_at_from.strip():
            published_at_from = None
        else:
            published_at_from = published_at_from.strip()[:10]

        published_at_to = kwargs.get("published_at_to")
        if not isinstance(published_at_to, str) or not published_at_to.strip():
            published_at_to = None
        else:
            published_at_to = published_at_to.strip()[:10]

        time_filter_at_qdrant = kwargs.pop("time_filter_at_qdrant", True)
        post_filter_from = published_at_from
        post_filter_to = published_at_to
        if not time_filter_at_qdrant:
            # Many news points lack published_at; filter dates in Python after vector search.
            published_at_from = None
            published_at_to = None

        domains_substring = kwargs.get("domains_substring")
        if isinstance(domains_substring, str):
            domains_substring = domains_substring.strip() or None
        else:
            domains_substring = None

        has_filters = any(
            [
                doc_kind,
                doc_kinds,
                geo_country,
                geo_countries,
                published_at_from,
                published_at_to,
                domains_substring,
            ]
        )

        vector_search_mode = kwargs.pop("vector_search_mode", None)
        if vector_search_mode is None:
            vector_search_mode = _env("RAG_QDRANT_VECTOR_SEARCH_MODE", "legacy") or "legacy"
        vector_search_mode = str(vector_search_mode).strip().lower()

        client = self._get_client()
        collection = self.collection_name
        use_hybrid = _collection_has_hybrid_sparse(collection)
        if use_hybrid:
            overfetch = 1
        else:
            overfetch = int(kwargs.get("overfetch_multiplier", 8 if has_filters else 1))
            overfetch = max(1, min(overfetch, 50))
        fetch_n = max(top_k, top_k * overfetch)
        exclude_section_roles = _research_exclude_roles() if _is_research_collection(collection) else frozenset()

        # Skip filter conditions whose key isn't indexed on this collection:
        # Qdrant otherwise rejects the whole query with a 400. Falls back to
        # "no constraint on indexed_fields" if the corpus profile lookup fails.
        try:
            from ml.rag.scripts.qdrant_collection_specs import (
                indexed_fields_for_corpus,
                indexed_fields_on_collection,
            )

            corpus_key = profile_for_collection(collection).corpus
            indexed_fields = indexed_fields_for_corpus(corpus_key)
            live = indexed_fields_on_collection(client, collection)
            if live is not None:
                indexed_fields = indexed_fields & live
        except Exception:
            indexed_fields = None

        try:
            q_filter = build_qdrant_filter(
                doc_kind=doc_kind,
                doc_kinds=doc_kinds,
                geo_country=geo_country if not geo_countries else None,
                geo_countries=geo_countries,
                published_at_from=published_at_from,
                published_at_to=published_at_to,
                domains_substring=domains_substring,
                exclude_section_roles=exclude_section_roles,
                indexed_fields=indexed_fields,
            )
        except Exception:
            q_filter = None

        query_kwargs: dict[str, Any] = {"limit": fetch_n, "with_payload": True}
        if q_filter is not None:
            query_kwargs["query_filter"] = q_filter

        # ----- legacy: single unnamed vector --------------------------------
        if vector_search_mode == "legacy":
            self._ensure_collection()
            query_vec = _embed_texts_for_indexing(
                [query], model_id=self._embed_model_id, mode=self._embed_mode, is_query=True
            )[0]
            resp = client.query_points(
                collection_name=collection,
                query=query_vec,
                **query_kwargs,
            )
            hits = resp.points or []

        # ----- dual: sentence + semantic (news/research JSONL loaders) -----
        elif vector_search_mode == "dual":
            using = _env("RAG_QDRANT_DUAL_QUERY_USING", "both").lower()
            if using == "sentence":
                vector_names: tuple[str, ...] = ("sentence",)
            elif using == "semantic":
                vector_names = ("semantic",)
            else:
                vector_names = DUAL_SENTENCE_SEMANTIC
            hits = self._query_named_vectors(query, vector_names, fetch_n, q_filter, top_k=top_k)

        # ----- dense_named: single named ``dense`` vector (news_data) -----
        elif vector_search_mode == "dense_named":
            hits = self._query_named_vectors(query, ("dense",), fetch_n, q_filter, top_k=top_k)

        # ----- sentence_named: single named ``sentence`` vector ------------
        elif vector_search_mode == "sentence_named":
            hits = self._query_named_vectors(query, ("sentence",), fetch_n, q_filter, top_k=top_k)

        # ----- research_dual: abstract_vector + content_vector --------------
        elif vector_search_mode == "research_dual":
            using = _env("RAG_QDRANT_RESEARCH_QUERY_USING", "content").lower()
            vector_names = _RESEARCH_USING.get(using, ("content_vector",))
            hits = self._query_named_vectors(query, vector_names, fetch_n, q_filter, top_k=top_k)

        # ----- ota_triple: insight / metric / recommendation ----------------
        elif vector_search_mode == "ota_triple":
            using = _env("RAG_QDRANT_OTA_QUERY_USING", "merge").lower()
            vector_names = _OTA_USING.get(using, OTA_VECTORS)
            hits = self._query_named_vectors(query, vector_names, fetch_n, q_filter, top_k=top_k)

        # ----- bq_triple: table / schema / business -------------------------
        elif vector_search_mode == "bq_triple":
            using = _env("RAG_QDRANT_BQ_QUERY_USING", "merge").lower()
            vector_names = _BQ_USING.get(using, BQ_VECTORS)
            hits = self._query_named_vectors(query, vector_names, fetch_n, q_filter, top_k=top_k)

        else:
            raise ValueError(
                f"Unknown vector_search_mode: {vector_search_mode!r} "
                "(use legacy, dense_named, dual, sentence_named, research_dual, ota_triple, or bq_triple)"
            )

        items: list[dict[str, Any]] = []
        for h in hits or []:
            payload = h.payload or {}
            if not isinstance(payload, dict):
                payload = {}
            content = str(payload.get("content") or "").strip()
            meta = _safe_payload({k: v for k, v in payload.items() if k != "content"})
            if not self._metadata_passes_filters(
                meta,
                doc_kind=doc_kind,
                doc_kinds=doc_kinds,
                geo_country=geo_country if not geo_countries else None,
                geo_countries=geo_countries,
                published_at_from=post_filter_from,
                published_at_to=post_filter_to,
                domains_substring=domains_substring,
                exclude_section_roles=exclude_section_roles,
            ):
                continue
            items.append(
                {
                    "content": content,
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                    "metadata": meta,
                    "source": "vector",
                }
            )
            if len(items) >= top_k:
                break

        items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        result = items[:top_k]
        update_current_span_metadata(
            {
                "corpus": profile_for_collection(collection).corpus,
                "collection": collection,
                "top_k": top_k,
                "hit_count": len(result),
                "vector_search_mode": vector_search_mode,
                "hybrid": use_hybrid,
                "geo_filter": bool(geo_country or geo_countries),
                "latency_ms": trace_elapsed_ms(t0),
            }
        )
        return result
