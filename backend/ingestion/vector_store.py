"""
Qdrant vector store: create collection, upsert points, search.
Hybrid search: vector + BM25 with RRF fusion, optional reranking.
"""
import hashlib
import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import settings
from backend.ingestion.embedder import VECTOR_SIZE

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legacylens-chunks"


def _rrf_fusion(
    rank_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked id lists using Reciprocal Rank Fusion. Returns [(id, score), ...]."""
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, pid in enumerate(rank_list):
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _to_payload_dict(payload: Any) -> dict[str, Any]:
    """Ensure payload is a plain dict (Qdrant may return Payload model or dict)."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return dict(payload.model_dump())
    try:
        return dict(payload)
    except (TypeError, ValueError):
        return {}


def make_point_id(chunk: dict[str, Any]) -> str:
    """Stable ID for idempotent upserts."""
    key = (
        chunk.get("file_path", "")
        + "|"
        + str(chunk.get("start_line"))
        + "|"
        + str(chunk.get("end_line"))
        + "\n"
        + (chunk.get("code_snippet") or "")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def get_client() -> QdrantClient:
    """Build Qdrant client from config: local path or remote URL."""
    if settings.qdrant_use_local():
        path = settings.qdrant_local_path
        return QdrantClient(path=str(path))
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection(client: QdrantClient, dim: int = VECTOR_SIZE) -> None:
    """Create collection if it doesn't exist; recreate not (preserve data)."""
    coll = settings.qdrant_collection
    try:
        client.get_collection(coll)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=coll,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )
    logger.info("Created collection %s with dim=%s", coll, dim)


# Max points per upsert request (larger = fewer round-trips; 50 is usually fine for Qdrant Cloud)
UPSERT_BATCH_SIZE = 50


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> int:
    """Upsert (idempotent) chunks with payloads and vectors. Returns count upserted."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    points = []
    for c, vec in zip(chunks, vectors):
        payload = {k: v for k, v in c.items() if k != "code_snippet"}
        payload["code_snippet"] = (c.get("code_snippet") or "")[:65535]  # avoid huge payloads
        pid = make_point_id(c)
        points.append(
            qmodels.PointStruct(
                id=pid,
                vector=vec,
                payload=payload,
            )
        )
    coll = settings.qdrant_collection
    total = len(points)
    try:
        for i in range(0, total, UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            client.upsert(collection_name=coll, points=batch)
        logger.info("Upserted %d points to %s (in batches of %d)", total, coll, UPSERT_BATCH_SIZE)
        return total
    except Exception as e:
        logger.error("Upsert failed for %d points to %s: %s", total, coll, e)
        raise


def search(
    client: QdrantClient,
    vector: list[float],
    *,
    limit: int = 15,
    score_threshold: float | None = 0.0,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return top-k hits with payloads. Each hit has 'id', 'score', 'payload'."""
    coll = settings.qdrant_collection
    query_filter = None
    if source_type and source_type != "all":
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_type",
                    match=qmodels.MatchValue(value=source_type),
                )
            ]
        )
    response = client.query_points(
        collection_name=coll,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    raw_points = getattr(response, "points", None) or getattr(response, "result", None) or []
    out = []
    for p in raw_points:
        out.append({
            "id": str(p.id) if hasattr(p, "id") else "",
            "score": float(getattr(p, "score", 0) or 0),
            "payload": _to_payload_dict(getattr(p, "payload", None)),
        })
    return out


def hybrid_search(
    client: QdrantClient,
    vector: list[float],
    query_text: str,
    *,
    top_k: int = 25,
    final_k: int = 10,
    score_threshold: float = 0.0,
    source_type: str | None = None,
    tags_filter: list[str] | None = None,
    use_reranker: bool = False,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: vector + BM25 (if available) with RRF, optional reranking.
    Returns list of hits with 'id', 'score', 'payload'.
    """
    from backend.ingestion.bm25_index import BM25Index

    rank_lists: list[list[str]] = []
    id_to_hit: dict[str, dict] = {}

    # 1. Vector search
    vec_hits = search(
        client,
        vector,
        limit=top_k,
        score_threshold=score_threshold if score_threshold > 0 else 0.0,
        source_type=source_type,
    )
    vec_ids = [h["id"] for h in vec_hits if h["id"]]
    for h in vec_hits:
        id_to_hit[h["id"]] = dict(h, vector_score=h.get("score"))
    rank_lists.append(vec_ids)

    # 2. BM25 search if hybrid enabled
    if settings.use_hybrid_search and query_text.strip():
        bm25 = BM25Index.load()
        if bm25:
            id_to_payload = {h["id"]: h.get("payload", {}) for h in vec_hits}
            bm25_hits = bm25.search(
                query_text,
                limit=top_k,
                source_type=source_type,
                tags_filter=tags_filter,
                id_to_payload=id_to_payload,
            )
            if bm25_hits:
                bm25_ids = [pid for pid, _ in bm25_hits]
                rank_lists.append(bm25_ids)
                bm25_payloads = getattr(bm25, "id_to_payload", None) or {}
                for pid, _ in bm25_hits:
                    if pid not in id_to_hit:
                        payload = bm25_payloads.get(pid) or {}
                        id_to_hit[pid] = {"id": pid, "score": 0.0, "payload": payload, "vector_score": None}

    # 3. RRF fusion
    fused = _rrf_fusion(rank_lists, k=settings.rrf_k)

    # 4. Apply tags_filter post-filter to vector-only results
    if tags_filter and len(rank_lists) == 1:
        filtered = []
        for pid, rrf_score in fused:
            hit = id_to_hit.get(pid)
            if not hit:
                continue
            chunk_tags = set((hit.get("payload") or {}).get("tags") or [])
            if any(t in chunk_tags for t in tags_filter):
                filtered.append((pid, rrf_score))
        fused = filtered

    # 5. Build result list and fetch missing payloads
    result_ids = [pid for pid, _ in fused[: top_k * 2]]
    missing = [pid for pid in result_ids if pid not in id_to_hit or not (id_to_hit[pid].get("payload"))]
    if missing:
        try:
            pts = client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=missing[:50],
                with_payload=True,
            )
            for p in pts or []:
                pid = str(p.id) if hasattr(p, "id") else ""
                if pid:
                    id_to_hit[pid] = {
                        "id": pid,
                        "score": 0.0,
                        "payload": _to_payload_dict(getattr(p, "payload", None)),
                        "vector_score": None,
                    }
        except Exception as e:
            logger.debug("Could not retrieve missing payloads: %s", e)

    ordered = []
    min_vec = settings.min_vector_score
    for pid, rrf_score in fused[: top_k * 2]:
        hit = id_to_hit.get(pid)
        if not hit:
            continue
        vec_score = hit.get("vector_score")
        if vec_score is not None and vec_score < min_vec:
            continue  # drop low-relevance vector hits (e.g. vec 0.09 junk)
        hit = dict(hit)
        hit["score"] = rrf_score
        hit["vector_score"] = vec_score
        ordered.append(hit)
        if len(ordered) >= final_k:
            break

    # Fallback: if min_vector_score filtered everything, relax and return best available
    if not ordered and fused:
        min_vec = 0.0
        for pid, rrf_score in fused[: top_k * 2]:
            hit = id_to_hit.get(pid)
            if not hit:
                continue
            vec_score = hit.get("vector_score")
            if vec_score is not None and vec_score < min_vec:
                continue
            hit = dict(hit)
            hit["score"] = rrf_score
            hit["vector_score"] = vec_score
            ordered.append(hit)
            if len(ordered) >= final_k:
                break
        if ordered:
            logger.info("Relaxed min_vector_score; returning %d results (scores may be low)", len(ordered))

    if not ordered:
        logger.warning(
            "hybrid_search: 0 results (vec_hits=%d, fused=%d, collection=%s)",
            len(vec_hits),
            len(fused),
            settings.qdrant_collection,
        )

    # 6. Optional reranking
    if use_reranker and settings.use_reranker and ordered and query_text.strip():
        ordered = _rerank(query_text, ordered)

    return ordered[:final_k]


def _rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rerank hits using cross-encoder. Tries FlagEmbedding first, falls back to
    sentence-transformers (easier to install on Windows; avoids zlib-state).
    """
    pairs = [(query, h.get("payload", {}).get("code_snippet") or "") for h in hits]

    # Try FlagEmbedding first
    try:
        from FlagEmbedding import FlagReranker

        if not hasattr(_rerank, "_model"):
            _rerank._model = FlagReranker(settings.reranker_model, use_fp16=True)
        model = _rerank._model
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        combined = list(zip(hits, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in combined]
    except ImportError:
        pass  # Fall through to sentence-transformers
    except Exception as e:
        logger.warning("FlagEmbedding rerank failed: %s; trying fallback", e)

    # Fallback: sentence-transformers (no zlib-state, works on Windows)
    try:
        from sentence_transformers import CrossEncoder

        if not hasattr(_rerank, "_st_model"):
            _rerank._st_model = CrossEncoder(settings.reranker_model)
        model = _rerank._st_model
        scores = model.predict(pairs)
        try:
            scores = [float(s) for s in scores]
        except TypeError:
            scores = [float(scores)]
        combined = list(zip(hits, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in combined]
    except ImportError:
        logger.debug(
            "Neither FlagEmbedding nor sentence-transformers installed; skipping rerank. "
            "Install one: pip install sentence-transformers  (or FlagEmbedding)"
        )
        return hits
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
        return hits


def count(client: QdrantClient | None = None) -> int:
    """Return the number of points in the collection."""
    if client is None:
        client = get_client()
    coll = settings.qdrant_collection
    try:
        info = client.get_collection(coll)
        return info.points_count or 0
    except Exception:
        return 0


def get_file_lines_from_chunks(
    client: QdrantClient,
    file_path: str,
    start_line: int,
    end_line: int,
) -> tuple[str, int] | None:
    """
    Fetch lines start_line..end_line (1-based) from chunks stored in Qdrant.
    """
    all_chunks = get_chunks_for_file(client, file_path)
    if not all_chunks:
        return None
    all_chunks.sort(key=lambda c: (c.get("start_line") or 0))
    max_end = max(c.get("end_line") or 0 for c in all_chunks)
    total_lines = max_end
    line_to_content: dict[int, str] = {}
    for c in all_chunks:
        s, e = c.get("start_line") or 0, c.get("end_line") or 0
        snippet = (c.get("code_snippet") or "").splitlines()
        for i, line in enumerate(snippet):
            ln = s + i
            if start_line <= ln <= end_line and ln not in line_to_content:
                line_to_content[ln] = line
    if not line_to_content:
        return None
    lines = [line_to_content[ln] for ln in sorted(line_to_content.keys())]
    return ("\n".join(lines), total_lines)


def get_chunks_for_file(client: QdrantClient, file_path: str) -> list[dict[str, Any]]:
    """Return all chunks for a file. Uses same logic as get_file_lines_from_chunks."""
    path_norm = (file_path or "").replace("\\", "/")
    if not path_norm:
        return []
    all_chunks: list[dict[str, Any]] = []
    path_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="file_path", match=qmodels.MatchValue(value=path_norm))]
    )
    try:
        zero_vec = [0.0] * VECTOR_SIZE
        resp = client.query_points(
            collection_name=settings.qdrant_collection,
            query=zero_vec,
            query_filter=path_filter,
            limit=500,
            with_payload=True,
        )
        for p in getattr(resp, "points", []) or []:
            payload = _to_payload_dict(getattr(p, "payload", None))
            if (payload.get("file_path") or "").replace("\\", "/") == path_norm:
                all_chunks.append(payload)
    except Exception:
        pass
    if not all_chunks:
        try:
            offset = None
            for _ in range(20):
                result, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=path_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for p in result or []:
                    payload = _to_payload_dict(getattr(p, "payload", None))
                    if (payload.get("file_path") or "").replace("\\", "/") == path_norm:
                        all_chunks.append(payload)
                if offset is None:
                    break
        except Exception:
            pass
    if not all_chunks:
        offset = None
        for _ in range(400):
            chunks, next_offset = list_chunks(client, limit=100, offset=offset, file_path_contains=path_norm)
            for c in chunks:
                if (c.get("file_path") or "").replace("\\", "/") == path_norm:
                    all_chunks.append(c)
            if next_offset is None:
                break
            offset = next_offset
            if not chunks:
                break
    all_chunks.sort(key=lambda c: (c.get("start_line") or 0))
    return all_chunks


def list_chunks(
    client: QdrantClient,
    *,
    limit: int = 50,
    offset: str | None = None,
    file_path_contains: str | None = None,
    source_type: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Scroll through stored chunks. Returns (list of payloads, next_offset or None).
    If file_path_contains is set, filter payloads by substring (client-side filter after scroll).
    """
    coll = settings.qdrant_collection
    result, next_offset = client.scroll(
        collection_name=coll,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    out = []
    for p in result or []:
        payload = _to_payload_dict(getattr(p, "payload", None))
        if file_path_contains and file_path_contains not in (payload.get("file_path") or ""):
            continue
        if source_type and source_type != "all" and payload.get("source_type") != source_type:
            continue
        out.append({
            "id": str(p.id) if hasattr(p, "id") else "",
            **payload,
        })
    next_off = None
    if next_offset is not None:
        try:
            next_off = str(next_offset)
        except Exception:
            next_off = None
    return out, next_off


def reset_collection(client: QdrantClient, *, dim: int = VECTOR_SIZE) -> None:
    """
    Drop and recreate the configured collection.
    Removes BM25 index so it gets rebuilt on next reingest.
    """
    coll = settings.qdrant_collection
    try:
        client.delete_collection(coll)
    except Exception:
        pass
    client.create_collection(
        collection_name=coll,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )
    # Remove stale BM25 index
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        try:
            bm25_path.unlink()
            logger.info("Removed BM25 index %s", bm25_path)
        except OSError:
            pass
    logger.info("Reset collection %s with dim=%s", coll, dim)


# Singleton for API to use
_client: QdrantClient | None = None


def get_vector_store() -> QdrantClient:
    """Cached client for use from API."""
    global _client
    if _client is None:
        _client = get_client()
        ensure_collection(_client)
    return _client
