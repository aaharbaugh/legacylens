"""
FastAPI app: health + vector search query.
Run from repo root: uvicorn backend.api.main:app --reload
"""
import logging
import os
import sys

# Load .env into os.environ so GOOGLE_APPLICATION_CREDENTIALS and other vars are available to Google libs
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Reduce reranker/HuggingFace noise (progress bars, HTTP logs, symlink warning, load report)
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Ensure backend activity is visible in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
for name in ("backend", "uvicorn.access"):
    logging.getLogger(name).setLevel(logging.INFO)
# Suppress noisy third-party logs (reranker load report, HuggingFace, httpx, GenAI AFC)
for name in ("httpx", "httpcore", "sentence_transformers", "transformers", "huggingface_hub", "google_genai.models"):
    logging.getLogger(name).setLevel(logging.ERROR)

import hashlib
import json
import time
from pathlib import Path
from typing import Any

# Prefetch cache: key (query_trimmed, folder) -> { "chunks": list[dict], "ts": float }
_prefetch_cache: dict[tuple[str, str], dict[str, Any]] = {}
# Prefetch cooldown: (query_trimmed,) -> last_embed_ts (skip re-embedding within cooldown window)
_prefetch_last_embed: dict[tuple[str, ...], float] = {}

from fastapi import Depends, FastAPI, Header, HTTPException, Response

logger = logging.getLogger(__name__)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api.ask_service import run_api_ask, run_api_ask_stream
from backend.api.chat_service import run_query_chat
from backend.config import settings
from backend.api.retrieval_utils import (
    hits_to_retrieved_chunks as _hits_to_retrieved_chunks,
    rank_hits_by_file as _rank_hits_by_file,
)
from backend.api.request_logs import (
    append_request_log as _append_request_log,
    load_request_logs as _load_request_logs,
    recent_request_logs,
)
from backend.api.schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    AdminReingestRequest,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    PrefetchRequest,
    PrefetchResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from backend.ingestion.embedder import Embedder
from backend.ingestion.pipeline import run_pipeline
from backend.ingestion.vector_store import (
    count,
    get_chunks_for_file,
    get_vector_store,
    hybrid_search,
    list_chunks,
    reset_collection,
    warm_bm25_index_cache,
)

app = FastAPI(title="LegacyLens", description="RAG over legacy codebases")

# RAG chat system prompt: tune this for answer quality (see docs/CALIBRATION.md)
CHAT_SYSTEM_PROMPT = """You are an expert GnuCOBOL architect. You are provided with retrieved context from the compiler's C source code, COBOL test suites, and documentation.

Answer the user's question based strictly on this context. Do not speculate beyond what the chunks show.

- Cite sources as [1], [2], [3] with file path and line range.
- Use ```c or ```cobol for code examples.
- For list questions (who, what, which), enumerate every relevant item found across all chunks.
- If the context is insufficient to answer, say so clearly and suggest a more specific query.
"""

_env_frontend = os.environ.get("FRONTEND_DIR", "").strip()
FRONTEND_DIR = Path(_env_frontend) if _env_frontend else Path(__file__).resolve().parents[2] / "frontend"
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"

if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="assets")

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _make_session_token(admin_token: str) -> str:
    """Deterministic session token derived from the admin token."""
    return hashlib.sha256((admin_token + "|session").encode("utf-8")).hexdigest()


def require_admin(
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> None:
    """
    Accept either:
    - X-Admin-Token header equal to ADMIN_TOKEN, or
    - Authorization: Bearer <session_token> returned from /admin/login.
    """
    if not settings.admin_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Direct admin token
    if x_admin_token and x_admin_token == settings.admin_token:
        return

    # Bearer session token
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token == _make_session_token(settings.admin_token):
            return

    raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the dedicated frontend app."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/app", include_in_schema=False)
def app_page() -> FileResponse:
    """Alias route for the dedicated frontend app."""
    return root()


@app.get("/admin")
def admin_page() -> RedirectResponse:
    """Legacy admin route: redirect to the unified frontend app."""
    return RedirectResponse(url="/app", status_code=307)


@app.post("/admin/login", response_model=AdminLoginResponse)
def admin_login(req: AdminLoginRequest) -> AdminLoginResponse:
    """
    Simple sign-in: client posts the ADMIN_TOKEN and receives a bearer token.
    Subsequent admin calls can use:
      Authorization: Bearer <access_token>
    instead of sending the raw ADMIN_TOKEN.
    """
    if not settings.admin_token:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN is not configured")
    if req.token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    access_token = _make_session_token(settings.admin_token)
    return AdminLoginResponse(access_token=access_token)


@app.get("/admin/runtime-config", dependencies=[Depends(require_admin)])
def admin_runtime_config() -> dict[str, Any]:
    """Return active runtime config values used by ingestion."""
    return {
        "code_root": str(settings.code_root) if settings.code_root else None,
        "code_extensions": settings.code_extensions,
        "max_file_size_mb": settings.max_file_size_mb,
        "ingest_text_only": settings.ingest_text_only,
        "cwd": str(Path.cwd()),
    }


@app.get("/admin/logs", dependencies=[Depends(require_admin)])
def admin_logs() -> dict[str, Any]:
    """Return recent request logs (query/chat latency breakdown) for the admin panel. Newest first."""
    logs = recent_request_logs()
    return {"logs": logs, "count": len(logs)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Any]:
    """
    Check whether Vertex is configured and likely in use for embeddings and LLM.
    - embeddings: "vertex" if GOOGLE_CLOUD_PROJECT is set (and credentials work at runtime), else "pseudo"
    - credentials_set: true if GOOGLE_APPLICATION_CREDENTIALS is set (local key file)
    - llm_enabled: true if LLM is configured (model + project)
    - use_reranker: true if USE_RERANKER env is true (cross-encoder rerank for chat)
    """
    project = settings.google_cloud_project
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    app_build = os.environ.get("APP_BUILD", "").strip() or None
    return {
        "status": "ok",
        "embeddings": "vertex" if project else "pseudo",
        "google_cloud_project": project or None,
        "credentials_set": bool(creds_env),
        "llm_enabled": bool(settings.llm_enabled and settings.llm_model and project),
        "llm_model": settings.llm_model if settings.llm_enabled else None,
        "app_build": app_build,
        "min_vector_score": settings.min_vector_score,
        "bm25_min_score": settings.bm25_min_score,
        "use_hybrid_search": settings.use_hybrid_search,
        "use_reranker": settings.use_reranker,
        "embed_metadata_prefix": settings.embed_metadata_prefix,
        "query_chat_final_k": settings.query_chat_final_k,
        "query_chat_top_k": settings.query_chat_top_k,
        "query_chat_max_chunks_per_file": settings.query_chat_max_chunks_per_file,
        "embed_batch_size": settings.embed_batch_size,
        "prefetch_enabled": getattr(settings, "prefetch_enabled", True),
        "prefetch_min_query_length": getattr(settings, "prefetch_min_query_length", 12),
        "prefetch_cooldown_sec": getattr(settings, "prefetch_cooldown_sec", 15.0),
    }


@app.get("/health/db")
def health_db() -> dict[str, Any]:
    """
    Diagnostic: where data is stored (local vs cloud) and current point count.
    Use this to verify ingestion is writing to the right place.
    """
    from backend.ingestion.vector_store import count, get_vector_store

    is_local = settings.qdrant_use_local()
    storage = (
        f"local: {settings.qdrant_local_path.resolve()}"
        if is_local
        else f"cloud: {settings.qdrant_url}"
    )
    try:
        client = get_vector_store()
        n = count(client)
        return {
            "storage_mode": "local" if is_local else "cloud",
            "storage_path_or_url": str(settings.qdrant_local_path.resolve()) if is_local else settings.qdrant_url,
            "collection": settings.qdrant_collection,
            "point_count": n,
            "status": "ok",
        }
    except Exception as e:
        return {
            "storage_mode": "local" if is_local else "cloud",
            "storage_path_or_url": str(settings.qdrant_local_path) if is_local else settings.qdrant_url,
            "error": str(e),
            "status": "error",
        }


def _get_prefetch_cache_key(query: str, folder: str) -> tuple[str, str]:
    return (query.strip(), (folder or "").strip())


def _get_cached_prefetch_chunks(query: str, folder: str) -> list[RetrievedChunk] | None:
    """Return cached chunks if present and not expired."""
    key = _get_prefetch_cache_key(query, folder)
    entry = _prefetch_cache.get(key)
    if not entry:
        return None
    ttl = getattr(settings, "prefetch_cache_ttl_sec", 120) or 120
    if time.time() - entry["ts"] > ttl:
        del _prefetch_cache[key]
        return None
    return entry.get("chunks")


def _set_prefetch_cache(query: str, folder: str, chunks: list[RetrievedChunk]) -> None:
    key = _get_prefetch_cache_key(query, folder)
    _prefetch_cache[key] = {"chunks": chunks, "ts": time.time()}


@app.post("/api/prefetch", response_model=PrefetchResponse)
def api_prefetch(req: PrefetchRequest) -> PrefetchResponse:
    """
    Prefetch: embed query, run folder-filtered hybrid search, rerank top 15, cache top 3.
    No LLM call. Respects prefetch_enabled, prefetch_min_query_length, prefetch_cooldown_sec to reduce quota.
    """
    query = (req.query or "").strip()
    if not query:
        return PrefetchResponse(ok=True)
    if not getattr(settings, "prefetch_enabled", True):
        return PrefetchResponse(ok=True)
    min_len = getattr(settings, "prefetch_min_query_length", 12) or 0
    if len(query) < min_len:
        return PrefetchResponse(ok=True)
    cooldown = getattr(settings, "prefetch_cooldown_sec", 15.0) or 0
    if cooldown > 0:
        key_embed = (query,)
        last = _prefetch_last_embed.get(key_embed, 0)
        if time.time() - last < cooldown:
            # Reuse existing cache if present; do not call Vertex
            existing = _get_cached_prefetch_chunks(query, (req.folder or "").strip() or "")
            if existing is not None:
                return PrefetchResponse(ok=True)
    folder = (req.folder or "").strip() or None
    embedder = get_embedder()
    try:
        vectors = embedder.embed_texts([query])
    except Exception as e:
        logger.warning("Prefetch embed failed: %s", e)
        return PrefetchResponse(ok=True)
    if cooldown > 0:
        _prefetch_last_embed[(query,)] = time.time()
    if not vectors or not vectors[0]:
        return PrefetchResponse(ok=True)
    client = get_vector_store()
    top_k = getattr(settings, "prefetch_candidates_k", 15) or 15
    cache_top = getattr(settings, "prefetch_cache_top_k", 3) or 3
    timings: dict[str, float] = {}
    use_reranker = getattr(settings, "ask_use_reranker", False)
    hits = hybrid_search(
        client,
        vectors[0],
        query,
        top_k=top_k,
        final_k=cache_top,
        score_threshold=settings.query_score_threshold,
        source_type=None,
        folder=folder,
        use_reranker=use_reranker,
        out_timings=timings,
    )
    chunks = _hits_to_retrieved_chunks(hits, settings.chat_snippet_max_lines)
    _set_prefetch_cache(query, folder or "", chunks)
    logger.info("Prefetch cached %d chunks for query len=%d folder=%s", len(chunks), len(query), folder)
    return PrefetchResponse(ok=True)


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest, response: Response) -> AskResponse:
    return await run_api_ask(req, _resolve_ask_chunks)


def _resolve_ask_chunks(
    query: str,
    folder: str | None,
    out_timings: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Resolve chunks for ask (cache or live retrieval). Optionally fill out_timings with embed_ms, search_ms, rerank_ms."""
    chunks = _get_cached_prefetch_chunks(query, folder or "")
    if chunks:
        return chunks
    t0 = time.perf_counter()
    embedder = get_embedder()
    vectors = embedder.embed_texts([query])
    if not vectors or not vectors[0]:
        return []
    if out_timings is not None:
        out_timings["embed_ms"] = (time.perf_counter() - t0) * 1000
    client = get_vector_store()
    top_k = getattr(settings, "prefetch_cache_top_k", 3) or 3
    use_reranker = getattr(settings, "ask_use_reranker", False)
    hits = hybrid_search(
        client,
        vectors[0],
        query,
        top_k=getattr(settings, "prefetch_candidates_k", 15) or 15,
        final_k=top_k,
        score_threshold=settings.query_score_threshold,
        source_type=None,
        folder=folder,
        use_reranker=use_reranker,
        out_timings=out_timings,
    )
    hits = _rank_hits_by_file(hits, settings.query_chat_max_chunks_per_file, top_k)
    return _hits_to_retrieved_chunks(hits, settings.chat_snippet_max_lines)


@app.post("/api/ask/stream")
async def api_ask_stream(req: AskRequest):
    return await run_api_ask_stream(req, _resolve_ask_chunks)


@app.post("/query/chat", response_model=ChatResponse)
def query_chat(req: ChatRequest, response: Response) -> ChatResponse:
    return run_query_chat(req, response, get_embedder, get_vector_store, CHAT_SYSTEM_PROMPT)


@app.get("/find-file")
def find_file(name: str) -> dict[str, Any]:
    """
    Search for a file by name under CODE_ROOT. Returns first match path or 404.
    Used for clickable #include to resolve libintl.h -> lib/libintl.h etc.
    """
    base = Path.cwd()
    if settings.code_root and settings.code_root.is_dir():
        base = settings.code_root.resolve()
    name_clean = name.strip().lstrip("/")
    if not name_clean or ".." in name_clean:
        raise HTTPException(status_code=400, detail="Invalid filename")
    found: Path | None = None
    try:
        for p in base.rglob(name_clean.split("/")[-1]):
            if not p.is_file():
                continue
            rel = p.relative_to(base)
            rel_posix = rel.as_posix()
            if p.name == name_clean or rel_posix == name_clean or rel_posix.endswith("/" + name_clean):
                found = p
                break
    except Exception as e:
        logger.warning("find-file search failed: %s", e)
    if found is None:
        raise HTTPException(status_code=404, detail=f"File not found: {name}")
    rel = found.relative_to(base)
    return {"path": rel.as_posix(), "name": found.name}


@app.get("/file-content")
def get_file_content(
    path: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    """
    Read lines start_line..end_line (1-based) from a file.
    When deployed (no CODE_ROOT): fetches from Qdrant chunks.
    When local: tries filesystem first, then Qdrant.
    """
    from backend.ingestion.vector_store import get_file_lines_from_chunks, get_vector_store

    # When deployed, CODE_ROOT is never set—go straight to Qdrant
    has_filesystem = settings.code_root and settings.code_root.is_dir()
    if has_filesystem:
        base = settings.code_root.resolve()
        if Path(path).is_absolute():
            raise HTTPException(status_code=400, detail="Absolute paths are not allowed")
        fp = (base / path).resolve()
        try:
            fp.relative_to(base)
        except ValueError:
            raise HTTPException(status_code=400, detail="Path is outside code root")
        if fp.is_file():
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    raise HTTPException(status_code=500, detail=str(e))
                lines = text.splitlines()
                start = max(0, start_line - 1)
                end = min(len(lines), end_line)
                if start >= end:
                    return {"content": "", "start_line": start_line, "end_line": end_line, "total_lines": len(lines)}
                snippet = "\n".join(lines[start:end])
                return {"content": snippet, "start_line": start + 1, "end_line": end, "total_lines": len(lines)}

    # Qdrant fallback (always used when deployed; fallback when local file missing)
    try:
        client = get_vector_store()
        result = get_file_lines_from_chunks(client, path, start_line, end_line)
        if result:
            content, total_lines = result
            return {"content": content, "start_line": start_line, "end_line": end_line, "total_lines": total_lines}
    except Exception as e:
        logger.warning("file-content Qdrant fallback failed for %s: %s", path, e)
    raise HTTPException(status_code=404, detail=f"File not found: {path}")


@app.get("/file-chunks")
def get_file_chunks(path: str) -> dict[str, Any]:
    """
    Return all chunks for a file from Qdrant. Frontend can use this to build expanded view
    client-side when file-content is slow or fails.
    """
    from backend.ingestion.vector_store import get_chunks_for_file, get_vector_store

    path_norm = (path or "").replace("\\", "/")
    if not path_norm:
        raise HTTPException(status_code=400, detail="path required")
    try:
        client = get_vector_store()
        chunks = get_chunks_for_file(client, path_norm)
        return {"path": path_norm, "chunks": chunks}
    except Exception as e:
        logger.warning("file-chunks failed for %s: %s", path, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/tags")
def list_tag_options() -> dict[str, list[str]]:
    """Return known tag prefixes and role tags for filter UI."""
    return {
        "role_tags": [
            "file_io",
            "display_io",
            "call_external",
            "control_flow",
            "data_definition",
            "business_logic",
            "error_handling",
        ],
        "structural_prefixes": ["div:", "data:", "para:", "section:", "program:"],
    }


@app.get("/chunks")
def list_stored_chunks(
    limit: int = 50,
    offset: str | None = None,
    file_path: str | None = None,
    source_type: str = "all",
) -> dict[str, Any]:
    """
    List what's in the vector DB (browse stored chunks).
    - limit: max chunks to return (default 50)
    - offset: pagination offset from a previous response's next_offset
    - file_path: filter by substring in file path (e.g. 'sample' or '.cbl')
    """
    client = get_vector_store()
    chunks, next_offset = list_chunks(
        client,
        limit=min(limit, 200),
        offset=offset,
        file_path_contains=file_path,
        source_type=source_type,
    )
    total = count(client)
    return {
        "total_chunks": total,
        "returned": len(chunks),
        "next_offset": next_offset,
        "chunks": chunks,
    }


@app.on_event("startup")
def startup():
    _load_request_logs()
    logger.info("LegacyLens API ready – POST /query, /admin/reingest, etc.")
    # Warn loudly if BM25 index is missing — hybrid search silently degrades without it
    bm25_docs = warm_bm25_index_cache()
    if bm25_docs <= 0:
        logger.warning(
            "WARNING: BM25 index not found at '%s'. "
            "Hybrid search is DISABLED — only vector search will be used. "
            "Run ingestion to build the index.",
            settings.bm25_index_path,
        )
    else:
        logger.info("BM25 index loaded: %d docs", bm25_docs)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, response: Response) -> QueryResponse:
    """Embed the query and run hybrid search (vector + BM25 + RRF); return top-k chunks."""
    t_total = time.perf_counter()
    logger.info("Query: %s", req.query[:60] + "..." if len(req.query) > 60 else req.query)
    embedder = get_embedder()
    t0 = time.perf_counter()
    try:
        vectors = embedder.embed_texts([req.query])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    if not vectors or not vectors[0]:
        raise HTTPException(status_code=500, detail="No embedding returned")
    embed_ms = (time.perf_counter() - t0) * 1000
    client = get_vector_store()
    top_k = req.top_k if req.top_k is not None else settings.query_final_k
    score_threshold = (
        req.score_threshold if req.score_threshold is not None else settings.query_score_threshold
    )
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    hits = hybrid_search(
        client,
        vectors[0],
        req.query,
        top_k=settings.query_top_k,
        final_k=top_k,
        score_threshold=score_threshold,
        source_type=req.source_type if req.source_type != "all" else None,
        tags_filter=req.tags,
        use_reranker=req.use_reranker,
        out_timings=timings,
    )
    search_ms = (time.perf_counter() - t0) * 1000
    hits = _rank_hits_by_file(
        hits,
        settings.query_chat_max_chunks_per_file,
        top_k,
    )
    results = []
    for h in hits:
        p = h.get("payload") or {}
        results.append(
            RetrievedChunk(
                id=h.get("id", ""),
                score=h.get("score", 0.0),
                vector_score=h.get("vector_score"),
                file_path=p.get("file_path", ""),
                start_line=p.get("start_line", 0),
                end_line=p.get("end_line", 0),
                division=p.get("division"),
                section_name=p.get("section_name"),
                paragraph_name=p.get("paragraph_name"),
                code_snippet=p.get("code_snippet", ""),
                language=p.get("language", "COBOL"),
                source_type=p.get("source_type", "code"),
            )
        )
    total_ms = (time.perf_counter() - t_total) * 1000
    rerank_ms = timings.get("rerank_ms")
    logger.info(
        "query_latency_ms=%.0f embed_ms=%.0f search_ms=%.0f rerank_ms=%s",
        total_ms,
        embed_ms,
        search_ms,
        f"{rerank_ms:.0f}" if rerank_ms is not None else "n/a",
    )
    _append_request_log({
        "type": "query",
        "total_ms": round(total_ms, 0),
        "embed_ms": round(embed_ms, 0),
        "search_ms": round(search_ms, 0),
        "rerank_ms": round(rerank_ms, 0) if rerank_ms is not None else None,
    })
    response.headers["X-Request-Duration-Ms"] = str(int(round(total_ms)))
    return QueryResponse(query=req.query, results=results)


@app.post("/admin/reset-db", dependencies=[Depends(require_admin)])
def admin_reset_db() -> dict[str, Any]:
    """
    Drop and recreate the vector collection.
    Useful when switching to a new codebase before reingesting.
    """
    client = get_vector_store()
    reset_collection(client)
    total = count(client)
    return {"status": "ok", "total_chunks": total}


@app.post("/admin/reingest", dependencies=[Depends(require_admin)])
def admin_reingest(req: AdminReingestRequest) -> dict[str, Any]:
    """
    Reset the collection and re-run ingestion.
    - If code_root is provided in the body, use that.
    - Otherwise, fall back to Settings.code_root.
    """
    logger.info("Reingest started (code_root=%s)", req.code_root or settings.code_root)
    code_root = req.code_root
    root_path = Path(code_root) if code_root else settings.code_root
    if root_path and not root_path.is_absolute():
        root_path = Path.cwd() / root_path
    if not root_path:
        raise HTTPException(
            status_code=400,
            detail="No code_root provided and CODE_ROOT is not configured",
        )
    # Try the given path; if it isn't a directory, also try a nested
    # folder with the same name (handles layouts like foo/foo/...).
    if not root_path.is_dir():
        alt = root_path / root_path.name
        if alt.is_dir():
            root_path = alt
        else:
            raise HTTPException(
                status_code=400,
                detail=f"code_root '{code_root}' is not a directory from server cwd={Path.cwd()}. "
                f"Resolved path: {root_path.resolve()}",
            )
    client = get_vector_store()
    reset_collection(client)
    extensions = settings.extensions_list()
    if req.code_extensions is not None:
        raw = req.code_extensions.strip()
        if not raw or raw == "*":
            extensions = []
        else:
            extensions = [e.strip().lstrip(".") for e in raw.split(",") if e.strip()]
    batch_size = req.batch_size or settings.embed_batch_size
    max_files = req.max_files if req.max_files is not None else settings.max_files
    # Reuse the same Qdrant client as the API to avoid multiple
    # local instances accessing the same .qdrant_data directory.
    files, chunks = run_pipeline(
        root_path,
        client=client,
        extensions=extensions,
        batch_size=batch_size,
        max_files=max_files,
    )
    total = count(client)
    logger.info("Reingest complete: %d files, %d chunks", files, total)
    storage_info = "local (.qdrant_data)" if settings.qdrant_use_local() else f"cloud ({settings.qdrant_url})"
    return {
        "status": "ok",
        "files_ingested": files,
        "chunks_upserted": chunks,
        "total_chunks": total,
        "code_root": str(root_path),
        "code_extensions": "*" if not extensions else ",".join(extensions),
        "batch_size": batch_size,
        "max_files": max_files,
        "storage": storage_info,
    }
