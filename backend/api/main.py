"""
FastAPI app: health + vector search query.
Run from repo root: uvicorn backend.api.main:app --reload
"""
import logging
import os
import sys

# Reduce reranker/HuggingFace noise (progress bars, HTTP logs, symlink warning)
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure backend activity is visible in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
for name in ("backend", "uvicorn.access"):
    logging.getLogger(name).setLevel(logging.INFO)
# Suppress noisy third-party logs (reranker, HuggingFace, httpx)
for name in ("httpx", "httpcore", "sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

import hashlib
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

logger = logging.getLogger(__name__)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.ingestion.embedder import Embedder
from backend.ingestion.pipeline import run_pipeline
from backend.ingestion.vector_store import (
    count,
    get_chunks_for_file,
    get_vector_store,
    hybrid_search,
    list_chunks,
    reset_collection,
)

app = FastAPI(title="LegacyLens", description="RAG over legacy codebases")

# RAG chat system prompt: tune this for answer quality (see docs/CALIBRATION.md)
CHAT_SYSTEM_PROMPT = """You are a senior code analyst for GnuCOBOL, an open-source COBOL compiler implemented in C. You are explaining code to a junior engineer: be clear, concrete, and helpful.

You will receive numbered code/document chunks from the codebase as context. Use ONLY this context to answer. Do not use external knowledge or guess.

**CRITICAL – Use the context you are given:** If the user asks about a specific keyword, verb, or concept (e.g. EVALUATE, PERFORM, 88-level, file I/O), search every chunk for that term. If you find it in any chunk, explain what the code or docs show—do NOT reply with "not enough information." Only say "The provided context does not contain enough information to answer this. Try rephrasing with specific keywords, paragraph or function names, or file paths from the codebase." when the term or topic does not appear in any of the chunks at all.

Rules:
1. **Grounding** – Answer from the context. If the topic appears in any chunk, summarize and cite it. Only use the "not enough information" line when the context is empty or none of the chunks mention the subject.

2. **Citations** – Always cite your sources with the chunk number in square brackets: [1], [2], [3]. When referring to specific code or behavior, include file path and line range (e.g. "in lib/file.cbl L45–60").

3. **Lists and enumerations** – When the question asks WHO, WHAT, or for a LIST (maintainers, contributors, options, flags, files, etc.), scan every chunk and list ALL relevant items. Do not stop at the first chunk or first section; include everyone/everything mentioned across the full context.

4. **Code** – When showing code examples, use fenced blocks with the right language: ```cobol or ```c.

5. **Precision** – Prefer specific file paths and line numbers. Do not invent file names, line numbers, or behavior that is not stated in the context.
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


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None  # uses settings.query_final_k if unset
    score_threshold: float | None = None  # uses settings if unset
    source_type: str = "all"
    tags: list[str] | None = None  # filter by tags e.g. ["file_io", "para:MAIN"]
    use_reranker: bool = False


class RetrievedChunk(BaseModel):
    id: str
    score: float  # RRF fusion score (for ordering)
    vector_score: float | None = None  # cosine similarity from vector search
    file_path: str
    start_line: int
    end_line: int
    division: str | None
    section_name: str | None
    paragraph_name: str | None
    code_snippet: str
    language: str = "COBOL"
    source_type: str = "code"


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None
    source_type: str = "all"
    tags: list[str] | None = None
    use_reranker: bool = False
    chunks: list[RetrievedChunk] | None = None  # when set, skip vector search and use these chunks only


class ChatResponse(BaseModel):
    query: str
    answer: str
    results: list[RetrievedChunk]


class AdminLoginRequest(BaseModel):
    token: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AdminReingestRequest(BaseModel):
    code_root: str | None = None
    code_extensions: str | None = None
    batch_size: int | None = None
    max_files: int | None = None


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


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> str:
    """
    Minimal admin UI with:
    - login form (ADMIN_TOKEN)
    - buttons for reset DB and reingest
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>LegacyLens Admin</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      margin: 0;
      padding: 2rem;
    }
    .card {
      background: #020617;
      border-radius: 0.75rem;
      padding: 1.5rem 2rem 2rem;
      box-shadow: 0 20px 30px rgba(15,23,42,0.7);
      max-width: 1200px;
      width: 100%;
      border: 1px solid #1f2937;
    }
    h1 {
      font-size: 1.5rem;
      margin-bottom: 0.25rem;
    }
    .subtitle {
      font-size: 0.875rem;
      color: #9ca3af;
      margin-bottom: 1.5rem;
    }
    label {
      display: block;
      font-size: 0.875rem;
      margin-bottom: 0.25rem;
    }
    input[type="password"],
    input[type="text"] {
      width: 100%;
      padding: 0.5rem 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
      font-size: 0.9rem;
      box-sizing: border-box;
    }
    input[type="password"]:focus,
    input[type="text"]:focus {
      outline: none;
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px #38bdf8;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 0.45rem 1rem;
      font-size: 0.9rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    button.primary {
      background: linear-gradient(135deg, #38bdf8, #6366f1);
      color: white;
    }
    button.secondary {
      background: #111827;
      color: #e5e7eb;
      border: 1px solid #374151;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .section {
      margin-top: 1.5rem;
      padding-top: 1.25rem;
      border-top: 1px solid #111827;
    }
    .row {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    .status {
      margin-top: 0.75rem;
      font-size: 0.8rem;
      color: #9ca3af;
      min-height: 1.2em;
      white-space: pre-line;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.15rem 0.6rem;
      border-radius: 999px;
      background: #111827;
      border: 1px solid #1f2937;
      font-size: 0.75rem;
      color: #9ca3af;
    }
    .pill-dot {
      width: 0.4rem;
      height: 0.4rem;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 2px rgba(34,197,94,0.25);
    }
    .muted {
      color: #6b7280;
      font-size: 0.75rem;
    }
    .workspace {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 1rem;
    }
    .pane {
      border: 1px solid #1f2937;
      border-radius: 0.6rem;
      padding: 0.9rem;
      background: #020617;
    }
    .pane h2 {
      margin: 0 0 0.5rem 0;
      font-size: 1rem;
    }
    .listbox {
      margin-top: 0.6rem;
      font-size: 0.8rem;
      white-space: pre-line;
      max-height: 240px;
      overflow: auto;
      border-top: 1px solid #111827;
      padding-top: 0.6rem;
    }
    @media (max-width: 900px) {
      .workspace {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
      <div>
        <h1>LegacyLens Admin</h1>
        <div class="subtitle">Manage ingestion and the vector DB for your legacy codebase.</div>
      </div>
      <div class="pill">
        <span class="pill-dot"></span>
        <span id="session-pill">Signed out</span>
      </div>
    </div>

    <div id="login-section">
      <label for="admin-token">Admin token</label>
      <input id="admin-token" type="password" autocomplete="current-password" />
      <div class="row" style="margin-top:0.75rem;">
        <button id="login-btn" class="primary">Sign in</button>
        <span class="muted">Uses ADMIN_TOKEN from your .env file.</span>
      </div>
      <div id="login-status" class="status"></div>
    </div>

    <div id="admin-tools" class="section" style="display:none;">
      <div class="workspace">
        <div class="pane">
          <h2>Code + docs corpus</h2>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
            <div id="db-summary" class="muted"></div>
            <select id="browse-source" class="secondary" style="padding:0.35rem 0.7rem;border-radius:999px;">
              <option value="all">All</option>
              <option value="code">Code only</option>
              <option value="docs">Docs only</option>
            </select>
          </div>
          <div class="row">
            <button id="refresh-count-btn" class="secondary">Refresh chunk count</button>
            <button id="load-corpus-btn" class="secondary">Load sample chunks</button>
          </div>
          <div id="corpus-list" class="listbox muted"></div>
          <div class="row">
            <button id="reset-btn" class="secondary">Reset DB</button>
          </div>
          <div class="status" id="db-status"></div>
          <label for="code-root" style="margin-top:0.75rem;">Code root (optional, overrides CODE_ROOT)</label>
          <input id="code-root" type="text" placeholder="C:\\path\\to\\legacy\\code" />
          <label for="code-extensions" style="margin-top:0.75rem;">Extensions</label>
          <input id="code-extensions" type="text" placeholder="* (all files) or .cob,.cbl,.c,.h,.md" value="*" />
          <div class="row">
            <button id="reingest-btn" class="primary">Reset + reingest</button>
          </div>
          <div class="status" id="reingest-status"></div>
        </div>

        <div class="pane">
          <h2>Query workspace</h2>
          <label for="query-text">Search query</label>
          <input id="query-text" type="text" placeholder="e.g. interest calculation, file io, parser option..." />
          <div class="row">
            <select id="query-source" class="secondary" style="padding:0.35rem 0.7rem;border-radius:999px;">
              <option value="all">Search all</option>
              <option value="code">Search code</option>
              <option value="docs">Search docs</option>
            </select>
            <button id="run-query-btn" class="primary">Run query</button>
          </div>
          <div class="status" id="query-status"></div>
          <div id="query-results" class="listbox"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let accessToken = null;

    function setSessionState(signedIn) {
      const pill = document.getElementById('session-pill');
      pill.textContent = signedIn ? 'Signed in' : 'Signed out';
      document.getElementById('login-section').style.display = signedIn ? 'none' : 'block';
      document.getElementById('admin-tools').style.display = signedIn ? 'block' : 'none';
    }

    async function login() {
      const token = document.getElementById('admin-token').value.trim();
      const status = document.getElementById('login-status');
      status.textContent = '';
      if (!token) {
        status.textContent = 'Enter your admin token.';
        return;
      }
      const btn = document.getElementById('login-btn');
      btn.disabled = true;
      status.textContent = 'Signing in...';
      try {
        const res = await fetch('/admin/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ token }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          status.textContent = data.detail || 'Login failed.';
          btn.disabled = false;
          return;
        }
        const data = await res.json();
        accessToken = data.access_token;
        status.textContent = 'Signed in.';
        setSessionState(true);
        refreshCount();
        loadCorpus();
      } catch (e) {
        status.textContent = 'Network error while logging in.';
      } finally {
        btn.disabled = false;
      }
    }

    async function refreshCount() {
      const status = document.getElementById('db-status');
      const summary = document.getElementById('db-summary');
      const source = document.getElementById('browse-source').value;
      status.textContent = '';
      summary.textContent = 'Loading...';
      try {
        const res = await fetch('/chunks?limit=1&source_type=' + encodeURIComponent(source));
        if (!res.ok) {
          summary.textContent = 'Error loading chunk count.';
          return;
        }
        const data = await res.json();
        summary.textContent = data.total_chunks + ' total chunks (' + source + ')';
      } catch {
        summary.textContent = 'Error loading chunk count.';
      }
    }

    async function loadCorpus() {
      const source = document.getElementById('browse-source').value;
      const list = document.getElementById('corpus-list');
      list.textContent = 'Loading chunks...';
      try {
        const res = await fetch('/chunks?limit=20&source_type=' + encodeURIComponent(source));
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          list.textContent = data.detail || 'Failed to load corpus chunks.';
          return;
        }
        const chunks = data.chunks || [];
        if (!chunks.length) {
          list.textContent = 'No chunks for this source filter.';
          return;
        }
        const lines = chunks.map((c, idx) => {
          const path = c.file_path || '?';
          const span = `L${c.start_line ?? '?'}-${c.end_line ?? '?'}`;
          const kind = c.source_type || 'code';
          const lang = c.language || '?';
          return `${idx + 1}. [${kind}/${lang}] ${path} ${span}`;
        });
        list.textContent = lines.join('\\n');
      } catch {
        list.textContent = 'Network error while loading corpus chunks.';
      }
    }

    async function resetDb() {
      const status = document.getElementById('db-status');
      status.textContent = 'Resetting collection...';
      const btn = document.getElementById('reset-btn');
      btn.disabled = true;
      try {
        const res = await fetch('/admin/reset-db', {
          method: 'POST',
          headers: {
            'Authorization': 'Bearer ' + accessToken,
          },
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const detail = data.detail || JSON.stringify(data) || 'Reset failed.';
          status.textContent = 'Reset failed: ' + detail;
        } else {
          status.textContent = 'Reset complete. total_chunks=' + (data.total_chunks ?? 0);
          refreshCount();
        }
      } catch {
        status.textContent = 'Network error while resetting DB.';
      } finally {
        btn.disabled = false;
      }
    }

    async function reingest() {
      const codeRoot = document.getElementById('code-root').value.trim();
      const codeExtensions = document.getElementById('code-extensions').value.trim();
      const status = document.getElementById('reingest-status');
      status.textContent = 'Reingesting... this may take a while.';
      const btn = document.getElementById('reingest-btn');
      btn.disabled = true;
      try {
        const body = {};
        if (codeRoot) {
          body.code_root = codeRoot;
        }
        if (codeExtensions) {
          body.code_extensions = codeExtensions;
        }
        const res = await fetch('/admin/reingest', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + accessToken,
          },
          body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const detail = data.detail || JSON.stringify(data) || 'Reingest failed.';
          status.textContent = 'Reingest failed: ' + detail;
        } else {
          status.textContent =
            'Reingest complete. files=' + (data.files_ingested ?? 0) +
            ', chunks=' + (data.chunks_upserted ?? 0) +
            ', total_chunks=' + (data.total_chunks ?? 0) +
            ', ext=' + (data.code_extensions ?? '?');
          refreshCount();
        }
      } catch {
        status.textContent = 'Network error while reingesting.';
      } finally {
        btn.disabled = false;
      }
    }

    async function runQuery() {
      const q = document.getElementById('query-text').value.trim();
      const source = document.getElementById('query-source').value;
      const status = document.getElementById('query-status');
      const resultsEl = document.getElementById('query-results');
      status.textContent = '';
      resultsEl.textContent = '';
      if (!q) {
        status.textContent = 'Enter a query to search.';
        return;
      }
      const btn = document.getElementById('run-query-btn');
      btn.disabled = true;
      status.textContent = 'Searching...';
      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            top_k: 10,
            score_threshold: 0.0,
            source_type: source,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          status.textContent = data.detail || 'Search failed.';
          return;
        }
        const results = data.results || [];
        if (!results.length) {
          status.textContent = 'No results.';
          return;
        }
        status.textContent = `Found ${results.length} chunks.`;
        const lines = results.map((r, idx) => {
          const path = r.file_path || '?';
          const span = `L${r.start_line ?? '?'}-${r.end_line ?? '?'}`;
          const kind = r.source_type || 'code';
          const para = r.paragraph_name || '(no paragraph)';
          const snippet = (r.code_snippet || '').replace(/\\n/g, ' ').slice(0, 140);
          return `${idx + 1}. [${kind}] ${path} ${span} | ${para}\\n   ${snippet}...`;
        });
        resultsEl.textContent = lines.join('\\n\\n');
      } catch {
        status.textContent = 'Network error while searching.';
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById('login-btn').addEventListener('click', login);
    document.getElementById('refresh-count-btn').addEventListener('click', refreshCount);
    document.getElementById('load-corpus-btn').addEventListener('click', loadCorpus);
    document.getElementById('browse-source').addEventListener('change', () => {
      refreshCount();
      loadCorpus();
    });
    document.getElementById('reset-btn').addEventListener('click', resetDb);
    document.getElementById('reingest-btn').addEventListener('click', reingest);
    document.getElementById('run-query-btn').addEventListener('click', runQuery);
  </script>
</body>
</html>
    """


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
    """
    project = settings.google_cloud_project
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    return {
        "status": "ok",
        "embeddings": "vertex" if project else "pseudo",
        "google_cloud_project": project or None,
        "credentials_set": bool(creds_env),
        "llm_enabled": bool(settings.llm_enabled and settings.llm_model and project),
        "llm_model": settings.llm_model if settings.llm_enabled else None,
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


# Known extensionless doc files - when query mentions these, fetch by file for reliable retrieval
_DOC_FILENAMES = frozenset({"thanks", "readme", "authors", "news", "copying", "todo", "changelog"})


def _extract_doc_filename(query: str) -> str | None:
    """If query mentions a known doc file (e.g. 'THANKS file'), return its name."""
    import re
    q = query.lower()
    for name in _DOC_FILENAMES:
        if re.search(rf"\b{re.escape(name)}\b", q):
            return name.upper()
    return None


@app.post("/query/chat", response_model=ChatResponse)
def query_chat(req: ChatRequest) -> ChatResponse:
    """
    Run hybrid search, build context, and generate answer via Vertex AI (if LLM configured).
    Falls back to returning just the prompt when LLM is not configured.
    When query mentions a specific file (THANKS, README, etc.), fetches that file's chunks directly.
    """
    logger.info("Chat: %s", req.query[:60] + "..." if len(req.query) > 60 else req.query)
    if req.chunks:
        results = req.chunks
    else:
        client = get_vector_store()
        top_k = req.top_k if req.top_k is not None else settings.query_chat_final_k
        results = []

        # When user asks about a specific doc file, fetch its chunks directly (avoids retrieval misses)
        doc_file = _extract_doc_filename(req.query)
        if doc_file:
            raw_chunks = get_chunks_for_file(client, doc_file)
            if not raw_chunks and doc_file != doc_file.lower():
                raw_chunks = get_chunks_for_file(client, doc_file.lower())
            if not raw_chunks:
                chunks_list, _ = list_chunks(client, limit=200, file_path_contains=doc_file)
                raw_chunks = [c for c in chunks_list if doc_file.lower() in (c.get("file_path") or "").lower()]
            if raw_chunks:
                raw_chunks.sort(key=lambda c: (c.get("start_line") or 0))
                results = [
                    RetrievedChunk(
                        id=c.get("id", ""),
                        score=0.0,
                        vector_score=None,
                        file_path=c.get("file_path", ""),
                        start_line=c.get("start_line", 0),
                        end_line=c.get("end_line", 0),
                        division=c.get("division"),
                        section_name=c.get("section_name"),
                        paragraph_name=c.get("paragraph_name"),
                        code_snippet=c.get("code_snippet", ""),
                        language=c.get("language", "COBOL"),
                        source_type=c.get("source_type", "code"),
                    )
                    for c in raw_chunks[:top_k]
                ]

        if not results:
            embedder = get_embedder()
            try:
                vectors = embedder.embed_texts([req.query])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
            if not vectors or not vectors[0]:
                raise HTTPException(status_code=500, detail="No embedding returned")
            hits = hybrid_search(
                client,
                vectors[0],
                req.query,
                top_k=settings.query_chat_top_k,
                final_k=top_k,
                score_threshold=settings.query_score_threshold,
                source_type=req.source_type if req.source_type != "all" else None,
                tags_filter=req.tags,
                use_reranker=req.use_reranker,
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
    max_lines = settings.chat_snippet_max_lines

    def _format_chunk(i: int, r: RetrievedChunk) -> str:
        meta_parts = [f"[{i}] {r.file_path} L{r.start_line}-{r.end_line}"]
        if r.paragraph_name:
            meta_parts.append(f"para:{r.paragraph_name}")
        if r.section_name and r.section_name != r.paragraph_name:
            meta_parts.append(f"section:{r.section_name}")
        if r.vector_score is not None:
            meta_parts.append(f"relevance:{r.vector_score:.2f}")
        snippet = r.code_snippet or ""
        if max_lines > 0:
            lines = snippet.splitlines()
            if len(lines) > max_lines:
                snippet = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return " | ".join(meta_parts) + "\n" + snippet

    context = "\n\n---\n\n".join(
        _format_chunk(i, r) for i, r in enumerate(results, 1)
    )
    full_prompt = (
        f"{CHAT_SYSTEM_PROMPT}\n"
        "---\n\n"
        "**Context** (numbered chunks; each has file path, line range, then code):\n\n"
        f"{context or '(No chunks retrieved.)'}\n\n"
        "---\n\n"
        f"**Question:** {req.query}\n\n"
        "**Answer:**"
    )

    if settings.llm_enabled and settings.llm_model and settings.google_cloud_project:
        try:
            from google import genai
            from google.genai.types import HttpOptions

            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                http_options=HttpOptions(api_version="v1"),
            )
            from google.genai.types import GenerateContentConfig

            resp = client.models.generate_content(
                model=settings.llm_model,
                contents=full_prompt,
                config=GenerateContentConfig(
                    max_output_tokens=settings.llm_max_output_tokens,
                    temperature=0.2,
                ),
            )
            answer = resp.text if resp and resp.text else "(No response from LLM)"
        except Exception as e:
            answer = f"(LLM error: {e})"
    else:
        answer = "(Enable LLM: set LLM_MODEL and LLM_ENABLED=true with GOOGLE_CLOUD_PROJECT)"

    return ChatResponse(query=req.query, answer=answer, results=results)


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
        fp = (base / path) if not Path(path).is_absolute() else Path(path)
        fp = fp.resolve()
        try:
            fp.relative_to(base)
        except ValueError:
            pass
        else:
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
    logger.info("LegacyLens API ready – POST /query, /admin/reingest, etc.")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Embed the query and run hybrid search (vector + BM25 + RRF); return top-k chunks."""
    logger.info("Query: %s", req.query[:60] + "..." if len(req.query) > 60 else req.query)
    embedder = get_embedder()
    try:
        vectors = embedder.embed_texts([req.query])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    if not vectors or not vectors[0]:
        raise HTTPException(status_code=500, detail="No embedding returned")
    client = get_vector_store()
    top_k = req.top_k if req.top_k is not None else settings.query_final_k
    score_threshold = (
        req.score_threshold if req.score_threshold is not None else settings.query_score_threshold
    )
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
