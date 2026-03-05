# LegacyLens

RAG over legacy codebases (GnuCOBOL and similar). Ingest source files, chunk with COBOL-aware rules, embed with OpenAI, store in Qdrant, and query via natural language with cited answers.

## Features

- **Hybrid search** – Vector (OpenAI `text-embedding-3-small`) + BM25 with RRF fusion; optional cross-encoder rerank
- **COBOL-aware chunking** – Paragraph/section boundaries with fixed-size fallback
- **Chat** – Ask questions; answers are grounded in retrieved chunks and cite file + line (OpenAI)
- **Local or Cloud** – Qdrant local (`.qdrant_data`) or Qdrant Cloud; deploy to Google Cloud Run or any Docker host

## Quick start (local)

From repo root:

1. **Install**
   ```bash
   pip install -e .
   ```

2. **Configure**  
   Copy `.env.example` to `.env` (or create `.env`) and set at least:
   - `CODE_ROOT` – path to the codebase to ingest (e.g. `gnucobol-3.2_win`)
   - For **embeddings/LLM**: `OPENAI_API_KEY` (from platform.openai.com)  
   Without it, the app uses pseudo-embeddings and no LLM.

3. **Ingest**
   ```bash
   python -m backend.ingestion.pipeline run --code-root gnucobol-3.2_win
   ```
   Uses `CODE_ROOT` from `.env` if you omit `--code-root`.

4. **Run API**
   ```bash
   python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Use the app**  
   - **UI:** http://localhost:8000 (or http://localhost:8000/app)  
   - **API docs:** http://localhost:8000/docs  
   - **Health / status:** http://localhost:8000/status  

## Deploy to Cloud Run

1. Put **Qdrant Cloud** URL and API key (and optionally `CLOUD_RUN_SERVICE_ACCOUNT`) in `.env`.
2. Run the deploy script (builds image, pushes, deploys):
   ```powershell
   .\deploy-cloudrun.ps1
   ```
   The script reads `QDRANT_API_KEY`, `ADMIN_TOKEN`, and `MIN_VECTOR_SCORE` from `.env` and passes them to Cloud Run.

**Force a clean image build (if online still has old behavior):**
```powershell
$env:NO_CACHE="1"; .\deploy-cloudrun.ps1
```

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for manual steps, Docker, and troubleshooting.

## Environment (.env)

| Variable | Purpose |
|----------|---------|
| `CODE_ROOT` | Path to codebase to ingest |
| `OPENAI_API_KEY` | OpenAI API key (embeddings + LLM) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud; leave unset for local `.qdrant_data` |
| `ADMIN_TOKEN` | Token for admin endpoints (reingest, reset DB) |
| `LLM_MODEL`, `LLM_ENABLED` | e.g. `gpt-4o-mini` and `true` for chat |
| `MIN_VECTOR_SCORE` | Filter chunks by vector score (higher = stricter) |

## Calibration and tuning

- **Retrieval / chat quality** – [docs/CALIBRATION.md](docs/CALIBRATION.md): `MIN_VECTOR_SCORE`, `query_chat_final_k`, reranker, chunking, prompt.
- **Chat prompt** – Edit `CHAT_SYSTEM_PROMPT` in `backend/api/main.py`; restart to apply.

## Project layout

- `backend/api/main.py` – FastAPI app: `/query` (search), `/query/chat` (RAG chat), `/chunks`, `/status`, admin, frontend serve
- `backend/ingestion/` – Discovery, chunker, embedder (OpenAI), vector_store (Qdrant), BM25, pipeline
- `backend/config.py` – Settings from env (chunk sizes, retrieval knobs, LLM)
- `frontend/` – HTML + JS assets; `index.html` (local, with Admin), `index.production.html` (minimal for deploy)
- `docs/DEPLOY.md` – Deployment guide  
- `docs/CALIBRATION.md` – Tuning retrieval and quality  
- `deploy-cloudrun.ps1` – Build, push, and deploy to Cloud Run (reads .env)
