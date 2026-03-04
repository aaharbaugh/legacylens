LegacyLens: RAG for Legacy Codebases
====================================

## Overview

LegacyLens is a RAG-powered navigator for large legacy codebases (starting with GnuCOBOL). It ingests source files, chunks them with syntax-aware rules, stores embeddings in a vector database, and exposes a query interface that lets engineers ask natural-language questions and get back grounded answers with file and line-number citations.

This project follows the Gauntlet Week 3 "LegacyLens" spec and your Pre-Search document, with a focus on:

- **Reliable retrieval** over legacy code (COBOL paragraph-aware chunking)
- **Hybrid search** (semantic + keyword) using Qdrant
- **Production-minded architecture** (Cloud Run backend, managed vector DB, secure secrets)

## High-Level Architecture

- **Ingestion pipeline (Python)**
  - Recursively scan the GnuCOBOL repo for `.cob` / `.cbl` files
  - Normalize and chunk code around COBOL-specific boundaries (e.g., PROCEDURE DIVISION, Area A paragraph endings)
  - Generate embeddings with `text-embedding-004` (Google)
  - Upsert dense (and optionally sparse) vectors + rich metadata into Qdrant

- **Retrieval & RAG backend (FastAPI)**
  - Accept natural-language queries
  - Perform multi-query expansion (GPT-4o-mini) and hybrid search in Qdrant
  - Merge, deduplicate, and re-rank chunks; enforce a relevance threshold
  - Stream GPT-4o answers via SSE, always grounded in retrieved code snippets

- **Frontend (later)**
  - Minimal CLI or simple web UI first
  - Later, a React / Next.js app with:
    - Query bar
    - Retrieved snippet list with syntax highlighting
    - File path + line-number citations
    - Answer panel with streaming responses

## Directory Layout

Planned high-level structure for this repo:

- `backend/`
  - `ingestion/` – scripts and modules for scanning, chunking, embedding, and upserting into Qdrant
  - `api/` – FastAPI app for query handling, retrieval, and answer generation
- `frontend/`
  - Placeholder for CLI or React/Next.js UI (to be bootstrapped with your preferred tool)
- `docs/`
  - Pre-search output (copied from your Gauntlet Pre-Search PDF)
  - RAG architecture doc
  - Cost analysis and any interview prep notes

You can refine this layout as the implementation solidifies (e.g., adding `tests/`, `scripts/`, and deployment artifacts like `Dockerfile`).

## MVP Scope (24-Hour Target)

Aligned with the Gauntlet MVP checklist:

- **Ingest** at least one legacy codebase (GnuCOBOL)
- **Chunk** code with syntax-aware splitting (paragraph-level in COBOL, with fallback fixed-size chunks)
- **Embed** all chunks with `text-embedding-004`
- **Store** vectors + metadata in Qdrant Cloud
- **Implement** semantic + keyword (hybrid) search
- **Expose** a basic query interface (CLI or simple web page)
- **Return** relevant code snippets with file/line references
- **Generate** basic answers via GPT-4o using only retrieved context

## Next Steps

Short-term implementation steps:

1. Initialize a Python backend in `backend/` with a `requirements.txt` and a simple FastAPI app.
2. Implement an ingestion script in `backend/ingestion/` that:
   - Walks the GnuCOBOL repo
   - Applies COBOL-aware chunking
   - Calls `text-embedding-004` in batches and upserts into Qdrant
3. Implement a query endpoint in `backend/api/` that:
   - Expands queries with GPT-4o-mini
   - Runs hybrid search in Qdrant
   - Assembles context and streams GPT-4o answers
4. Add a minimal CLI or web UI in `frontend/` that hits the query endpoint.
5. Copy your full Pre-Search output into `docs/` and start an `ARCHITECTURE.md` that you will keep updated as you iterate.

As you build, keep the Gauntlet Pre-Search checklist, RAG architecture doc, and cost analysis requirements in `docs/` so Cursor can use them as context while you work.

---

## Quick start: functional vector DB

**→ For a direct walkthrough with exact commands, see [docs/SETUP.md](docs/SETUP.md).**

Short version (from repo root):

1. **Install:** `pip install -e .`
2. **Ingest:** `python -m backend.ingestion.pipeline run --code-root sample_cobol`
3. **Start API:** `python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000`
4. **Query:** `python scripts/query_cli.py "Where is CALCULATE-INTEREST?"` (or use `http://localhost:8000/app` or `http://localhost:8000/docs`)

Uses pseudo-embeddings by default (no GCP). For real embeddings, set `GOOGLE_CLOUD_PROJECT` and re-run ingestion. Qdrant data lives in `.qdrant_data/`; set `QDRANT_URL` and `QDRANT_API_KEY` for Qdrant Cloud.

**Search the DB:** `GET http://localhost:8000/chunks` to list stored chunks; `POST /query` for semantic search. Both endpoints support source partitioning (`source_type=all|code|docs`) after re-ingestion with current metadata. **Calibration:** see [docs/CALIBRATION.md](docs/CALIBRATION.md) for tuning retrieval and improving quality.

