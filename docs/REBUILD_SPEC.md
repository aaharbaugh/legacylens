# LegacyLens Rebuild Specification

Single handoff document for rebuilding this system from scratch with a new architecture. Captures product intent, current behavior, what was learned the hard way, and recommendations.

---

## 1. Product Summary

**LegacyLens** is a RAG (retrieval-augmented generation) app over legacy codebases (target: GnuCOBOL and similar). Users ingest source files, then ask questions in natural language and get answers grounded in code with file + line citations.

**Core value:** Hybrid search (vector + lexical) over chunked source, optional LLM to turn retrieved chunks into a direct answer with citations.

---

## 2. Current Architecture (As-Built)

### 2.1 Stack

- **Backend:** Python 3.10+, FastAPI, uvicorn
- **Embeddings:** OpenAI `text-embedding-3-small` (768 dims); fallback: deterministic pseudo-embeddings when no API key
- **LLM:** OpenAI chat (e.g. `gpt-4o-mini`) for chat answers, ask intro, ask extractor, and summarization
- **Vector DB:** Qdrant (local `.qdrant_data` or Qdrant Cloud)
- **Lexical search:** BM25 (`rank-bm25`), persisted as `bm25_index.pkl`
- **Config:** pydantic-settings from `.env`; single `Settings` in `backend/config.py`

### 2.2 Data Flow

**Ingestion (batch, CLI):**

1. **Discover** files under `CODE_ROOT` by extension (e.g. `.cob,.cbl,.c,.h,.md,.at`).
2. **Chunk** each file: COBOL paragraph/section boundaries with fixed-size fallback; emit `CodeChunk` with `file_path`, `start_line`, `end_line`, `code_snippet`, metadata.
3. **Enrich** each batch of chunks: optional LLM **summaries** per chunk (`summary_text`), then set `chunk_role` = `child_summary`, `parent_code_snippet` = copy of `code_snippet`.
4. **Embed** chunk text (summary or code, with optional `metadata_prefix`) via OpenAI; support batching and token-limit sub-batches.
5. **Upsert** to Qdrant: one point per chunk (vector + payload); payload includes all metadata, `code_snippet` truncated (e.g. 65535 chars).
6. **BM25:** Build index over same corpus (e.g. `summary_text` or `code_snippet` + prefix), save pickle.

**Query / Chat / Ask (API):**

- **Query:** Embed query → vector search in Qdrant (with optional filters) + BM25 search → RRF fusion → optional rerank → return top-k chunks.
- **Chat:** Same retrieval; build context from chunks; send to LLM with system prompt; stream or return answer with citations.
- **Ask:** Resolve chunks (prefetch cache or embed + hybrid search); LLM “intro” (short intent paraphrase) and “extractor” (answer + code snippet from context); return or stream.

### 2.3 Key Payload / Schema

**Chunk payload in Qdrant (and BM25 doc):**

- **Identity / location:** `file_path`, `start_line`, `end_line`, `file_dir`, `file_name`, `file_ext`, `folder`, `phase`
- **Content:** `code_snippet`, `summary_text`, `metadata_prefix`
- **Roles / hierarchy:** `chunk_role` (`child_summary` | `parent_code`), `chunk_type`, `parent_id`, `retrieval_parent_id`, `parent_start_line`, `parent_end_line`, `parent_code_snippet`
- **Filtering / UX:** `source_type`, `tags`, `include_headers`
- **Embedding input:** For “summary-first” indexing, embed `summary_text` (or heuristic) + optional prefix; keep raw `code_snippet` in payload for display.

**Qdrant:**

- One collection; vector dim 768; cosine distance.
- **Filtering:** Queries filter by `chunk_role` (e.g. `child_summary`) and optionally `source_type`. **Qdrant Cloud requires payload indexes for any filtered key** (keyword type).

### 2.4 API Surface

- `GET /health`, `GET /status` (config/readiness)
- `GET /health/db` (Qdrant + BM25 info)
- `POST /query` – hybrid search, top-k chunks
- `POST /query/chat` – RAG chat (search + LLM)
- `POST /api/ask` – ask (intro + extractor, non-stream)
- `POST /api/ask/stream` – ask with SSE stream (intro chunks, then result)
- `GET /chunks` – list chunks (paginated, filter by file)
- Admin (token-protected): reingest, reset collection, request logs
- Frontend: static HTML/JS; dev vs production (with/without admin UI)

### 2.5 Config (Env) Highlights

- **Paths:** `CODE_ROOT`, `qdrant_local_path`, `qdrant_url`, `qdrant_api_key`, `qdrant_collection`, `embeddings_cache_path`, `bm25_index_path`
- **Embedding:** `openai_api_key`, `openai_base_url`, `embed_model`, `embed_dimensions`, `embed_batch_size`, `embed_max_tokens_per_request`, `embed_max_workers`
- **Chunking:** `code_extensions`, `max_files`, `max_paragraph_chunk_lines`, `fallback_chunk_lines`, `fallback_overlap_lines`
- **Retrieval:** `query_top_k`, `query_final_k`, `query_chat_top_k`, `query_chat_final_k`, `min_vector_score`, `use_hybrid_search`, `use_reranker`, `rrf_k`, `bm25_min_score`
- **Summaries:** `summary_generation_enabled`, `summary_model`, `summary_max_concurrency`, `summary_input_max_chars`, `summary_timeout_sec`
- **LLM:** `llm_model`, `llm_enabled`, `llm_max_output_tokens`, `llm_intro_model`, `llm_extractor_model`, ask/extractor token limits
- **Ask / prefetch:** `prefetch_enabled`, `prefetch_candidates_k`, `prefetch_cache_top_k`, `prefetch_cache_ttl_sec`, `prefetch_cooldown_sec`, `ask_use_reranker`, reranker timeouts

---

## 3. What Broke or Hurt (Lessons Learned)

### 3.1 Embedding Cache and “Embed” Latency

- **Issue:** A single “embed” phase in request logs was ~13+ seconds.
- **Cause:** The same `Embedder` instance is used for both **ingestion** (writes a large `embeddings.json` cache) and **query-time** (one query string). On first API request, `get_embedder()` builds `Embedder()` and calls `_load_cache()`, which reads the **entire** `embeddings.json`. After a full ingestion that file can be huge (e.g. 10M+ lines of JSON). Loading and parsing it dominates the “embed” time; the actual OpenAI call for one query is a small fraction.
- **Lesson:** Do **not** share one big ingestion cache with the query path. Either: (a) separate caches (ingestion cache vs query-time cache or no cache for queries), (b) lazy-load or limit what gets loaded at API startup, or (c) don’t load the ingestion cache in the API process at all.

### 3.2 Qdrant Cloud: Collection and Index Semantics

- **409 “Collection already exists”:** If the app calls “create collection” when the collection already exists (e.g. created in Cloud UI or by another process), Qdrant returns 409. The “ensure collection” logic must treat 409 as success (collection exists) and not fail the request.
- **400 “Index required but not found for chunk_role”:** Qdrant Cloud requires **payload indexes** for any payload key used in a filter. The app filters by `chunk_role` (and optionally `source_type`). Without keyword indexes on these fields, search returns 400.
- **Lesson:** On “ensure collection” (or first use of the collection), create payload indexes for every filter key (e.g. `chunk_role`, `source_type`) with the appropriate type (e.g. keyword). Handle 409 on index creation (index already exists).

### 3.3 Redundant and Confusing Payload

- **Issue:** `parent_code_snippet` was set to the same value as `code_snippet` for child-summary chunks (copy in pipeline). So two fields duplicated the same content.
- **Lesson:** In a rebuild, either store a single source of truth for “code to show” and derive the rest, or clearly separate “snippet for display” vs “snippet for parent reference” so the model isn’t redundant.

### 3.4 Ingestion Visibility

- **Issue:** No progress logs during the long “summary” and “embed” phase per batch. Users see a long silence and don’t know if it’s working.
- **Lesson:** Emit progress (e.g. “Summarizing batch 1/10”, “Embedding batch 2/10”, “Upserted N chunks”) so ingestion is observable.

### 3.5 Provider Coupling

- **Issue:** Original design was tightly coupled to GCP/Vertex (embeddings + LLM). Migrating to OpenAI required touching many call sites and config.
- **Lesson:** Abstract the embedding and LLM behind a thin interface (e.g. “embed(texts)”, “complete(messages, options)”) and one place that chooses the provider (OpenAI, etc.). Config and status should refer to “embedding/LLM provider” and keys (e.g. API key), not “Vertex” or “GCP”.

---

## 4. Recommendations for a Rebuild

### 4.1 Separation of Ingestion vs Query

- **Ingestion:** CLI or job that reads codebase → chunk → (optional) summarize → embed in batches → write to vector store + BM25. Use a **dedicated embedding cache only for ingestion** (or no cache; or a small LRU). Persist vectors and BM25 index; do not require the API to be running.
- **API / Query:** Serve search, chat, ask. **Do not load the ingestion embedding cache.** For query, either: no cache, or a small in-memory (or separate file) cache keyed by query hash, with TTL and size limit. First request should be “one embed call” latency, not “load 100MB JSON”.

### 4.2 Vector Store and Indexes

- **Collection lifecycle:** “Ensure collection” = get collection if exists, else create. On create **or** when attaching to an existing collection, ensure **payload indexes exist** for every filter field (e.g. `chunk_role`, `source_type` as keyword). Catch 409 / “already exists” for both collection and index creation.
- **Filter keys:** Document which payload keys are used in filters and that they must be indexed (and with which type) for the target Qdrant (Cloud vs local).

### 4.3 Observability

- **Ingestion:** Log at least: start; per-batch summary/embed/upsert progress; batch size and cumulative chunks; finish and total time.
- **Request logs:** Keep a clear breakdown (embed_ms, search_ms, rerank_ms, llm_ms, etc.). If “embed_ms” is measured, define it as “time for the embedding API call(s) for this request” only, not “time to init embedder + load cache + call API”.

### 4.4 Config and Provider Abstraction

- **Single config module** (env-driven) for paths, model names, limits, feature flags.
- **Thin provider layer:** e.g. `EmbeddingProvider.embed(texts)`, `LLMProvider.complete(messages, options)`, `LLMProvider.stream(...)`. One implementation per provider (OpenAI, etc.); switch via config. No provider-specific types in business logic (ask/chat/ingestion).

### 4.5 Chunk and Payload Model

- **Explicit chunk model:** One canonical shape (file_path, start_line, end_line, content for display, content for embedding, role, parent refs if any). Avoid duplicating the same string in multiple payload fields; derive display vs embed text from a single source where possible.
- **BM25 and Qdrant:** Use the same chunk identity and content view so hybrid fusion stays consistent.

### 4.6 Ask / Chat Behavior

- **Ask:** Resolve chunks (search) → optional short “intro” (intent) → “extractor” (answer + snippet from context). Streaming: emit intro tokens or chunks, then extractor result. Keep token limits and context windows in config.
- **Chat:** Search → build context string → LLM with system + user message; cite by chunk index and file/line. Same retrieval knobs (top_k, min_vector_score, reranker) as search.

---

## 5. Reference: Important File Roles (Current Repo)

- **backend/config.py** – All settings and env mapping.
- **backend/ingestion/pipeline.py** – Discovery → chunk → summarize → embed → upsert; batch loop; BM25 build.
- **backend/ingestion/embedder.py** – Embedding (OpenAI or pseudo); cache load/save; batch and token splitting.
- **backend/ingestion/summarizer.py** – Per-chunk summary (LLM or heuristic).
- **backend/ingestion/vector_store.py** – Qdrant client; ensure_collection + payload indexes; upsert; search; hybrid_search (vector + BM25 + RRF + optional rerank).
- **backend/ingestion/chunker.py** – COBOL-aware chunking (paragraph, fallback).
- **backend/ingestion/bm25_index.py** – BM25 index build and search.
- **backend/api/main.py** – FastAPI app; routes; prefetch cache; _resolve_ask_chunks; get_embedder / get_vector_store singletons.
- **backend/api/ask_service.py** – run_api_ask, run_api_ask_stream (intro + extractor).
- **backend/api/chat_service.py** – run_query_chat (search + LLM).
- **backend/llm_client.py** – OpenAI helpers (openai_generate, openai_chat, openai_generate_stream, openai_embed).

---

## 6. Quick Checklist for New Build

- [ ] Ingestion and API use **separate** embedding cache (or no cache for API).
- [ ] “Ensure collection” creates payload indexes for all filter keys; handle 409.
- [ ] Ingestion logs progress per batch (summary, embed, upsert).
- [ ] Request “embed” timing = only embedding API time, not cache load.
- [ ] Single provider abstraction for embeddings and LLM; config-driven.
- [ ] Chunk/payload model has no redundant copies of the same content.
- [ ] Docs list required payload indexes for Qdrant (and 409 handling).

This document is the single artifact to take into a new project and rebuild the architecture with the above fixes and boundaries in mind.
