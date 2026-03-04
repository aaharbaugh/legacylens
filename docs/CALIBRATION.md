# Calibrating LegacyLens and making it better

This doc covers: **how to search/inspect the DB**, then **what to tune** to improve retrieval quality.

---

## 1. How to search and inspect the DB

### Semantic search (what you already have)

- **API:** `POST /query` with `{"query": "your question", "top_k": 15}`  
- **CLI:** `python scripts/query_cli.py "your question"`  
- **Browser:** http://localhost:8000/docs → POST /query

This embeds your question and returns the most similar stored chunks (file, lines, snippet, score).

### Browse what’s stored (list chunks)

**From the API:**

- **GET /chunks** – list stored chunks (paginated).
  - `http://localhost:8000/chunks` – first 50 chunks  
  - `http://localhost:8000/chunks?limit=20`  
  - `http://localhost:8000/chunks?file_path=sample` – only chunks whose file path contains `"sample"`

**From the command line:**

```powershell
python scripts/inspect_db.py
python scripts/inspect_db.py --limit 10
python scripts/inspect_db.py --file sample
python scripts/inspect_db.py --json
```

Use this to check that ingestion did what you expect (paragraph names, line ranges, which files are indexed).

---

## 2. What to calibrate (knobs that affect quality)

| Knob | Where | What it does | Suggestion |
|------|--------|--------------|------------|
| **top_k** | Query request | How many chunks to return for `/query` (e.g. 5, 15, 20). | Start at 15; lower if you want fewer, noisier results. |
| **query_chat_final_k** | Env / `config.py` | Chunks sent to LLM for chat (default 12). | Fewer = less noise, better focus; increase if answers lack context. |
| **score_threshold** | Query request | Ignore chunks with score below this (e.g. 0.5, 0.7). | Start at 0.0 to see all scores; raise to 0.5–0.7 to hide weak hits. |
| **min_vector_score** | Env | Drop chunks with vector cosine below this (default 0.15). | Raise to 0.2+ to filter junk; lower to 0.08 if missing good hits. |
| **use_reranker** | Env | Enable cross-encoder reranking for chat (adds ~5–15s). | Set `USE_RERANKER=false` for ~10s faster; `true` for better quality. |
| **reranker_model** | Env | Reranker model. | `cross-encoder/ms-marco-MiniLM-L6-v2` (fast default); `BAAI/bge-reranker-base` for quality. |
| **Embeddings** | Env / ingestion | Pseudo (no GCP) vs real (Vertex `text-embedding-004`). | Use real embeddings for production; set `GOOGLE_CLOUD_PROJECT` and re-run ingestion. |
| **Chunking** | `backend/config.py` + `ingestion/chunker.py` | Paragraph vs fixed-size; fallback size/overlap. | If everything is one big chunk per file, tighten paragraph detection or lower `fallback_chunk_lines`. |
| **Excluded files** | `ingestion/discovery.py` | Build artifacts (.vcxproj, build_windows/, etc.) are excluded. | Re-ingest after changing exclusions. |
| **Re-ingest** | Pipeline | Re-run after changing chunking, embeddings, or BM25 tokenizer. | After any of these changes, run ingestion again. |

### Tuning retrieval from the API

- **Fewer, stricter results:** `POST /query` with `{"query": "...", "top_k": 5, "score_threshold": 0.6}`.  
- **More, exploratory:** `top_k`: 20–30, `score_threshold`: 0.0.  
- Inspect scores in the response; if good chunks are just below a threshold, lower it slightly or increase `top_k`.

### Tuning chunking

- **Config** (`backend/config.py`):  
  - `fallback_chunk_lines` (default 120), `fallback_overlap_lines` (default 25).  
  - Smaller chunks = more precise but more fragments; larger = more context, less precision.  
- **Logic** (`backend/ingestion/chunker.py`):  
  - COBOL paragraph detection (Area A, period). If your code doesn’t follow that, you’ll get fallback chunks; you can relax or tighten the regex for your codebase.  
- After changes: **re-run ingestion** so the vector DB reflects the new chunks.

### Chat prompt (answer quality)

The LLM system prompt is defined in **`backend/api/main.py`** as `CHAT_SYSTEM_PROMPT`. It controls:

- **Grounding** – Answer only from context; exact phrase when context is insufficient.
- **Citations** – Use [1], [2], [3] and file path + line range.
- **Enumerations** – For WHO/WHAT/LIST questions, scan all chunks and list every relevant item.
- **Code** – Use \`\`\`cobol or \`\`\`c for examples.

Edit `CHAT_SYSTEM_PROMPT` to tighten or relax these rules, add domain-specific instructions (e.g. COBOL level numbers, C function names), or change the “not enough context” wording. No reingest needed; restart the API to pick up changes.

### Reranker for better chat answers

The chat endpoint uses a cross-encoder reranker when enabled. This reorders retrieved chunks by relevance before sending to the LLM.

1. Install: `pip install sentence-transformers` (recommended on Windows; avoids zlib-state build issues). Alternatively: `pip install FlagEmbedding`
2. Set in `.env`: `USE_RERANKER=true`
3. Restart the API. Chat will automatically use reranking.

---

## 3. Simple evaluation (optional)

To measure “does the right chunk show up in top‑k?”:

1. **Add a small ground-truth file** (e.g. `docs/eval_queries.json`):

```json
[
  { "query": "Where is CALCULATE-INTEREST?", "expected_paragraph": "CALCULATE-INTEREST" },
  { "query": "file I/O or customer file", "expected_paragraph": "READ-NEXT" }
]
```

2. **Run retrieval** for each query (same as `POST /query` or `query_cli.py`).  
3. **Check** whether any returned chunk’s `paragraph_name` (or file/line) matches `expected_paragraph`.  
4. **Track** e.g. “3 of 5 queries had a correct chunk in top‑5” and tune `top_k` / `score_threshold` / chunking until that number is good (e.g. >70% for top‑5).

You can do this manually at first (run a few queries, look at results), then automate with a small script that reads the JSON, calls the query endpoint, and prints hit/miss.

---

## 4. Quick checklist to make it better

1. **Inspect the DB** – `GET /chunks` or `python scripts/inspect_db.py` – confirm chunks look right (files, paragraphs, lines).  
2. **Use real embeddings** – set `GOOGLE_CLOUD_PROJECT`, re-run ingestion, then query again; compare score distribution.  
3. **Enable reranker for chat** – `pip install sentence-transformers`, set `USE_RERANKER=true` in `.env`.  
4. **Adjust top_k, query_chat_final_k, and score_threshold** – try 5, 10, 15 and 0.0, 0.5, 0.7; see what feels right.  
5. **Improve chunking** – if chunks are too big or too small, change fallback size/overlap or paragraph rules and re-ingest.  
6. **Add a few eval queries** – write 5–10 query + expected chunk, run them, and tune until most hit in top‑5.
