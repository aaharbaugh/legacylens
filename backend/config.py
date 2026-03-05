"""
Central configuration for LegacyLens backend.
Reads from environment; use .env for local dev.
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env" if Path(".env").exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Codebase to ingest
    code_root: Optional[Path] = None

    # File patterns (comma-separated). Use * for all files (recommended for GnuCOBOL: C impl + COBOL + docs)
    code_extensions: str = ".cob,.cbl,.c,.h,.md,.at"
    # Max files to ingest (None = no limit). Stops discovery early for smaller indexes.
    max_files: Optional[int] = None
    # Skip oversized files during ingestion (protects full-repo indexing runs)
    max_file_size_mb: int = 2
    # When true, skip files that do not look text-readable.
    ingest_text_only: bool = True

    # Qdrant: use local path for dev (no server), or URL for Qdrant Cloud
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_local_path: Path = Path(".qdrant_data")
    qdrant_collection: str = "legacylens-chunks"

    # Embedding (tuned for Vertex quota: 2 req/s — avoid parallel burst during ingestion)
    embed_batch_size: int = 80  # chunks per pipeline batch (higher = fewer batches)
    embed_max_tokens_per_request: int = 18_000  # Vertex allows 20k; stay under to avoid INVALID_ARGUMENT
    embed_max_workers: int = 1  # parallel sub-batches (1 = sequential to avoid 429 burst)
    embed_delay_between_requests_sec: float = 0.0  # set to 0.5 if you still hit 429 (2 req/s); 0 = no extra delay
    embed_model: str = "text-embedding-004"
    embed_dimensions: int = 768
    # Google Cloud (Vertex AI)
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = "us-central1"

    # Embedding cache (ingestion)
    embeddings_cache_path: Path = Path("embeddings.json")

    # Chunking: smaller chunks often embed and retrieve better
    max_paragraph_chunk_lines: int = 45  # cap one "paragraph" chunk
    fallback_chunk_lines: int = 45  # fixed-size fallback window
    fallback_overlap_lines: int = 0  # 0 = no overlap between adjacent fallback chunks

    # Retrieval
    query_top_k: int = 50  # candidates before rerank/fusion (main /query)
    query_chat_top_k: int = 30  # candidate pool for chat; larger = better for keyword queries (e.g. EVALUATE)
    query_final_k: int = 25  # chunks for /query endpoint
    query_chat_final_k: int = 8  # chunks sent to LLM; fewer = less context, faster
    query_chat_max_chunks_per_file: int = 2  # cap chunks per file (rank by file, then take best N per file); 0 = no cap
    chat_snippet_max_lines: int = 20  # max lines per chunk in prompt and returned sources; 0=no truncation
    query_score_threshold: float = 0.0  # min cosine score at Qdrant (0=disabled)
    min_vector_score: float = 0.15  # drop results with vector_score below this (filters junk)
    use_hybrid_search: bool = True  # BM25 + vector with RRF
    use_reranker: bool = False  # cross-encoder rerank (adds ~5-15s; set false for speed)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # fast; or BAAI/bge-reranker-base for quality
    rrf_k: int = 60  # RRF constant
    bm25_min_score: float = 1e-9  # ignore lexical candidates when BM25 score is effectively zero

    # Embedding prefix (improves retrieval for similar snippets)
    embed_metadata_prefix: bool = True

    # Summary-first hierarchical indexing (child summaries -> parent raw code)
    summary_generation_enabled: bool = True
    summary_model: str = "gemini-1.5-flash"
    summary_max_concurrency: int = 8
    summary_input_max_chars: int = 5000
    summary_timeout_sec: float = 20.0

    # BM25 index persistence
    bm25_index_path: Path = Path("bm25_index.pkl")

    # Optional LLM for integrated chat (Vertex AI Gemini). For better answers use "gemini-1.5-pro" or "gemini-2.0-flash".
    llm_model: Optional[str] = None  # e.g. "gemini-1.5-flash" (fast) or "gemini-1.5-pro" (better quality)
    llm_enabled: bool = False  # set true when llm_model configured
    llm_max_output_tokens: int = 2048  # lower = faster responses; 4096 for long answers

    # Prefetch + Ask pipeline (Phase 3) — tune for quality vs speed
    prefetch_candidates_k: int = 20  # candidates before rerank (more = better recall)
    prefetch_cache_top_k: int = 6  # top chunks after rerank sent to extractor (more context)
    prefetch_cache_ttl_sec: int = 120  # cache entry TTL
    prefetch_enabled: bool = True  # set False to disable prefetch (saves embedding calls)
    prefetch_min_query_length: int = 12  # don't embed for shorter queries (saves quota)
    prefetch_cooldown_sec: float = 15.0  # skip re-embedding same query within this window
    ask_use_reranker: bool = True  # rerank with cross-encoder for better chunk order (adds ~2–4s)
    rerank_max_candidates: int = 6  # when rerank on: how many to rerank (more = better, slower)
    rerank_timeout_ms: int = 350  # hard timeout for rerank stage; fallback to fused order on timeout
    rerank_skip_if_score_gap_ge: float = 0.02  # skip rerank when top fused score is clearly ahead
    llm_intro_model: Optional[str] = None  # fast model for intro (default: llm_model)
    llm_extractor_model: Optional[str] = "gemini-1.5-flash"  # fast model for extractor; override with pro for quality
    extractor_explanation_max_words: int = 60  # max words for technical_explanation
    ask_intro_context_chars: int = 3000  # intro has no context; kept for compatibility
    ask_intro_max_words: int = 15  # intro must be this many words or fewer
    ask_intro_max_tokens: int = 50  # max output tokens for intro LLM
    ask_extractor_context_chars: int = 8000  # max context chars for extractor (lower for latency/TTFT)
    ask_extractor_max_tokens: int = 512  # max output tokens for extractor (enough for snippet + explanation)

    # Admin / management
    admin_token: Optional[str] = None
    request_logs_path: Path = Path("request_logs.jsonl")  # persist request logs across restarts

    def qdrant_use_local(self) -> bool:
        return not (self.qdrant_url and self.qdrant_url.strip())

    def extensions_list(self) -> list[str]:
        """
        Return configured extensions without dots.
        If code_extensions is "*" or empty, treat as 'all extensions'.
        """
        raw = (self.code_extensions or "").strip()
        if not raw or raw == "*":
            return []
        return [e.strip().lstrip(".") for e in raw.split(",") if e.strip()]


settings = Settings()
