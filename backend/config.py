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

    # Embedding (larger pipeline batch = fewer cycles; embedder splits by token limit and runs sub-batches in parallel)
    embed_batch_size: int = 40  # chunks per pipeline batch
    embed_max_tokens_per_request: int = 18_000  # Vertex allows 20k; stay under to avoid INVALID_ARGUMENT
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
    fallback_overlap_lines: int = 10

    # Retrieval
    query_top_k: int = 50  # candidates before rerank/fusion (main /query)
    query_chat_top_k: int = 30  # candidate pool for chat; larger = better for keyword queries (e.g. EVALUATE)
    query_final_k: int = 25  # chunks for /query endpoint
    query_chat_final_k: int = 12  # chunks sent to LLM; more = better chance to hit relevant code
    chat_snippet_max_lines: int = 50  # truncate long chunks in prompt; 0=no truncation (lower=faster LLM)
    query_score_threshold: float = 0.0  # min cosine score at Qdrant (0=disabled)
    min_vector_score: float = 0.15  # drop results with vector_score below this (filters junk)
    use_hybrid_search: bool = True  # BM25 + vector with RRF
    use_reranker: bool = False  # cross-encoder rerank (adds ~5-15s; set false for speed)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # fast; or BAAI/bge-reranker-base for quality
    rrf_k: int = 60  # RRF constant

    # Embedding prefix (improves retrieval for similar snippets)
    embed_metadata_prefix: bool = True

    # BM25 index persistence
    bm25_index_path: Path = Path("bm25_index.pkl")

    # Optional LLM for integrated chat (Vertex AI Gemini)
    llm_model: Optional[str] = None  # e.g. "gemini-1.5-flash" (faster) or "gemini-1.0-pro"
    llm_enabled: bool = False  # set true when llm_model configured
    llm_max_output_tokens: int = 2048  # lower = faster responses; 4096 for long answers

    # Admin / management
    admin_token: Optional[str] = None

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
