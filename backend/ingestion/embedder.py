"""
Embedding generation: OpenAI embeddings API with batch + cache.
Fallback: deterministic pseudo-embeddings when OPENAI_API_KEY is not set.
"""
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

VECTOR_SIZE = 768
# Only log the first OpenAI fallback per process to avoid spamming
_openai_fallback_logged = False


def _pseudo_embed(text: str) -> list[float]:
    """Deterministic 768-dim vector from text hash. For dev without OpenAI API key."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(0, min(768 * 4, len(h) * 4), 4):
        b = h[i % len(h) : (i % len(h)) + 4].ljust(4, b"\x00")
        val = (int.from_bytes(b, "big") / (2**32)) * 0.2 - 0.1
        out.append(val)
    while len(out) < VECTOR_SIZE:
        out.append(0.0)
    return out[:VECTOR_SIZE]


def _estimate_tokens(text: str) -> int:
    """Rough token count. Code is ~3–4 chars per token."""
    return max(1, len(text) // 3)


def _split_texts_by_token_limit(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Split texts into sub-batches that each stay under max_tokens total."""
    if max_tokens <= 0:
        return [texts]
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for t in texts:
        n = _estimate_tokens(t)
        if current_tokens + n > max_tokens and current:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(t)
        current_tokens += n
    if current:
        batches.append(current)
    return batches


def _embed_one_batch(sub: list[str], model: str, dimensions: int) -> list[list[float]]:
    """Embed a single sub-batch via OpenAI API."""
    from backend.llm_client import openai_embed
    return openai_embed(sub, model=model, dimensions=dimensions)


def _get_openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API. Splits by token limit; runs sub-batches with optional throttle."""
    if not (settings.openai_api_key and settings.openai_api_key.strip()):
        raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")
    max_tokens = settings.embed_max_tokens_per_request
    sub_batches = _split_texts_by_token_limit(texts, max_tokens)
    delay_sec = getattr(settings, "embed_delay_between_requests_sec", 0.0) or 0.0
    max_workers = max(1, min(getattr(settings, "embed_max_workers", 1), len(sub_batches)))

    if len(sub_batches) == 1:
        return _embed_one_batch(
            sub_batches[0],
            settings.embed_model,
            settings.embed_dimensions,
        )

    all_vectors: list[list[float]] = []
    if max_workers <= 1:
        for i, sub in enumerate(sub_batches):
            if delay_sec > 0 and i > 0:
                time.sleep(delay_sec)
            all_vectors.extend(
                _embed_one_batch(sub, settings.embed_model, settings.embed_dimensions)
            )
        return all_vectors

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_embed_one_batch, sub, settings.embed_model, settings.embed_dimensions): i
            for i, sub in enumerate(sub_batches)
        }
        results = [None] * len(sub_batches)
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            if delay_sec > 0:
                time.sleep(delay_sec)
    for r in results:
        all_vectors.extend(r)
    return all_vectors


class Embedder:
    """Batch embed with optional file cache. Uses OpenAI when OPENAI_API_KEY is set, else pseudo-embedding."""

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        use_openai: bool | None = None,
    ):
        self.cache_path = cache_path or settings.embeddings_cache_path
        self._cache: dict[str, list[float]] = {}
        self._load_cache()
        self._use_openai = (
            use_openai
            if use_openai is not None
            else bool(settings.openai_api_key and settings.openai_api_key.strip())
        )
        if self._use_openai:
            logger.info("Embeddings: OpenAI (%s).", settings.embed_model)
        else:
            logger.info("Embeddings: pseudo (no OPENAI_API_KEY). Set it in .env for real embeddings.")

    def _load_cache(self) -> None:
        if not self.cache_path or not Path(self.cache_path).exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)
        except Exception as e:
            logger.warning("Could not load embedding cache %s: %s", self.cache_path, e)

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=0)
        except Exception as e:
            logger.warning("Could not save embedding cache %s: %s", self.cache_path, e)

    @staticmethod
    def chunk_id(chunk: dict[str, Any]) -> str:
        """Stable id for cache key."""
        key = (
            chunk.get("file_path", "")
            + "|"
            + str(chunk.get("start_line"))
            + "|"
            + str(chunk.get("end_line"))
            + "|"
            + (chunk.get("code_snippet") or "")
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Uses cache when keyed by content hash."""
        results: list[list[float]] = []
        to_fetch: list[int] = []
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if h in self._cache:
                results.append(self._cache[h])
            else:
                results.append([])
                to_fetch.append(i)
        if not to_fetch:
            return results
        batch_texts = [texts[i] for i in to_fetch]
        if self._use_openai:
            try:
                batch_vectors = _get_openai_embeddings(batch_texts)
            except Exception as e:
                global _openai_fallback_logged
                if not _openai_fallback_logged:
                    logger.warning(
                        "OpenAI embedding failed, using pseudo embeddings for this run: %s",
                        e,
                    )
                    _openai_fallback_logged = True
                else:
                    logger.debug("OpenAI embedding failed (pseudo): %s", e)
                batch_vectors = [_pseudo_embed(t) for t in batch_texts]
        else:
            batch_vectors = [_pseudo_embed(t) for t in batch_texts]
        for idx, j in enumerate(to_fetch):
            vec = batch_vectors[idx]
            results[j] = vec
            h = hashlib.sha256(texts[j].encode("utf-8")).hexdigest()
            self._cache[h] = vec
        self._save_cache()
        return results

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """Embed by code_snippet, optionally prefixed by metadata_prefix."""
        if settings.embed_metadata_prefix:
            texts = [self._retrieval_text(c, include_prefix=True) for c in chunks]
        else:
            texts = [self._retrieval_text(c, include_prefix=False) for c in chunks]
        return self.embed_texts(texts)

    @staticmethod
    def _retrieval_text(chunk: dict[str, Any], *, include_prefix: bool) -> str:
        # Summary-first indexing: embed summary text when available, keep raw code in payload.
        snippet = chunk.get("summary_text") or chunk.get("code_snippet") or ""
        if not include_prefix:
            return snippet
        prefix = (chunk.get("metadata_prefix") or "").strip()
        return f"{prefix}\n{snippet}" if prefix else snippet
