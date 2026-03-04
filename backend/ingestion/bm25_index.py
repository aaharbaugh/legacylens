"""
BM25 index for hybrid retrieval. Built at ingestion, persisted to disk.
Uses rank_bm25 for lexical search; results fused with vector search via RRF.
"""
import logging
import pickle
import re
from pathlib import Path
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = Path("bm25_index.pkl")


def _tokenize(text: str) -> list[str]:
    """
    Tokenizer for COBOL/code: keeps hyphenated terms (END-IF, 88-level) as single tokens.
    Splits on spaces and punctuation; lowercase; filters very short tokens.
    """
    text = (text or "").lower()
    # Include hyphen so COBOL words like end-if, 88-level stay together
    tokens = re.findall(r"[a-z0-9_-]+", text)
    return [t for t in tokens if len(t) >= 2]


def build_index(chunks: list[dict[str, Any]]) -> "BM25Index | None":
    """
    Build BM25 index from chunks. Each chunk must have 'id' and 'search_text' or 'code_snippet'.
    Stores payload map so BM25-only hits have metadata without Qdrant retrieve.
    Returns the index or None if chunks empty.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; hybrid search disabled")
        return None
    if not chunks:
        return None
    doc_ids: list[str] = []
    corpus_tokens: list[list[str]] = []
    id_to_payload: dict[str, dict] = {}
    for c in chunks:
        pid = c.get("id")
        if not pid:
            continue
        text = c.get("search_text") or c.get("code_snippet") or ""
        tokens = _tokenize(text)
        if not tokens:
            continue
        doc_ids.append(pid)
        corpus_tokens.append(tokens)
        id_to_payload[pid] = {
            "file_path": c.get("file_path", ""),
            "start_line": c.get("start_line", 0),
            "end_line": c.get("end_line", 0),
            "code_snippet": (c.get("code_snippet") or "")[:65535],
            "language": c.get("language", "COBOL"),
            "source_type": c.get("source_type", "code"),
            "division": c.get("division"),
            "section_name": c.get("section_name"),
            "paragraph_name": c.get("paragraph_name"),
        }
    if not doc_ids:
        return None
    bm25 = BM25Okapi(corpus_tokens)
    return BM25Index(doc_ids=doc_ids, bm25=bm25, corpus_tokens=corpus_tokens, id_to_payload=id_to_payload)


class BM25Index:
    """Persisted BM25 index with point_id mapping and payload cache."""

    def __init__(
        self,
        *,
        doc_ids: list[str],
        bm25: Any,
        corpus_tokens: list[list[str]],
        id_to_payload: dict[str, dict] | None = None,
    ):
        self.doc_ids = doc_ids
        self.bm25 = bm25
        self.corpus_tokens = corpus_tokens
        self.id_to_payload = id_to_payload or {}

    def search(
        self,
        query: str,
        limit: int = 50,
        source_type: str | None = None,
        tags_filter: list[str] | None = None,
        id_to_payload: dict[str, dict] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top limit (point_id, score) pairs. Optional filter by source_type/tags
        if id_to_payload is provided.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        indexed = list(zip(self.doc_ids, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for pid, score in indexed[: limit * 2]:  # overfetch for filtering
            if id_to_payload and (source_type or tags_filter):
                payload = id_to_payload.get(pid) or {}
                if source_type and source_type != "all" and payload.get("source_type") != source_type:
                    continue
                if tags_filter:
                    chunk_tags = set(payload.get("tags") or [])
                    if not any(t in chunk_tags for t in tags_filter):
                        continue
            results.append((pid, float(score)))
            if len(results) >= limit:
                break
        return results

    def save(self, path: Path | None = None) -> None:
        path = path or settings.bm25_index_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "bm25": self.bm25,
                    "corpus_tokens": self.corpus_tokens,
                    "id_to_payload": self.id_to_payload,
                },
                f,
            )
        logger.info("Saved BM25 index to %s (%d docs)", path, len(self.doc_ids))

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index | None":
        path = path or settings.bm25_index_path
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls(
                doc_ids=data["doc_ids"],
                bm25=data["bm25"],
                corpus_tokens=data["corpus_tokens"],
                id_to_payload=data.get("id_to_payload"),
            )
        except Exception as e:
            logger.warning("Could not load BM25 index %s: %s", path, e)
            return None
