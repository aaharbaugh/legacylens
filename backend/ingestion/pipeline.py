"""
Ingestion pipeline: discover -> chunk -> embed -> upsert to Qdrant.
Run from repo root: python -m backend.ingestion.pipeline run --code-root /path/to/gnucobol
"""
import argparse
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from backend.config import settings
from backend.ingestion.bm25_index import BM25Index, build_index
from backend.ingestion.chunker import CodeChunk, chunk_file
from backend.ingestion.discovery import discover_files
from backend.ingestion.embedder import Embedder
from backend.ingestion.vector_store import ensure_collection, get_client, make_point_id, upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    code_root: Path,
    *,
    extensions: list[str] | None = None,
    batch_size: int = 100,
    max_files: int | None = None,
    client: QdrantClient | None = None,
) -> tuple[int, int]:
    """
    Ingest all matching files under code_root into Qdrant.
    Returns (files_processed, chunks_upserted).
    """
    # Important: empty list means "all file extensions".
    # Only fall back to settings when extensions is truly unset.
    if extensions is None:
        extensions = settings.extensions_list()
    if max_files is None:
        max_files = settings.max_files
    if client is None:
        client = get_client()
    ensure_collection(client)
    embedder = Embedder()
    files_processed = 0
    total_chunks = 0
    all_chunks: list[dict] = []
    bm25_corpus: list[dict] = []  # accumulate for BM25 index

    for path in discover_files(code_root, extensions):
        if max_files is not None and files_processed >= max_files:
            logger.info("Reached max_files=%d, stopping discovery", max_files)
            break
        try:
            chunks = chunk_file(
                path,
                max_paragraph_chunk_lines=settings.max_paragraph_chunk_lines,
                fallback_chunk_lines=settings.fallback_chunk_lines,
                fallback_overlap_lines=settings.fallback_overlap_lines,
            )
        except Exception as e:
            logger.warning("Chunking failed for %s: %s", path, e)
            continue
        root_resolved = code_root.resolve()
        for c in chunks:
            if not isinstance(c, CodeChunk):
                continue
            payload = c.to_payload()
            # Store path relative to code_root for cleaner display
            try:
                abs_path = Path(c.file_path).resolve()
                rel = abs_path.relative_to(root_resolved)
                payload["file_path"] = rel.as_posix()
            except (ValueError, TypeError):
                pass
            all_chunks.append(payload)
        files_processed += 1
        # Batch embed and upsert
        if len(all_chunks) >= batch_size:
            for c in all_chunks:
                bm25_corpus.append(dict(c, id=make_point_id(c)))
            vectors = embedder.embed_chunks(all_chunks)
            upsert_chunks(client, all_chunks, vectors)
            total_chunks += len(all_chunks)
            logger.info("Upserted batch of %d chunks (%d files so far)", len(all_chunks), files_processed)
            all_chunks = []

    if all_chunks:
        for c in all_chunks:
            bm25_corpus.append(dict(c, id=make_point_id(c)))
        vectors = embedder.embed_chunks(all_chunks)
        upsert_chunks(client, all_chunks, vectors)
        total_chunks += len(all_chunks)

    # Build BM25 index for hybrid search
    if settings.use_hybrid_search and bm25_corpus:
        idx = build_index(bm25_corpus)
        if idx:
            idx.save()
            logger.info("Built BM25 index (%d docs)", len(idx.doc_ids))

    return files_processed, total_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="LegacyLens ingestion pipeline")
    parser.add_argument("run", nargs="?", default="run", help="run (default)")
    parser.add_argument("--code-root", type=Path, default=settings.code_root, help="Root directory of codebase")
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size, help="Embed batch size")
    args = parser.parse_args()
    if not args.code_root or not args.code_root.is_dir():
        logger.error("Provide a valid --code-root directory (or set CODE_ROOT in env)")
        return 1
    files, chunks = run_pipeline(args.code_root, batch_size=args.batch_size)
    logger.info("Done: %d files, %d chunks upserted", files, chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
