"""
Ingestion pipeline: discover -> chunk -> embed -> upsert to Qdrant.
Run from repo root: python -m backend.ingestion.pipeline run --code-root /path/to/gnucobol
"""
import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from backend.config import settings
from backend.ingestion.bm25_index import build_index
from backend.ingestion.chunker import CodeChunk, chunk_file
from backend.ingestion.discovery import discover_files
from backend.ingestion.embedder import Embedder
from backend.ingestion.summarizer import FunctionalSummaryGenerator
from backend.ingestion.vector_store import ensure_collection, get_client, make_point_id, upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

_PHASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontend": ("lexer", "scanner", "token", "parser", "grammar", "parse", "syntax", "yacc", "bison"),
    "middle-end": ("ir", "intermediate", "optimizer", "optim", "ssa", "cfg", "semantic", "analy"),
    "backend": ("backend", "codegen", "emit", "asm", "assembler", "register", "target", "machine", "object"),
}


def _normalize_payload_path(raw_path: str, code_root: Path) -> str:
    """
    Store paths as  <code_root_name>/<relative_path>  (e.g. gnucobol-3.2_win/tests/foo.at).
    No absolute paths ever reach the payload.
    """
    norm_fallback = (raw_path or "").replace("\\", "/")
    try:
        abs_path = Path(raw_path).resolve()
    except Exception:
        return norm_fallback
    root = code_root.resolve()
    root_name = root.name  # e.g. "gnucobol-3.2_win"
    try:
        rel = abs_path.relative_to(root).as_posix()
        return f"{root_name}/{rel}"
    except ValueError:
        # Windows drive/casing edge cases
        try:
            rel = os.path.relpath(str(abs_path), str(root)).replace("\\", "/")
            if rel and not rel.startswith("../") and rel != "..":
                return f"{root_name}/{rel}"
        except Exception:
            pass
        return norm_fallback


def _build_metadata_prefix(payload: dict) -> str:
    """Build metadata_prefix from an already-normalized payload dict."""
    parts: list[str] = []
    if payload.get("phase"):
        parts.append(f"[phase:{payload['phase']}]")
    if payload.get("program_id"):
        parts.append(f"[program:{payload['program_id']}]")
    if payload.get("division"):
        parts.append(f"[{payload['division']}]")
    if payload.get("section_name"):
        parts.append(f"[section:{payload['section_name']}]")
    if payload.get("paragraph_name"):
        parts.append(f"[para:{payload['paragraph_name']}]")
    parts.append(payload.get("file_name") or Path(payload.get("file_path", "")).name)
    parts.append(f"L{payload.get('start_line', '')}-{payload.get('end_line', '')}")
    tags = payload.get("tags") or []
    if tags:
        parts.extend(tags[:5])
    return " ".join(parts)


def _extract_include_headers(file_path: Path) -> list[str]:
    """Parse local #include "header.h" dependencies from a source file."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    headers: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _INCLUDE_RE.match(line)
        if not m:
            continue
        header = m.group(1).strip()
        if header and header not in seen:
            seen.add(header)
            headers.append(header)
    return headers


def _infer_phase(norm_path: str, payload: dict) -> str:
    """Compiler phase routing tag used for retrieval filters and hierarchy."""
    hay = " ".join(
        [
            norm_path.lower(),
            str(payload.get("function_name") or "").lower(),
            str(payload.get("paragraph_name") or "").lower(),
            " ".join(str(t).lower() for t in (payload.get("tags") or [])),
        ]
    )
    for phase, kws in _PHASE_KEYWORDS.items():
        if any(k in hay for k in kws):
            return phase
    # Conservative fallback: compiler internals often flow middle-end by default
    return "middle-end"


def _stable_parent_id(payload: dict) -> str:
    """Stable parent identifier for parent-child retrieval linkage."""
    key = (
        (payload.get("file_path") or "")
        + "|"
        + str(payload.get("start_line") or "")
        + "|"
        + str(payload.get("end_line") or "")
        + "|"
        + (payload.get("code_snippet") or "")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _attach_summaries(chunks: list[dict], summaries: list[str]) -> list[dict]:
    for i, c in enumerate(chunks):
        summary = summaries[i] if i < len(summaries) else ""
        c["summary_text"] = summary
        c["chunk_type"] = "child_summary"
        # For parent-child retrieval: child points to canonical parent id.
        # Raw parent code remains in code_snippet (payload page_content analogue).
        c["retrieval_parent_id"] = c.get("parent_id")
        # metadata_prefix should include phase and structural context; keep summary out to reduce noise.
        c["metadata_prefix"] = _build_metadata_prefix(c)
    return chunks


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
    summarizer = FunctionalSummaryGenerator()
    files_processed = 0
    total_chunks = 0
    all_chunks: list[dict] = []
    bm25_corpus: list[dict] = []  # accumulate for BM25 index

    for path in discover_files(code_root, extensions):
        if max_files is not None and files_processed >= max_files:
            logger.info("Reached max_files=%d, stopping discovery", max_files)
            break
        include_headers = _extract_include_headers(path)
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
        for c in chunks:
            if not isinstance(c, CodeChunk):
                continue
            payload = c.to_payload()
            norm_path = _normalize_payload_path(c.file_path, code_root)
            p = Path(norm_path)
            payload["file_path"] = norm_path
            payload["file_dir"] = p.parent.as_posix()
            payload["file_name"] = p.stem
            # file_ext already set by chunker; keep it consistent
            payload["file_ext"] = p.suffix.lstrip(".").lower() or None
            # Routing tag: top-level folder (e.g. cobc, libcob) for hard-filtering in vector search
            payload["folder"] = (
                p.parts[1] if len(p.parts) > 2 else (p.parts[0] if p.parts else "")
            )
            payload["phase"] = _infer_phase(norm_path, payload)
            payload["include_headers"] = include_headers
            payload["parent_id"] = _stable_parent_id(payload)
            # Recompute metadata_prefix now that file_path is normalized to relative
            payload["metadata_prefix"] = _build_metadata_prefix(payload)
            all_chunks.append(payload)
        files_processed += 1
        # Batch embed and upsert
        if len(all_chunks) >= batch_size:
            summaries = asyncio.run(summarizer.summarize_payloads(all_chunks))
            all_chunks = _attach_summaries(all_chunks, summaries)
            for c in all_chunks:
                bm25_corpus.append(dict(c, id=make_point_id(c)))
            vectors = embedder.embed_chunks(all_chunks)
            upsert_chunks(client, all_chunks, vectors)
            total_chunks += len(all_chunks)
            logger.info("Upserted batch of %d chunks (%d files so far)", len(all_chunks), files_processed)
            all_chunks = []

    if all_chunks:
        summaries = asyncio.run(summarizer.summarize_payloads(all_chunks))
        all_chunks = _attach_summaries(all_chunks, summaries)
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
