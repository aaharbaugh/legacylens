import re
from collections import defaultdict
from typing import Any

from backend.api.schemas import RetrievedChunk

# Known extensionless doc files - when query mentions these, fetch by file for reliable retrieval
_DOC_FILENAMES = frozenset({"thanks", "readme", "authors", "news", "copying", "todo", "changelog"})


def extract_doc_filename(query: str) -> str | None:
    """If query mentions a known doc file (e.g. 'THANKS file'), return its name."""
    q = query.lower()
    for name in _DOC_FILENAMES:
        if re.search(rf"\b{re.escape(name)}\b", q):
            return name.upper()
    return None


def truncate_snippet(snippet: str, max_lines: int) -> str:
    """Truncate snippet to max_lines for display/response; 0 = no truncation."""
    if not snippet or max_lines <= 0:
        return snippet or ""
    lines = snippet.splitlines()
    if len(lines) <= max_lines:
        return snippet
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def rank_hits_by_file(
    hits: list[dict[str, Any]],
    max_per_file: int,
    final_k: int,
) -> list[dict[str, Any]]:
    """Reorder hits by best file, cap chunks per file for prompt diversity."""
    if max_per_file <= 0 or not hits:
        return hits[:final_k]
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hits:
        fp = (h.get("payload") or {}).get("file_path") or ""
        by_file[fp].append(h)

    def file_best_score(chunk_list: list[dict[str, Any]]) -> float:
        return max(
            (c.get("vector_score") if c.get("vector_score") is not None else c.get("score")) or 0
            for c in chunk_list
        )

    sorted_files = sorted(
        by_file.items(),
        key=lambda x: file_best_score(x[1]),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for _fp, chunks in sorted_files:
        for c in chunks[:max_per_file]:
            out.append(c)
            if len(out) >= final_k:
                return out
    return out[:final_k]


def hits_to_retrieved_chunks(hits: list[dict[str, Any]], snippet_max_lines: int = 0) -> list[RetrievedChunk]:
    """Convert hybrid_search hits to RetrievedChunk list."""
    return [
        RetrievedChunk(
            id=h.get("id", ""),
            score=h.get("score", 0.0),
            vector_score=h.get("vector_score"),
            file_path=(h.get("payload") or {}).get("file_path", ""),
            start_line=(h.get("payload") or {}).get("start_line", 0),
            end_line=(h.get("payload") or {}).get("end_line", 0),
            division=(h.get("payload") or {}).get("division"),
            section_name=(h.get("payload") or {}).get("section_name"),
            paragraph_name=(h.get("payload") or {}).get("paragraph_name"),
            code_snippet=truncate_snippet((h.get("payload") or {}).get("code_snippet", ""), snippet_max_lines),
            language=(h.get("payload") or {}).get("language", "COBOL"),
            source_type=(h.get("payload") or {}).get("source_type", "code"),
        )
        for h in hits
    ]
