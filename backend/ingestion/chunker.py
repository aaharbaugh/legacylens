"""
COBOL-aware chunking: paragraph/section boundaries (Area A) with fixed-size fallback.
Populates structural and role tags for filtering and boosted retrieval.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# COBOL Area A: columns 8-11 (1-based) = index 7-10 (0-based). Paragraph name ends with period.
AREA_A_START = 7
AREA_A_END = 11
# Minimum line length to have a period in area B
MIN_LINE_LEN = 12

# Role tags: keywords (uppercase) that imply a tag. Order doesn't matter.
_ROLE_KEYWORDS: dict[str, set[str]] = {
    "file_io": {"READ", "WRITE", "REWRITE", "DELETE", "OPEN", "CLOSE", "START"},
    "display_io": {"DISPLAY", "ACCEPT"},
    "call_external": {"CALL", "GOBACK"},
    "control_flow": {"PERFORM", "IF", "END-IF", "EVALUATE", "END-EVALUATE", "GO ", "GO TO", "EXIT PROGRAM", "EXIT PARAGRAPH", "EXIT SECTION"},
    "data_definition": {"01 ", "77 ", "PIC ", "PICTURE ", "USAGE ", "VALUE ", "88 ", "REDEFINES", "OCCURS "},
    "business_logic": {"COMPUTE", "ADD ", "SUBTRACT", "MULTIPLY", "DIVIDE", "MOVE "},
    "error_handling": {"ON EXCEPTION", "NOT ON EXCEPTION", "INVALID KEY", "AT END", "NOT AT END"},
}

@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    division: str | None
    section_name: str | None
    paragraph_name: str | None
    code_snippet: str
    language: str = "COBOL"
    source_type: str = "code"
    file_ext: str | None = None
    tags: list[str] | None = None
    program_id: str | None = None

    def to_payload(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "division": self.division,
            "section_name": self.section_name,
            "paragraph_name": self.paragraph_name,
            "code_snippet": self.code_snippet,
            "search_text": self.search_text,
            "language": self.language,
            "source_type": self.source_type,
            "file_ext": self.file_ext,
            "tags": self.tags or [],
            "program_id": self.program_id,
        }

    @property
    def search_text(self) -> str:
        """Text used for embedding and BM25: metadata prefix + snippet for better retrieval."""
        parts: list[str] = []
        if self.program_id:
            parts.append(f"[program:{self.program_id}]")
        if self.division:
            parts.append(f"[{self.division}]")
        if self.section_name:
            parts.append(f"[section:{self.section_name}]")
        if self.paragraph_name:
            parts.append(f"[para:{self.paragraph_name}]")
        parts.append(self.file_path)
        parts.append(f"L{self.start_line}-{self.end_line}")
        if self.tags:
            parts.extend(self.tags[:5])  # limit tag noise
        prefix = " ".join(parts)
        return f"{prefix}\n{self.code_snippet}"


_DOC_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".adoc",
    ".texi",
    ".info",
}


def _infer_source_type(path: Path) -> str:
    return "docs" if path.suffix.lower() in _DOC_EXTENSIONS else "code"


def _infer_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".cob", ".cbl"}:
        return "COBOL"
    if ext == ".c":
        return "C"
    if ext == ".h":
        return "C Header"
    if ext == ".md":
        return "Markdown"
    if ext in {".txt", ".info", ".texi", ".rst", ".adoc"}:
        return "Text"
    return ext.lstrip(".").upper() if ext else "TEXT"


def _is_paragraph_boundary(line: str) -> bool:
    """True if line looks like a COBOL paragraph/section name in Area A ending with period."""
    if len(line) < MIN_LINE_LEN:
        return False
    # Exclude division headers (IDENTIFICATION/DATA/PROCEDURE DIVISION.)
    up = line.upper()
    if " DIVISION" in up:
        return False
    # Exclude data section headers (WORKING-STORAGE SECTION. etc); keep procedure sections (MAIN SECTION.)
    if " SECTION" in up:
        segment = line[AREA_A_START:].rstrip().rstrip(".")
        if " SECTION" in segment.upper():
            before_section = segment.upper().split(" SECTION")[0].strip()
            if " " in before_section:
                return False  # multi-word e.g. WORKING-STORAGE
    # Area A: cols 8-11 (0-indexed 7-11), must have non-space and line must end with period
    area_a = line[AREA_A_START:AREA_A_END + 1]
    rest = line[AREA_A_END + 1 :].rstrip()
    if not area_a.strip():
        return False
    if not rest.endswith("."):
        return False
    # Exclude DATA DIVISION level numbers (01, 77, 02-49) in Area A
    segment = line[AREA_A_START:].rstrip()
    if segment.endswith("."):
        segment = segment[:-1].rstrip()
    first_word = segment.split()[0] if segment else ""
    if first_word.isdigit() and len(first_word) <= 2:
        return False
    # Optional: exclude comment lines (column 7 is * or /)
    if line[:7].strip() and line[6:7] in "*!/":
        return False
    return True


def _extract_paragraph_name(line: str) -> str | None:
    """Return the identifier in Area A (and up to the period) or None."""
    if not _is_paragraph_boundary(line):
        return None
    # Take from start of Area A to the period
    segment = line[AREA_A_START:].rstrip()
    if segment.endswith("."):
        segment = segment[:-1].rstrip()
    return segment.split()[0] if segment else None


def _detect_division(line: str) -> str | None:
    """Return PROCEDURE, DATA, ENVIRONMENT, etc. if line is a division header."""
    s = " ".join(line.strip().upper().split())  # normalize spaces
    for div in ("IDENTIFICATION DIVISION", "ENVIRONMENT DIVISION", "DATA DIVISION", "PROCEDURE DIVISION"):
        if s.startswith(div) or s == div.replace(" DIVISION", "."):
            return div.split()[0]
    return None


def _extract_program_id(lines: list[str]) -> str | None:
    """Return PROGRAM-ID name from IDENTIFICATION DIVISION, or None."""
    in_id_div = False
    for line in lines:
        s = line.strip().upper()
        if "IDENTIFICATION" in s and "DIVISION" in s:
            in_id_div = True
            continue
        if in_id_div:
            if "ENVIRONMENT" in s or "DATA" in s or "PROCEDURE" in s:
                break
            # PROGRAM-ID. name or PROGRAM-ID. name.
            if s.startswith("PROGRAM-ID"):
                rest = line.strip()[11:].lstrip(".").strip()  # PROGRAM-ID = 11 chars
                # take first word or first token
                name = rest.split()[0] if rest.split() else None
                if name:
                    return name.rstrip(".")
    return None


def _structural_tags(
    division: str | None,
    section_name: str | None,
    paragraph_name: str | None,
    program_id: str | None,
) -> list[str]:
    """Build structural tags for filter/boost: div:, data:, para:, section:, program:."""
    tags: list[str] = []
    if division:
        tags.append(f"div:{division}")
    if division == "DATA" and section_name:
        # Normalize: WORKING-STORAGE SECTION -> WORKING-STORAGE
        sec = section_name.upper().replace(" SECTION", "").strip()
        if sec in ("WORKING-STORAGE", "LINKAGE", "FILE", "LOCAL-STORAGE"):
            tags.append(f"data:{sec}")
    if paragraph_name:
        tags.append(f"para:{paragraph_name}")
    if section_name and section_name != paragraph_name:
        tags.append(f"section:{section_name}")
    if program_id:
        tags.append(f"program:{program_id}")
    return tags


def _infer_role_tags(snippet: str) -> list[str]:
    """Infer role tags from COBOL verbs/keywords in chunk text."""
    if not snippet.strip():
        return []
    upper = snippet.upper()
    tags: list[str] = []
    for tag, keywords in _ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in upper:
                tags.append(tag)
                break
    return tags


def _emit_chunk(
    chunk_lines: list[str],
    chunk_start: int,
    end_line: int,
    current_division: str | None,
    current_section: str | None,
    paragraph_name: str | None,
    program_id: str | None,
    file_path: str,
) -> CodeChunk:
    """Build one CodeChunk from accumulated lines and metadata."""
    snippet = "\n".join(chunk_lines)
    struct = _structural_tags(
        current_division, current_section, paragraph_name, program_id
    )
    role = _infer_role_tags(snippet)
    return CodeChunk(
        file_path=file_path,
        start_line=chunk_start,
        end_line=end_line,
        division=current_division,
        section_name=current_section,
        paragraph_name=paragraph_name,
        code_snippet=snippet,
        tags=struct + role,
        program_id=program_id,
    )


def _chunk_by_paragraphs(
    lines: list[str],
    file_path: str,
    program_id: str | None = None,
    max_chunk_lines: int = 80,
) -> Iterator[CodeChunk]:
    """Yield chunks using paragraph boundaries, capped at max_chunk_lines (avoids huge data-division blocks)."""
    current_division: str | None = None
    current_section: str | None = None
    chunk_start = 1  # 1-based line number
    chunk_lines: list[str] = []
    paragraph_name: str | None = None

    def flush(sub_chunk_start: int) -> CodeChunk | None:
        """Emit current chunk and reset; returns chunk if any, else None."""
        nonlocal chunk_start, chunk_lines, paragraph_name
        if not chunk_lines:
            return None
        end = chunk_start + len(chunk_lines) - 1
        c = _emit_chunk(
            chunk_lines,
            chunk_start,
            end,
            current_division,
            current_section,
            paragraph_name,
            program_id,
            file_path,
        )
        chunk_start = sub_chunk_start
        chunk_lines = []
        paragraph_name = None
        return c

    for i, line in enumerate(lines):
        one_based = i + 1
        div = _detect_division(line)
        if div:
            current_division = div
            current_section = None
        if _is_paragraph_boundary(line):
            name = _extract_paragraph_name(line)
            if name:
                current_section = name
                paragraph_name = name
            out = flush(one_based)
            if out is not None:
                yield out
            chunk_lines = [line]
            paragraph_name = _extract_paragraph_name(line)
            continue
        if len(chunk_lines) >= max_chunk_lines:
            out = flush(one_based)
            if out is not None:
                yield out
            chunk_lines = [line]
            continue
        chunk_lines.append(line)

    if chunk_lines:
        yield _emit_chunk(
            chunk_lines,
            chunk_start,
            len(lines),
            current_division,
            current_section,
            paragraph_name,
            program_id,
            file_path,
        )


def _chunk_fixed_size(
    lines: list[str],
    file_path: str,
    chunk_lines: int,
    overlap: int,
    language: str = "COBOL",
    source_type: str = "code",
    file_ext: str | None = None,
    program_id: str | None = None,
) -> Iterator[CodeChunk]:
    """Fallback: fixed-size windows with overlap. Only role tags (no structural)."""
    for start in range(0, len(lines), chunk_lines - overlap):
        window = lines[start : start + chunk_lines]
        if not window:
            continue
        snippet = "\n".join(window)
        role = _infer_role_tags(snippet)
        tags = ([f"program:{program_id}"] if program_id else []) + role
        yield CodeChunk(
            file_path=file_path,
            start_line=start + 1,
            end_line=start + len(window),
            division=None,
            section_name=None,
            paragraph_name=None,
            code_snippet=snippet,
            language=language,
            source_type=source_type,
            file_ext=file_ext,
            tags=tags,
            program_id=program_id,
        )


def chunk_file(
    file_path: Path | str,
    *,
    max_paragraph_chunk_lines: int = 80,
    fallback_chunk_lines: int = 80,
    fallback_overlap_lines: int = 15,
) -> list[CodeChunk]:
    """
    Chunk a single COBOL file. Uses paragraph boundaries when possible (capped at
    max_paragraph_chunk_lines); otherwise falls back to fixed-size chunks.
    """
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    if not lines:
        return []

    language = _infer_language(path)
    source_type = _infer_source_type(path)
    file_ext = path.suffix.lower().lstrip(".") or None
    program_id = _extract_program_id(lines)

    try:
        chunks = list(
            _chunk_by_paragraphs(
                lines, str(path), program_id, max_chunk_lines=max_paragraph_chunk_lines
            )
        )
    except Exception:
        chunks = []

    # Fallback if no paragraph-based chunks (e.g. non-standard format)
    if not chunks:
        chunks = list(
            _chunk_fixed_size(
                lines,
                str(path),
                fallback_chunk_lines,
                fallback_overlap_lines,
                language=language,
                source_type=source_type,
                file_ext=file_ext,
                program_id=program_id,
            )
        )
    elif len(chunks) == 1 and len(lines) > fallback_chunk_lines:
        # Single oversized chunk – re-chunk with fixed size
        chunks = list(
            _chunk_fixed_size(
                lines,
                str(path),
                fallback_chunk_lines,
                fallback_overlap_lines,
                language=language,
                source_type=source_type,
                file_ext=file_ext,
                program_id=program_id,
            )
        )

    for c in chunks:
        c.language = language
        c.source_type = source_type
        c.file_ext = file_ext

    return chunks
