"""
Recursively discover COBOL (and other configured) source files under a code root.
"""
from pathlib import Path
from typing import Iterable

from backend.config import settings


# Extensionless doc files - yielded first so they're never cut off by max_files
_EXTENSIONLESS_DOCS = frozenset(
    {"thanks", "readme", "authors", "news", "copying", "todo", "changelog", "about-nls", "hacking", "dependencies"}
)

# Build/config artifacts - exclude from ingestion (noise, not useful for code Q&A)
_EXCLUDE_EXTENSIONS = frozenset(
    {"vcxproj", "vcxproj.filters", "sln", "suo", "user", "ncb", "aps", "sdf", "opensdf", "db", "cache"}
)
_EXCLUDE_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".qdrant_data", "build_windows", "vs2017", "vs2019", "vs2022", ".vs", "x64", "Win32", "Debug", "Release"}
)


def discover_files(
    code_root: Path,
    extensions: list[str],
    *,
    exclude_dirs: frozenset[str] | None = None,
) -> Iterable[Path]:
    """
    Yield paths to files whose suffix (without dot) is in extensions.
    Skips directories in exclude_dirs.
    Extensionless doc files (THANKS, README, etc.) are yielded first so they're never cut off by max_files.
    """
    code_root = code_root.resolve()
    if not code_root.is_dir():
        return
    exclude = exclude_dirs if exclude_dirs is not None else _EXCLUDE_DIRS
    norm_exts = {e.lstrip(".").lower() for e in extensions} if extensions else None
    max_size_bytes = max(settings.max_file_size_mb, 1) * 1024 * 1024
    yielded: set[Path] = set()

    def _should_include(path: Path) -> bool:
        if not path.is_file():
            return False
        if any(part in exclude for part in path.parts):
            return False
        if path.suffix and path.suffix.lstrip(".").lower() in _EXCLUDE_EXTENSIONS:
            return False
        try:
            if path.stat().st_size > max_size_bytes:
                return False
        except OSError:
            return False
        if settings.ingest_text_only and not _is_text_readable(path):
            return False
        if norm_exts is None or path.suffix.lstrip(".").lower() in norm_exts:
            return True
        if not path.suffix and path.name.lower() in _EXTENSIONLESS_DOCS:
            return True
        return False

    # First: yield extensionless doc files from root (ensures THANKS, README, etc. are never cut off)
    for name in _EXTENSIONLESS_DOCS:
        p = code_root / name
        if _should_include(p):
            yielded.add(p.resolve())
            yield p

    # Then: normal rglob, skipping already yielded
    for path in code_root.rglob("*"):
        if path.resolve() in yielded:
            continue
        if _should_include(path):
            yielded.add(path.resolve())
            yield path


def _is_text_readable(path: Path) -> bool:
    """
    Lightweight text/binary detection using a small file sample.
    Returns True for plain text-like content, False for likely binary blobs.
    """
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    if not sample:
        return True
    # Null bytes are a strong signal for binary formats.
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass

    # Fallback heuristic: reject if too many control chars.
    bad = 0
    for b in sample:
        if b in (9, 10, 13):  # tab/newline/carriage return
            continue
        if b < 32 or b == 127:
            bad += 1
    return (bad / len(sample)) < 0.10
