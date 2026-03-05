import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

_request_logs: deque[dict[str, Any]] = deque(maxlen=200)
_request_logs_lock = threading.Lock()


def load_request_logs() -> None:
    """Load persisted request logs from disk (survives restarts/hard refresh)."""
    path = getattr(settings, "request_logs_path", None) or Path("request_logs.jsonl")
    path = Path(path)
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
        lines = [s.strip() for s in raw.splitlines() if s.strip()]
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
        if entries:
            keep = entries[-200:] if len(entries) > 200 else entries
            with _request_logs_lock:
                _request_logs.clear()
                for e in keep:
                    _request_logs.append(e)
            if len(entries) > 200:
                path.write_text("\n".join(json.dumps(e) for e in keep) + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("Could not load request logs from %s: %s", path, e)


def append_request_log(entry: dict[str, Any]) -> None:
    record = {**entry, "ts": time.time()}
    with _request_logs_lock:
        _request_logs.append(record)
    path = getattr(settings, "request_logs_path", None) or Path("request_logs.jsonl")
    path = Path(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("Could not append request log to %s: %s", path, e)


def recent_request_logs() -> list[dict[str, Any]]:
    with _request_logs_lock:
        return list(reversed(list(_request_logs)))


def ask_log_entry(
    req_type: str,
    total_ms: float,
    timings: dict[str, float],
    llm_ms: float | None = None,
    intro_ms: float | None = None,
    extractor_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Build request log entry for ask/ask_stream with full breakdown."""
    entry: dict[str, Any] = {"type": req_type, "total_ms": round(total_ms, 0)}
    if timings.get("embed_ms") is not None:
        entry["embed_ms"] = round(timings["embed_ms"], 0)
    if timings.get("search_ms") is not None:
        entry["search_ms"] = round(timings["search_ms"], 0)
    if timings.get("rerank_ms") is not None:
        entry["rerank_ms"] = round(timings["rerank_ms"], 0)
    if llm_ms is not None:
        entry["llm_ms"] = round(llm_ms, 0)
    if intro_ms is not None:
        entry["intro_ms"] = round(intro_ms, 0)
    if extractor_ms is not None:
        entry["extractor_ms"] = round(extractor_ms, 0)
    if input_tokens is not None:
        entry["input_tokens"] = input_tokens
    if output_tokens is not None:
        entry["output_tokens"] = output_tokens
    return entry
