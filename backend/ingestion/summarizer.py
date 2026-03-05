"""
Async functional summary generation for hierarchical ("summary-first") indexing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)


class FunctionalSummaryGenerator:
    """
    Create short compiler-focused functional summaries for child docs.

    The summary is embedded; raw code remains in payload/code_snippet.
    """

    def __init__(self) -> None:
        self.enabled = bool(settings.summary_generation_enabled)
        self.max_chars = max(500, int(settings.summary_input_max_chars))
        self.max_concurrency = max(1, int(settings.summary_max_concurrency))
        self.timeout_sec = max(1.0, float(settings.summary_timeout_sec))
        self.model = settings.summary_model

    async def summarize_payloads(self, payloads: list[dict[str, Any]]) -> list[str]:
        if not payloads:
            return []
        if not self.enabled:
            return [self._heuristic_summary(p) for p in payloads]

        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._summarize_one(p, sem) for p in payloads]
        return await asyncio.gather(*tasks)

    async def _summarize_one(self, payload: dict[str, Any], sem: asyncio.Semaphore) -> str:
        async with sem:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._summarize_one_sync, payload),
                    timeout=self.timeout_sec,
                )
            except Exception as exc:
                logger.debug("Summary generation fallback (%s): %s", payload.get("file_path"), exc)
                return self._heuristic_summary(payload)

    def _summarize_one_sync(self, payload: dict[str, Any]) -> str:
        snippet = (payload.get("code_snippet") or "")[: self.max_chars]
        if not snippet.strip():
            return self._heuristic_summary(payload)
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except Exception:
            return self._heuristic_summary(payload)

        if not settings.google_cloud_project:
            return self._heuristic_summary(payload)

        client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            http_options=HttpOptions(api_version="v1"),
        )

        prompt = self._build_prompt(payload, snippet)
        resp = client.models.generate_content(model=self.model, contents=prompt)
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return self._heuristic_summary(payload)
        return self._normalize_summary(text)

    def _build_prompt(self, payload: dict[str, Any], snippet: str) -> str:
        return (
            "You summarize compiler implementation blocks for retrieval.\n"
            "Return EXACTLY 3 sentences and keep it concise.\n"
            "Sentence 1: what COBOL syntax/semantic concern this block handles.\n"
            "Sentence 2: pipeline IO role (what input it receives, what output/artifact it produces).\n"
            "Sentence 3: operational impact in the compiler phase.\n"
            "Avoid markdown, bullets, and speculation.\n\n"
            f"Phase: {payload.get('phase') or 'unknown'}\n"
            f"Language: {payload.get('language') or 'unknown'}\n"
            f"File: {payload.get('file_path') or ''}\n"
            f"Function: {payload.get('function_name') or payload.get('paragraph_name') or 'unknown'}\n"
            f"Dependencies: {', '.join(payload.get('include_headers') or []) or 'none'}\n\n"
            "Code block:\n"
            f"{snippet}"
        )

    def _heuristic_summary(self, payload: dict[str, Any]) -> str:
        phase = payload.get("phase") or "unknown"
        fn = payload.get("function_name") or payload.get("paragraph_name") or "this block"
        deps = payload.get("include_headers") or []
        dep_text = ", ".join(deps[:3]) if deps else "no direct local headers"
        ext = (payload.get("file_ext") or "").lower()
        syntax_focus = "COBOL statements"
        if ext in {"c", "h", "cpp", "hpp"}:
            syntax_focus = "COBOL runtime/compiler support logic"
        return (
            f"This block ({fn}) likely handles {syntax_focus} in the {phase} phase. "
            f"It consumes intermediate compiler state and emits transformed state or artifacts used by downstream stages. "
            f"It coordinates with {dep_text} to preserve phase-specific behavior."
        )

    @staticmethod
    def _normalize_summary(text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        # Keep only first three sentence-like segments.
        parts = re.split(r"(?<=[.!?])\s+", clean)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 3:
            return " ".join(parts[:3])
        # Ensure exactly three sentences for consistency.
        if len(parts) == 2:
            return f"{parts[0]} {parts[1]} This behavior is relevant to compiler execution."
        if len(parts) == 1:
            return f"{parts[0]} It defines how data flows through the compiler pipeline. It is important for phase-correct code generation."
        return (
            "This block handles COBOL-related compiler behavior. "
            "It transforms or routes data through the compiler pipeline. "
            "It is relevant to phase-specific execution."
        )
