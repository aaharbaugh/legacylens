import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from backend.api.request_logs import append_request_log, ask_log_entry
from backend.api.schemas import AskRequest, AskResponse, RetrievedChunk
from backend.config import settings

logger = logging.getLogger(__name__)

ResolveAskChunksFn = Callable[[str, str | None, dict[str, float] | None], list[RetrievedChunk]]


def _usage_from_resp(resp: Any) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from Gemini generate_content response."""
    if not resp or not getattr(resp, "usage_metadata", None):
        return None, None
    um = resp.usage_metadata
    inp = getattr(um, "prompt_token_count", None) or getattr(um, "promptTokenCount", None)
    out = getattr(um, "candidates_token_count", None) or getattr(um, "candidatesTokenCount", None)
    try:
        return (int(inp) if inp is not None else None, int(out) if out is not None else None)
    except (TypeError, ValueError):
        return None, None


async def run_api_ask(req: AskRequest, resolve_ask_chunks: ResolveAskChunksFn) -> AskResponse:
    t_start = time.perf_counter()
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    folder = (req.folder or "").strip() or None
    timings: dict[str, float] = {}

    try:
        chunks = await asyncio.to_thread(resolve_ask_chunks, query, folder, timings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not chunks:
        total_ms = (time.perf_counter() - t_start) * 1000
        append_request_log(ask_log_entry("ask", total_ms, timings))
        return AskResponse(
            intro="No relevant context found for this query.",
            code_snippet="",
            technical_explanation="Try a more specific query or different keywords.",
            results=[],
        )

    context = "\n\n---\n\n".join(
        f"[{i}] {r.file_path} L{r.start_line}-{r.end_line}\n{r.code_snippet}"
        for i, r in enumerate(chunks, 1)
    )
    max_words = getattr(settings, "extractor_explanation_max_words", 50) or 50
    intro_model = getattr(settings, "llm_intro_model", None) or settings.llm_model
    extractor_model = getattr(settings, "llm_extractor_model", None) or settings.llm_model

    def _run_intro_sync() -> tuple[str, float, int | None, int | None]:
        if not settings.google_cloud_project or not intro_model:
            return "Looking that up.", 0.0, None, None
        t0 = time.perf_counter()
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig, HttpOptions
            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                http_options=HttpOptions(api_version="v1"),
            )
            intro_words = max(10, getattr(settings, "ask_intro_max_words", 15) or 15)
            intro_tokens = max(32, getattr(settings, "ask_intro_max_tokens", 50) or 50)
            prompt = (
                f"In {intro_words} words or fewer, one short sentence: paraphrase this question or state the user's intent "
                f"as if you're about to answer it. No code, no context—just the intent.\n\nQuestion: {query}"
            )
            resp = client.models.generate_content(
                model=intro_model,
                contents=prompt,
                config=GenerateContentConfig(max_output_tokens=intro_tokens, temperature=0.3),
            )
            ms = (time.perf_counter() - t0) * 1000
            inp, out = _usage_from_resp(resp)
            return (resp.text or "").strip() or "Looking that up.", ms, inp, out
        except Exception as e:
            logger.warning("Intro LLM failed: %s", e)
            return "Looking that up.", (time.perf_counter() - t0) * 1000, None, None

    def _run_extractor_sync() -> tuple[str, str, float, int | None, int | None]:
        default_snippet = chunks[0].code_snippet[:2000] if chunks else ""
        default_expl = "No concise answer could be extracted from the retrieved context."
        if not settings.google_cloud_project or not extractor_model:
            return default_snippet, default_expl, 0.0, None, None
        t0 = time.perf_counter()
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig, HttpOptions
            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                http_options=HttpOptions(api_version="v1"),
            )
            ext_ctx = getattr(settings, "ask_extractor_context_chars", 4000) or 4000
            ext_tokens = max(256, getattr(settings, "ask_extractor_max_tokens", 384) or 384)
            prompt = (
                "You must answer the user's question directly. Output a single JSON object with exactly two keys:\n"
                '- "technical_explanation": In a few sentences, directly answer the question. Do NOT say "see context above" or "see above". '
                f"Use the context to give a concrete, specific answer (max {max_words} words).\n"
                '- "code_snippet": The single most relevant short code excerpt (max 15-20 lines). If none is needed for the answer, use "".\n\n'
                f"Context:\n{context[:ext_ctx]}\n\nQuestion: {query}\n\nJSON:"
            )
            resp = client.models.generate_content(
                model=extractor_model,
                contents=prompt,
                config=GenerateContentConfig(max_output_tokens=ext_tokens, temperature=0.2),
            )
            ms = (time.perf_counter() - t0) * 1000
            inp, out = _usage_from_resp(resp)
            text = (resp.text or "").strip()
            for start in ("```json", "```"):
                if start in text:
                    text = text.split(start, 1)[-1].replace("```", "").strip()
            data = json.loads(text)
            expl = str(data.get("technical_explanation", ""))[:500].strip()
            if not expl or "see context above" in expl.lower() or "see above" in expl.lower():
                expl = default_expl
            snippet = str(data.get("code_snippet", ""))[:2000].strip() or default_snippet
            if snippet.count("\n") > 25:
                snippet = "\n".join(snippet.split("\n")[:25]) + "\n\n…"
            return (snippet, expl, ms, inp, out)
        except Exception as e:
            logger.warning("Extractor LLM failed: %s", e)
            return default_snippet, default_expl, (time.perf_counter() - t0) * 1000, None, None

    t_llm_start = time.perf_counter()
    (intro_text, intro_ms, intro_in, intro_out), (code_snippet, technical_explanation, extractor_ms, ext_in, ext_out) = await asyncio.gather(
        asyncio.to_thread(_run_intro_sync),
        asyncio.to_thread(_run_extractor_sync),
    )
    llm_ms = (time.perf_counter() - t_llm_start) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000
    in_tok = (intro_in or 0) + (ext_in or 0) or None
    out_tok = (intro_out or 0) + (ext_out or 0) or None
    append_request_log(ask_log_entry("ask", total_ms, timings, llm_ms, intro_ms=intro_ms, extractor_ms=extractor_ms, input_tokens=in_tok, output_tokens=out_tok))
    return AskResponse(
        intro=intro_text,
        code_snippet=code_snippet,
        technical_explanation=technical_explanation,
        results=chunks,
    )


async def run_api_ask_stream(req: AskRequest, resolve_ask_chunks: ResolveAskChunksFn) -> StreamingResponse:
    t_start = time.perf_counter()
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    folder = (req.folder or "").strip() or None
    timings: dict[str, float] = {}

    chunks: list[RetrievedChunk] = []
    try:
        chunks = await asyncio.to_thread(resolve_ask_chunks, query, folder, timings)
    except Exception as e:
        logger.warning("Ask stream resolve chunks failed: %s", e)

    def _sse_event(name: str, data: Any) -> bytes:
        payload = json.dumps(data)
        return f"event: {name}\ndata: {payload}\n\n".encode("utf-8")

    if not chunks:
        total_ms = (time.perf_counter() - t_start) * 1000
        append_request_log(ask_log_entry("ask_stream", total_ms, timings))

        async def _empty_stream():
            yield _sse_event("intro", "No relevant context found for this query.")
            yield _sse_event("result", {"code_snippet": "", "technical_explanation": "Try a more specific query.", "results": []})
            yield _sse_event("done", {"ok": True})

        return StreamingResponse(
            _empty_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    context = "\n\n---\n\n".join(
        f"[{i}] {r.file_path} L{r.start_line}-{r.end_line}\n{r.code_snippet}"
        for i, r in enumerate(chunks, 1)
    )
    max_words = getattr(settings, "extractor_explanation_max_words", 50) or 50
    intro_model = getattr(settings, "llm_intro_model", None) or settings.llm_model
    extractor_model = getattr(settings, "llm_extractor_model", None) or settings.llm_model

    def _stream_intro_worker(chunk_queue: queue.Queue) -> None:
        if not settings.google_cloud_project or not intro_model:
            chunk_queue.put(None)
            return
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig, HttpOptions
            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                http_options=HttpOptions(api_version="v1"),
            )
            intro_words = max(10, getattr(settings, "ask_intro_max_words", 15) or 15)
            intro_tokens = max(32, getattr(settings, "ask_intro_max_tokens", 50) or 50)
            prompt = (
                f"In {intro_words} words or fewer, one short sentence: paraphrase this question or state the user's intent "
                f"as if you're about to answer it. No code, no context—just the intent.\n\nQuestion: {query}"
            )
            config = GenerateContentConfig(max_output_tokens=intro_tokens, temperature=0.3)
            stream = client.models.generate_content_stream(
                model=intro_model,
                contents=prompt,
                config=config,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None) or ""
                if text:
                    chunk_queue.put(text)
        except Exception as e:
            logger.warning("Intro LLM stream failed: %s", e)
        finally:
            chunk_queue.put(None)

    def _run_extractor_sync() -> tuple[str, str, float, int | None, int | None]:
        default_snippet = chunks[0].code_snippet[:2000] if chunks else ""
        default_expl = "No concise answer could be extracted from the retrieved context."
        if not settings.google_cloud_project or not extractor_model:
            return default_snippet, default_expl, 0.0, None, None
        t0 = time.perf_counter()
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig, HttpOptions
            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                http_options=HttpOptions(api_version="v1"),
            )
            ext_ctx = getattr(settings, "ask_extractor_context_chars", 4000) or 4000
            ext_tokens = max(256, getattr(settings, "ask_extractor_max_tokens", 384) or 384)
            prompt = (
                "You must answer the user's question directly. Output a single JSON object with exactly two keys:\n"
                '- "technical_explanation": In 1-3 sentences, directly answer the question. Do NOT say "see context above" or "see above". '
                f"Use the context to give a concrete, specific answer (max {max_words} words).\n"
                '- "code_snippet": The single most relevant short code excerpt (max 15-20 lines). If none is needed for the answer, use "".\n\n'
                f"Context:\n{context[:ext_ctx]}\n\nQuestion: {query}\n\nJSON:"
            )
            resp = client.models.generate_content(
                model=extractor_model,
                contents=prompt,
                config=GenerateContentConfig(max_output_tokens=ext_tokens, temperature=0.2),
            )
            ms = (time.perf_counter() - t0) * 1000
            inp, out = _usage_from_resp(resp)
            text = (resp.text or "").strip()
            for start in ("```json", "```"):
                if start in text:
                    text = text.split(start, 1)[-1].replace("```", "").strip()
            data = json.loads(text)
            expl = str(data.get("technical_explanation", ""))[:500].strip()
            if not expl or "see context above" in expl.lower() or "see above" in expl.lower():
                expl = default_expl
            snippet = str(data.get("code_snippet", ""))[:2000].strip() or default_snippet
            if snippet.count("\n") > 25:
                snippet = "\n".join(snippet.split("\n")[:25]) + "\n\n…"
            return (snippet, expl, ms, inp, out)
        except Exception as e:
            logger.warning("Extractor LLM failed: %s", e)
            return default_snippet, default_expl, (time.perf_counter() - t0) * 1000, None, None

    async def _stream():
        t_llm_start = time.perf_counter()
        intro_start = None
        chunk_queue: queue.Queue = queue.Queue()
        intro_thread = threading.Thread(target=_stream_intro_worker, args=(chunk_queue,))
        intro_thread.start()
        extractor_task = asyncio.create_task(asyncio.to_thread(_run_extractor_sync))
        loop = asyncio.get_event_loop()
        intro_parts: list[str] = []
        while True:
            chunk = await loop.run_in_executor(None, chunk_queue.get)
            if chunk is None:
                break
            if intro_start is None:
                intro_start = time.perf_counter()
            intro_parts.append(chunk)
            yield _sse_event("intro_chunk", chunk)
        intro_thread.join(timeout=1.0)
        full_intro = ("".join(intro_parts).strip() or "Looking that up.") if intro_parts else "Looking that up."
        intro_ms = (time.perf_counter() - intro_start) * 1000 if intro_start else 0.0
        yield _sse_event("intro_done", full_intro)
        code_snippet, technical_explanation, extractor_ms, ext_in, ext_out = await extractor_task
        llm_ms = (time.perf_counter() - t_llm_start) * 1000
        in_tok = ext_in
        out_tok = ext_out
        results_payload = [
            {
                "id": r.id,
                "score": r.score,
                "vector_score": r.vector_score,
                "file_path": r.file_path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "division": r.division,
                "section_name": r.section_name,
                "paragraph_name": r.paragraph_name,
                "code_snippet": r.code_snippet,
                "language": r.language,
                "source_type": r.source_type,
            }
            for r in chunks
        ]
        total_ms = (time.perf_counter() - t_start) * 1000
        append_request_log(
            ask_log_entry(
                "ask_stream",
                total_ms,
                timings,
                llm_ms,
                intro_ms=intro_ms,
                extractor_ms=extractor_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        )
        yield _sse_event("result", {"code_snippet": code_snippet, "technical_explanation": technical_explanation, "results": results_payload})
        yield _sse_event("done", {"ok": True})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
