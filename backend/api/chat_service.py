import logging
import time
from typing import Any, Callable

from fastapi import HTTPException, Response

from backend.api.request_logs import append_request_log
from backend.api.retrieval_utils import extract_doc_filename, rank_hits_by_file, truncate_snippet
from backend.api.schemas import ChatRequest, ChatResponse, RetrievedChunk
from backend.config import settings
from backend.ingestion.vector_store import hybrid_search, list_chunks

logger = logging.getLogger(__name__)


EmbedderGetter = Callable[[], Any]
VectorStoreGetter = Callable[[], Any]


def run_query_chat(
    req: ChatRequest,
    response: Response,
    get_embedder: EmbedderGetter,
    get_vector_store: VectorStoreGetter,
    chat_system_prompt: str,
) -> ChatResponse:
    """
    Run hybrid search, build context, and generate answer via OpenAI (if LLM configured).
    Falls back to a config hint when LLM is not configured.
    """
    t_total = time.perf_counter()
    embed_ms = 0.0
    search_ms = 0.0
    rerank_ms: float | None = None
    llm_ms = 0.0
    timings: dict[str, float] = {}
    snippet_max_lines = settings.chat_snippet_max_lines
    logger.info("Chat: %s", req.query[:60] + "..." if len(req.query) > 60 else req.query)
    if req.chunks:
        results = req.chunks
    else:
        client = get_vector_store()
        top_k = req.top_k if req.top_k is not None else settings.query_chat_final_k
        results = []

        doc_file = extract_doc_filename(req.query)
        if doc_file:
            t0 = time.perf_counter()
            chunks_list, _ = list_chunks(client, limit=200, file_path_contains=doc_file.lower())
            raw_chunks = [c for c in chunks_list if doc_file.lower() in (c.get("file_path") or "").lower()]
            if raw_chunks:
                raw_chunks.sort(key=lambda c: (c.get("start_line") or 0))
                results = [
                    RetrievedChunk(
                        id=c.get("id", ""),
                        score=0.0,
                        vector_score=None,
                        file_path=c.get("file_path", ""),
                        start_line=c.get("start_line", 0),
                        end_line=c.get("end_line", 0),
                        division=c.get("division"),
                        section_name=c.get("section_name"),
                        paragraph_name=c.get("paragraph_name"),
                        code_snippet=truncate_snippet(c.get("code_snippet", ""), snippet_max_lines),
                        language=c.get("language", "COBOL"),
                        source_type=c.get("source_type", "code"),
                    )
                    for c in raw_chunks[:top_k]
                ]
            search_ms = (time.perf_counter() - t0) * 1000

        if not results:
            embedder = get_embedder()
            t0 = time.perf_counter()
            try:
                vectors = embedder.embed_texts([req.query])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
            if not vectors or not vectors[0]:
                raise HTTPException(status_code=500, detail="No embedding returned")
            embed_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            use_reranker = settings.use_reranker or req.use_reranker
            hits = hybrid_search(
                client,
                vectors[0],
                req.query,
                top_k=settings.query_chat_top_k,
                final_k=top_k,
                score_threshold=settings.query_score_threshold,
                source_type=req.source_type if req.source_type != "all" else None,
                tags_filter=req.tags,
                use_reranker=use_reranker,
                out_timings=timings,
            )
            search_ms = (time.perf_counter() - t0) * 1000
            rerank_ms = timings.get("rerank_ms")
            hits = rank_hits_by_file(
                hits,
                settings.query_chat_max_chunks_per_file,
                top_k,
            )
            results = []
            for h in hits:
                p = h.get("payload") or {}
                results.append(
                    RetrievedChunk(
                        id=h.get("id", ""),
                        score=h.get("score", 0.0),
                        vector_score=h.get("vector_score"),
                        file_path=p.get("file_path", ""),
                        start_line=p.get("start_line", 0),
                        end_line=p.get("end_line", 0),
                        division=p.get("division"),
                        section_name=p.get("section_name"),
                        paragraph_name=p.get("paragraph_name"),
                        code_snippet=truncate_snippet(p.get("code_snippet", ""), snippet_max_lines),
                        language=p.get("language", "COBOL"),
                        source_type=p.get("source_type", "code"),
                    )
                )

    def _format_chunk(i: int, r: RetrievedChunk) -> str:
        meta_parts = [f"[{i}] {r.file_path} L{r.start_line}-{r.end_line}"]
        if r.paragraph_name:
            meta_parts.append(f"para:{r.paragraph_name}")
        if r.section_name and r.section_name != r.paragraph_name:
            meta_parts.append(f"section:{r.section_name}")
        if r.vector_score is not None:
            meta_parts.append(f"relevance:{r.vector_score:.2f}")
        snippet = r.code_snippet or ""
        return " | ".join(meta_parts) + "\n" + snippet

    context = "\n\n---\n\n".join(_format_chunk(i, r) for i, r in enumerate(results, 1))
    logger.info("Chat context: %d chunks, %d chars", len(results), len(context))
    req_id = str(int(time.time() * 1000))
    user_content = (
        "**Context** (numbered chunks; each has file path, line range, then code):\n\n"
        f"{context or '(No chunks retrieved.)'}\n\n"
        "---\n\n"
        f"**Question:** {req.query}\n\n"
        f"[Request {req_id}]\n\n"
        "**Answer:**"
    )

    if settings.llm_enabled and settings.llm_model and settings.openai_api_key:
        try:
            from backend.llm_client import openai_chat

            t0 = time.perf_counter()
            answer, input_tokens, output_tokens = openai_chat(
                settings.llm_model,
                [
                    {"role": "system", "content": chat_system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=settings.llm_max_output_tokens,
                temperature=0.35,
            )
            llm_ms = (time.perf_counter() - t0) * 1000
            if not answer:
                answer = "(No response from LLM)"
        except Exception as e:
            answer = f"(LLM error: {e})"
            input_tokens = None
            output_tokens = None
    else:
        answer = "(Enable LLM: set LLM_MODEL and LLM_ENABLED=true with OPENAI_API_KEY)"
        input_tokens = None
        output_tokens = None

    total_ms = (time.perf_counter() - t_total) * 1000
    logger.info(
        "chat_latency_ms=%.0f embed_ms=%.0f search_ms=%.0f rerank_ms=%s llm_ms=%.0f input_tokens=%s output_tokens=%s",
        total_ms,
        embed_ms,
        search_ms,
        f"{rerank_ms:.0f}" if rerank_ms is not None else "n/a",
        llm_ms,
        input_tokens if input_tokens is not None else "n/a",
        output_tokens if output_tokens is not None else "n/a",
    )
    append_request_log({
        "type": "chat",
        "total_ms": round(total_ms, 0),
        "embed_ms": round(embed_ms, 0),
        "search_ms": round(search_ms, 0),
        "rerank_ms": round(rerank_ms, 0) if rerank_ms is not None else None,
        "llm_ms": round(llm_ms, 0),
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
    })
    response.headers["X-Request-Duration-Ms"] = str(int(round(total_ms)))
    return ChatResponse(query=req.query, answer=answer, results=results)
