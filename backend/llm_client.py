"""
Shared OpenAI client helpers for chat, completions, and embeddings.
Used by chat_service, ask_service, embedder, and summarizer.
"""
from typing import Any, Iterator

from backend.config import settings


def _get_client() -> Any:
    """Return configured OpenAI client. Raises if OPENAI_API_KEY not set."""
    from openai import OpenAI

    api_key = settings.openai_api_key
    if not api_key or not api_key.strip():
        raise ValueError("OPENAI_API_KEY must be set for OpenAI API calls")
    kwargs = {"api_key": api_key.strip()}
    if getattr(settings, "openai_base_url", None) and str(settings.openai_base_url or "").strip():
        kwargs["base_url"] = str(settings.openai_base_url).strip()
    return OpenAI(**kwargs)


def openai_generate(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.35,
) -> tuple[str, int | None, int | None]:
    """
    Single completion. Returns (text, input_tokens, output_tokens).
    Uses a single user message; for system+user use openai_chat.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = ""
    if resp.choices:
        msg = resp.choices[0].message
        if msg and msg.content:
            text = msg.content
    inp = resp.usage.prompt_tokens if resp.usage else None
    out = resp.usage.completion_tokens if resp.usage else None
    return (text.strip(), inp, out)


def openai_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.35,
) -> tuple[str, int | None, int | None]:
    """
    Chat completion with messages (e.g. [{"role":"system","content":"..."},{"role":"user","content":"..."}]).
    Returns (content, input_tokens, output_tokens).
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = ""
    if resp.choices:
        msg = resp.choices[0].message
        if msg and msg.content:
            text = msg.content
    inp = resp.usage.prompt_tokens if resp.usage else None
    out = resp.usage.completion_tokens if resp.usage else None
    return (text.strip(), inp, out)


def openai_generate_stream(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> Iterator[str]:
    """Stream completion chunks (text deltas)."""
    client = _get_client()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def openai_embed(texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
    """
    Embed a list of texts. Returns list of embedding vectors.
    dimensions is supported for text-embedding-3-small; ignored for other models.
    """
    client = _get_client()
    kwargs: dict[str, Any] = {"model": model, "input": texts}
    if dimensions is not None and "embedding-3" in model:
        kwargs["dimensions"] = dimensions
    resp = client.embeddings.create(**kwargs)
    out: list[list[float]] = []
    # API returns in order; preserve it
    by_index = {d.index: d.embedding for d in resp.data}
    for i in range(len(texts)):
        out.append(by_index.get(i, []))
    return out
