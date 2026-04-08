from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

API_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"


def parse_sse_line(line: str) -> tuple[str | None, dict | None]:
    """Parse a single SSE line. Returns (content_delta, usage_dict)."""
    if not line.startswith("data: "):
        return None, None

    data = line[6:]
    if data == "[DONE]":
        return None, None

    chunk = json.loads(data)
    usage = extract_usage(chunk)

    choices = chunk.get("choices", [])
    if not choices:
        return None, usage

    delta = choices[0].get("delta", {})
    content = delta.get("content")
    return content or None, usage


def extract_usage(chunk: dict) -> dict | None:
    """Extract usage info from a chunk if present."""
    usage = chunk.get("usage")
    return usage if usage else None


async def stream_chat(
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "MiniMax-Text-01",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> AsyncIterator[tuple[str | None, dict | None]]:
    """Stream chat completions from Minimax API.

    Yields (content_delta, usage) tuples.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                content, usage = parse_sse_line(line)
                yield content, usage
