from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import httpx

API_URL = "https://api.minimax.io/v1/chat/completions"


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


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


async def stream_chat(
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "MiniMax-M2.7",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> AsyncIterator[tuple[str | None, dict | None]]:
    """Stream chat completions from Minimax API.

    Yields (content_delta, usage) tuples. Filters out <think> blocks.
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

    in_think_block = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                content, usage = parse_sse_line(line)

                if content:
                    # Filter out <think>...</think> blocks in streaming
                    if "<think>" in content:
                        in_think_block = True
                    if in_think_block:
                        if "</think>" in content:
                            in_think_block = False
                            # Keep any text after </think>
                            after = content.split("</think>", 1)[1].strip()
                            if after:
                                yield after, usage
                        continue
                    yield content, usage
                elif usage:
                    yield None, usage
