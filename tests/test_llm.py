import json
import pytest

from jarvis.llm import parse_sse_line, extract_usage


def test_parse_sse_line_with_content():
    data = json.dumps({
        "choices": [{"delta": {"content": "Ola"}, "finish_reason": None}]
    })
    line = f"data: {data}"
    assert parse_sse_line(line) == ("Ola", None)


def test_parse_sse_line_done():
    assert parse_sse_line("data: [DONE]") == (None, None)


def test_parse_sse_line_empty():
    assert parse_sse_line("") == (None, None)
    assert parse_sse_line("event: ping") == (None, None)


def test_parse_sse_line_with_finish_reason():
    data = json.dumps({
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    })
    line = f"data: {data}"
    content, usage = parse_sse_line(line)
    assert content is None
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


def test_extract_usage():
    chunk = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    assert extract_usage(chunk) == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_extract_usage_missing():
    assert extract_usage({}) is None
