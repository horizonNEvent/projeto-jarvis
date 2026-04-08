import json
from pathlib import Path

from jarvis.token_logger import TokenLogger


def test_log_usage_creates_file_and_writes(tmp_path):
    log_file = tmp_path / "token_usage.jsonl"
    logger = TokenLogger(str(log_file))

    logger.log(input_tokens=100, output_tokens=50)

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["input_tokens"] == 100
    assert entry["output_tokens"] == 50
    assert entry["total"] == 150
    assert entry["cumulative"] == 150
    assert "timestamp" in entry


def test_cumulative_tracking(tmp_path):
    log_file = tmp_path / "token_usage.jsonl"
    logger = TokenLogger(str(log_file))

    logger.log(input_tokens=100, output_tokens=50)
    logger.log(input_tokens=200, output_tokens=80)

    lines = log_file.read_text().strip().split("\n")
    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])
    assert entry1["cumulative"] == 150
    assert entry2["cumulative"] == 430


def test_loads_existing_cumulative(tmp_path):
    log_file = tmp_path / "token_usage.jsonl"
    log_file.write_text(
        json.dumps({"input_tokens": 500, "output_tokens": 200, "total": 700, "cumulative": 700, "timestamp": "x"})
        + "\n"
    )

    logger = TokenLogger(str(log_file))
    assert logger.cumulative == 700

    logger.log(input_tokens=50, output_tokens=30)
    assert logger.cumulative == 780
