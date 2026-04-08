from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TokenLogger:
    def __init__(self, log_path: str = "logs/token_usage.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.cumulative = self._load_cumulative()

    def _load_cumulative(self) -> int:
        if not self._path.exists():
            return 0
        last_line = ""
        for line in self._path.read_text().strip().split("\n"):
            if line.strip():
                last_line = line
        if not last_line:
            return 0
        return json.loads(last_line).get("cumulative", 0)

    def log(self, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        self.cumulative += total
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total": total,
            "cumulative": self.cumulative,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
