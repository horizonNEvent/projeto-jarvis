from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class SentenceSplitter:
    """Accumulates characters and yields complete sentences on . ! ?"""

    _ABBREVIATIONS = {"sr", "sra", "dr", "dra", "prof", "vs", "etc", "ex"}

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def feed(self, char: str) -> str | None:
        self._buffer.append(char)
        if char in ".!?":
            text = "".join(self._buffer).strip()
            if char == "." and self._is_abbreviation(text):
                return None
            self._buffer.clear()
            return text if text else None
        return None

    def flush(self) -> list[str]:
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return [text] if text else []

    def _is_abbreviation(self, text: str) -> bool:
        words = text.rstrip(".").rsplit(None, 1)
        if not words:
            return False
        last_word = words[-1].lower().rstrip(".")
        return last_word in self._ABBREVIATIONS


class PiperTTS:
    """Synthesizes text to audio using Piper TTS (CPU)."""

    def __init__(self, model_path: str, sample_rate: int = 22050) -> None:
        self._model_path = model_path
        self.sample_rate = sample_rate

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to int16 numpy array."""
        cmd = [
            "piper",
            "--model", self._model_path,
            "--output-raw",
        ]
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        return np.frombuffer(result.stdout, dtype=np.int16)
