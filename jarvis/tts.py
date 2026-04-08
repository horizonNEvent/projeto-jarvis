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
    """Synthesizes text to audio using Piper TTS standalone binary (CPU)."""

    def __init__(self, model_path: str, piper_dir: str = "models/piper", sample_rate: int = 22050) -> None:
        self._model_path = model_path
        self._piper_dir = Path(piper_dir)
        self.sample_rate = sample_rate

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        self._piper_exe = self._find_piper_exe()

    def _find_piper_exe(self) -> str:
        """Find piper executable — local binary or system PATH."""
        local_exe = self._piper_dir / "piper.exe"
        if local_exe.exists():
            return str(local_exe)
        # Fallback to system PATH
        return "piper"

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to int16 numpy array."""
        cmd = [
            self._piper_exe,
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
