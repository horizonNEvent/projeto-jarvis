from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values


@dataclass(frozen=True)
class Config:
    minimax_api_key: str
    minimax_model: str = "MiniMax-Text-01"
    hotkey: str = "ctrl+alt+j"
    wake_word: str = "hey_jarvis"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    piper_voice: str = "pt_BR-faber-medium"
    piper_model_dir: str = "models/piper"
    max_history: int = 5
    log_tokens: bool = True
    vad_threshold: float = 0.5
    wake_threshold: float = 0.5
    silence_duration_ms: int = 1500


def _get(values: dict, key: str, default: str = "") -> str:
    """Return value from dotenv file dict, falling back to default."""
    file_val = values.get(key)
    if file_val is not None:
        return file_val
    return default


def load_config(env_path: str = ".env") -> Config:
    values = dotenv_values(env_path)

    api_key = _get(values, "MINIMAX_API_KEY").strip()
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is required in .env")

    return Config(
        minimax_api_key=api_key,
        minimax_model=_get(values, "MINIMAX_MODEL", "MiniMax-Text-01"),
        hotkey=_get(values, "HOTKEY", "ctrl+alt+j"),
        wake_word=_get(values, "WAKE_WORD", "hey_jarvis"),
        whisper_model=_get(values, "WHISPER_MODEL", "large-v3"),
        whisper_device=_get(values, "WHISPER_DEVICE", "cuda"),
        whisper_compute_type=_get(values, "WHISPER_COMPUTE_TYPE", "int8"),
        piper_voice=_get(values, "PIPER_VOICE", "pt_BR-faber-medium"),
        piper_model_dir=_get(values, "PIPER_MODEL_DIR", "models/piper"),
        max_history=int(_get(values, "MAX_HISTORY", "5")),
        log_tokens=_get(values, "LOG_TOKENS", "true").lower() == "true",
        vad_threshold=float(_get(values, "VAD_THRESHOLD", "0.5")),
        wake_threshold=float(_get(values, "WAKE_THRESHOLD", "0.5")),
        silence_duration_ms=int(_get(values, "SILENCE_DURATION_MS", "1500")),
    )
