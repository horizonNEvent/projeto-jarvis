from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    """Transcribes audio using faster-whisper on GPU."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe int16 audio array to text."""
        audio_float = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_float,
            language="pt",
            beam_size=3,
            vad_filter=True,
        )

        text_parts = [segment.text.strip() for segment in segments]
        return " ".join(text_parts).strip()
