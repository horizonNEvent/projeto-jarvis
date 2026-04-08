from __future__ import annotations

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel


class WakeWordDetector:
    """Listens for 'hey jarvis' wake word using openwakeword."""

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1280  # 80ms at 16kHz

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model_name = model_name
        self._model = OWWModel(wakeword_models=[model_name])

    def listen_blocking(self) -> bool:
        """Block until wake word is detected. Returns True."""
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK_SIZE,
        ) as stream:
            while True:
                chunk, _ = stream.read(self.CHUNK_SIZE)
                chunk_1d = chunk[:, 0]

                prediction = self._model.predict(chunk_1d)

                for ww, score in prediction.items():
                    if score > self._threshold:
                        self._model.reset()
                        return True

    def reset(self) -> None:
        self._model.reset()
