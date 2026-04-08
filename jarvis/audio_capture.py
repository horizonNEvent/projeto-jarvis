from __future__ import annotations

import numpy as np
import sounddevice as sd
import torch


class AudioCapture:
    """Captures audio from microphone with VAD-based endpoint detection."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    VAD_CHUNK = 512  # 32ms at 16kHz

    def __init__(self, vad_threshold: float = 0.5, silence_duration_ms: int = 1500, max_record_s: int = 30) -> None:
        self._vad_threshold = vad_threshold
        self._silence_samples = int(silence_duration_ms * self.SAMPLE_RATE / 1000)
        self._max_samples = max_record_s * self.SAMPLE_RATE
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    def record_until_silence(self) -> np.ndarray | None:
        """Record audio until silence is detected. Returns int16 numpy array or None."""
        frames: list[np.ndarray] = []
        silence_count = 0
        total_samples = 0
        speech_detected = False

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="int16",
            blocksize=self.VAD_CHUNK,
        ) as stream:
            while True:
                chunk, _ = stream.read(self.VAD_CHUNK)
                chunk_1d = chunk[:, 0]
                frames.append(chunk_1d.copy())
                total_samples += len(chunk_1d)

                audio_float = torch.from_numpy(chunk_1d.astype(np.float32) / 32768.0)
                confidence = self._vad_model(audio_float, self.SAMPLE_RATE).item()

                if confidence >= self._vad_threshold:
                    speech_detected = True
                    silence_count = 0
                else:
                    silence_count += len(chunk_1d)

                if speech_detected and silence_count >= self._silence_samples:
                    break
                if total_samples >= self._max_samples:
                    break

        self._vad_model.reset_states()

        if not speech_detected:
            return None

        return np.concatenate(frames)

    @staticmethod
    def get_input_devices() -> list[dict]:
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
