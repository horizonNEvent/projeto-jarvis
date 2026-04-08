from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd


class AudioPlayer:
    """Plays audio chunks sequentially from a queue. Thread-safe."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._playing = False

    def start(self) -> None:
        self._playing = True
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._playing = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, audio: np.ndarray) -> None:
        self._queue.put(audio)

    def finish(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join()

    def _playback_loop(self) -> None:
        while self._playing:
            chunk = self._queue.get()
            if chunk is None:
                break
            sd.play(chunk, samplerate=self.sample_rate)
            sd.wait()
