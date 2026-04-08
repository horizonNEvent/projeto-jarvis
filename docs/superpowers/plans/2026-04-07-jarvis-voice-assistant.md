# Jarvis Voice Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid voice assistant that listens via wake word or hotkey, transcribes speech locally, gets responses from Minimax API with streaming, and speaks them back using Piper TTS.

**Architecture:** Pipeline with streaming — audio capture feeds into local STT (faster-whisper GPU), text goes to Minimax API via SSE streaming, response chunks are synthesized sentence-by-sentence with Piper TTS (CPU) and played back incrementally. Wake word (openwakeword) and hotkey (pynput) activate the pipeline.

**Tech Stack:** Python 3.12, faster-whisper (CTranslate2/CUDA), Piper TTS (ONNX/CPU), openwakeword, silero-vad (PyTorch CPU), httpx (async HTTP), sounddevice, pynput, python-dotenv.

---

## File Structure

```
projeto-jarvis/
├── jarvis/
│   ├── __init__.py           # Package init
│   ├── config.py             # Load .env, expose typed settings
│   ├── audio_capture.py      # Mic capture + silero-vad recording
│   ├── wake_word.py          # openwakeword listener
│   ├── hotkey.py             # pynput hotkey listener
│   ├── stt.py                # faster-whisper transcription
│   ├── llm.py                # Minimax API streaming client
│   ├── conversation.py       # Sliding window history
│   ├── tts.py                # Piper TTS synthesis + sentence splitting
│   ├── audio_player.py       # sounddevice playback queue
│   ├── token_logger.py       # Token usage logging
│   └── main.py               # Async orchestrator loop
├── tests/
│   ├── test_config.py
│   ├── test_conversation.py
│   ├── test_tts_splitter.py
│   ├── test_token_logger.py
│   └── test_llm.py
├── models/                   # Downloaded model files (gitignored)
│   └── piper/
├── logs/                     # Runtime logs (gitignored)
├── .env.example              # Template for .env
├── .env                      # Actual secrets (gitignored)
├── .gitignore
├── requirements.txt
├── setup_models.py           # Downloads required models
└── start_jarvis.bat          # Windows startup script
```

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `jarvis/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd D:/Workspace/projeto-jarvis
git init
```

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

```
__pycache__/
*.pyc
.env
models/
logs/
*.egg-info/
dist/
build/
.venv/
venv/
```

- [ ] **Step 3: Create requirements.txt**

Create `requirements.txt`:

```
faster-whisper==1.1.0
piper-tts==1.2.0
openwakeword==0.6.0
silero-vad==5.1.2
httpx==0.28.1
sounddevice==0.5.1
numpy==1.26.4
pynput==1.7.7
python-dotenv==1.0.1
```

- [ ] **Step 4: Create .env.example**

Create `.env.example`:

```env
MINIMAX_API_KEY=sk-cp-your-key-here
MINIMAX_MODEL=MiniMax-Text-01
HOTKEY=ctrl+alt+j
WAKE_WORD=hey_jarvis
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8
PIPER_VOICE=pt_BR-faber-medium
PIPER_MODEL_DIR=models/piper
MAX_HISTORY=5
LOG_TOKENS=true
VAD_THRESHOLD=0.5
WAKE_THRESHOLD=0.5
SILENCE_DURATION_MS=1500
```

- [ ] **Step 5: Create .env from example with real key**

Copy `.env.example` to `.env` and fill in `MINIMAX_API_KEY` with the real key.

- [ ] **Step 6: Create virtual environment and install dependencies**

```bash
python -m venv .venv
.venv/Scripts/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Note: torch with CUDA 12.1 is needed for silero-vad. faster-whisper uses its own CTranslate2 CUDA backend, not PyTorch.

- [ ] **Step 7: Create jarvis/__init__.py**

Create `jarvis/__init__.py`:

```python
"""Jarvis — Assistente de voz hibrido."""
```

- [ ] **Step 8: Create directories**

```bash
mkdir -p models/piper logs tests
```

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt .env.example jarvis/__init__.py
git commit -m "chore: project scaffolding with dependencies and structure"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `jarvis/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import os
import pytest


def test_config_loads_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIMAX_API_KEY=sk-test-123\n"
        "MINIMAX_MODEL=MiniMax-Text-01\n"
        "MAX_HISTORY=3\n"
        "LOG_TOKENS=false\n"
        "SILENCE_DURATION_MS=2000\n"
    )
    monkeypatch.chdir(tmp_path)

    from jarvis.config import load_config

    cfg = load_config(str(env_file))
    assert cfg.minimax_api_key == "sk-test-123"
    assert cfg.minimax_model == "MiniMax-Text-01"
    assert cfg.max_history == 3
    assert cfg.log_tokens is False
    assert cfg.silence_duration_ms == 2000


def test_config_defaults(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MINIMAX_API_KEY=sk-test\n")
    monkeypatch.chdir(tmp_path)

    from jarvis.config import load_config

    cfg = load_config(str(env_file))
    assert cfg.max_history == 5
    assert cfg.log_tokens is True
    assert cfg.whisper_model == "large-v3"
    assert cfg.whisper_device == "cuda"
    assert cfg.silence_duration_ms == 1500


def test_config_missing_api_key(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    monkeypatch.chdir(tmp_path)

    from jarvis.config import load_config

    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        load_config(str(env_file))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ImportError: cannot import name 'load_config' from 'jarvis.config'`

- [ ] **Step 3: Write the implementation**

Create `jarvis/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


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


def load_config(env_path: str = ".env") -> Config:
    load_dotenv(env_path, override=True)

    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is required in .env")

    return Config(
        minimax_api_key=api_key,
        minimax_model=os.getenv("MINIMAX_MODEL", "MiniMax-Text-01"),
        hotkey=os.getenv("HOTKEY", "ctrl+alt+j"),
        wake_word=os.getenv("WAKE_WORD", "hey_jarvis"),
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        piper_voice=os.getenv("PIPER_VOICE", "pt_BR-faber-medium"),
        piper_model_dir=os.getenv("PIPER_MODEL_DIR", "models/piper"),
        max_history=int(os.getenv("MAX_HISTORY", "5")),
        log_tokens=os.getenv("LOG_TOKENS", "true").lower() == "true",
        vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
        wake_threshold=float(os.getenv("WAKE_THRESHOLD", "0.5")),
        silence_duration_ms=int(os.getenv("SILENCE_DURATION_MS", "1500")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/config.py tests/test_config.py
git commit -m "feat: add config module with .env loading and defaults"
```

---

### Task 3: Conversation History Manager

**Files:**
- Create: `jarvis/conversation.py`
- Create: `tests/test_conversation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_conversation.py`:

```python
from jarvis.conversation import Conversation

SYSTEM_PROMPT = "Voce e Jarvis."


def test_initial_messages_has_system_only():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    msgs = conv.get_messages()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_add_turn_and_retrieve():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("Ola", "Ola, como posso ajudar?")
    msgs = conv.get_messages()
    assert len(msgs) == 3
    assert msgs[1] == {"role": "user", "content": "Ola"}
    assert msgs[2] == {"role": "assistant", "content": "Ola, como posso ajudar?"}


def test_sliding_window_evicts_oldest():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=2)
    conv.add_turn("msg1", "resp1")
    conv.add_turn("msg2", "resp2")
    conv.add_turn("msg3", "resp3")
    msgs = conv.get_messages()
    # system + 2 turns (msg2, msg3) = 5 messages
    assert len(msgs) == 5
    assert msgs[1]["content"] == "msg2"
    assert msgs[3]["content"] == "msg3"


def test_get_messages_for_api_includes_new_user_msg():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("old question", "old answer")
    msgs = conv.get_messages_for_api("new question")
    assert len(msgs) == 4
    assert msgs[-1] == {"role": "user", "content": "new question"}


def test_clear_resets_to_system_only():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("a", "b")
    conv.clear()
    assert len(conv.get_messages()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_conversation.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `jarvis/conversation.py`:

```python
from __future__ import annotations


class Conversation:
    def __init__(self, system_prompt: str, max_turns: int = 5) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._turns.append((user_msg, assistant_msg))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_messages(self) -> list[dict[str, str]]:
        msgs = [self._system.copy()]
        for user_msg, assistant_msg in self._turns:
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": assistant_msg})
        return msgs

    def get_messages_for_api(self, new_user_msg: str) -> list[dict[str, str]]:
        msgs = self.get_messages()
        msgs.append({"role": "user", "content": new_user_msg})
        return msgs

    def clear(self) -> None:
        self._turns.clear()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_conversation.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/conversation.py tests/test_conversation.py
git commit -m "feat: add conversation history with sliding window"
```

---

### Task 4: Token Usage Logger

**Files:**
- Create: `jarvis/token_logger.py`
- Create: `tests/test_token_logger.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_token_logger.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_token_logger.py -v
```

Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `jarvis/token_logger.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_token_logger.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/token_logger.py tests/test_token_logger.py
git commit -m "feat: add token usage logger with cumulative tracking"
```

---

### Task 5: TTS Sentence Splitter + Piper Integration

**Files:**
- Create: `jarvis/tts.py`
- Create: `tests/test_tts_splitter.py`

- [ ] **Step 1: Write the failing test for sentence splitting**

Create `tests/test_tts_splitter.py`:

```python
from jarvis.tts import SentenceSplitter


def test_splits_on_period():
    sp = SentenceSplitter()
    results = []
    for char in "Ola mundo. Como vai?":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["Ola mundo.", "Como vai?"]


def test_splits_on_exclamation_and_question():
    sp = SentenceSplitter()
    results = []
    for char in "Sim! Nao? Talvez.":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["Sim!", "Nao?", "Talvez."]


def test_flush_returns_incomplete():
    sp = SentenceSplitter()
    for char in "Sem ponto final":
        sp.feed(char)
    remaining = sp.flush()
    assert remaining == ["Sem ponto final"]


def test_flush_empty_returns_nothing():
    sp = SentenceSplitter()
    assert sp.flush() == []


def test_ignores_abbreviation_dots():
    sp = SentenceSplitter()
    results = []
    # Dots after single letter (e.g., "Dr.") should not split
    for char in "O Sr. Silva chegou. Boa tarde.":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["O Sr. Silva chegou.", "Boa tarde."]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_tts_splitter.py -v
```

Expected: FAIL

- [ ] **Step 3: Write the sentence splitter**

Create `jarvis/tts.py` (first part — splitter only):

```python
from __future__ import annotations

import io
import subprocess
import wave
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
            # Check if dot is after a short abbreviation (e.g. "Sr.")
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
        # Check if the last word before the dot is a known abbreviation
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_tts_splitter.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/tts.py tests/test_tts_splitter.py
git commit -m "feat: add TTS module with sentence splitter and Piper integration"
```

---

### Task 6: Minimax API Streaming Client

**Files:**
- Create: `jarvis/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `jarvis/llm.py`:

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

API_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"


def parse_sse_line(line: str) -> tuple[str | None, dict | None]:
    """Parse a single SSE line. Returns (content_delta, usage_dict)."""
    if not line.startswith("data: "):
        return None, None

    data = line[6:]
    if data == "[DONE]":
        return None, None

    chunk = json.loads(data)
    usage = extract_usage(chunk)

    choices = chunk.get("choices", [])
    if not choices:
        return None, usage

    delta = choices[0].get("delta", {})
    content = delta.get("content")
    return content or None, usage


def extract_usage(chunk: dict) -> dict | None:
    """Extract usage info from a chunk if present."""
    usage = chunk.get("usage")
    return usage if usage else None


async def stream_chat(
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "MiniMax-Text-01",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> AsyncIterator[tuple[str | None, dict | None]]:
    """Stream chat completions from Minimax API.

    Yields (content_delta, usage) tuples. Content is None when no text.
    Usage is only present in the final chunk.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                content, usage = parse_sse_line(line)
                yield content, usage
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/llm.py tests/test_llm.py
git commit -m "feat: add Minimax API streaming client with SSE parsing"
```

---

### Task 7: Audio Capture with VAD

**Files:**
- Create: `jarvis/audio_capture.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/audio_capture.py`:

```python
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
        """Record audio until silence is detected. Returns int16 numpy array or None if too short."""
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

                # VAD expects float32 tensor
                audio_float = torch.from_numpy(chunk_1d.astype(np.float32) / 32768.0)
                confidence = self._vad_model(audio_float, self.SAMPLE_RATE).item()

                if confidence >= self._vad_threshold:
                    speech_detected = True
                    silence_count = 0
                else:
                    silence_count += len(chunk_1d)

                # End conditions
                if speech_detected and silence_count >= self._silence_samples:
                    break
                if total_samples >= self._max_samples:
                    break

        self._vad_model.reset_states()

        if not speech_detected:
            return None

        audio = np.concatenate(frames)
        return audio

    @staticmethod
    def get_input_devices() -> list[dict]:
        """List available input audio devices."""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
```

- [ ] **Step 2: Manual test — verify microphone capture works**

```bash
python -c "
from jarvis.audio_capture import AudioCapture
print('Available microphones:')
for d in AudioCapture.get_input_devices():
    print(f'  [{d[\"index\"]}] {d[\"name\"]}')
print()
print('Speak something, then stay silent for 1.5s...')
cap = AudioCapture()
audio = cap.record_until_silence()
if audio is not None:
    print(f'Captured {len(audio) / 16000:.1f}s of audio ({len(audio)} samples)')
else:
    print('No speech detected')
"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/audio_capture.py
git commit -m "feat: add audio capture with silero-vad endpoint detection"
```

---

### Task 8: Wake Word Detection

**Files:**
- Create: `jarvis/wake_word.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/wake_word.py`:

```python
from __future__ import annotations

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel


class WakeWordDetector:
    """Listens for 'hey jarvis' wake word using openwakeword."""

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1280  # 80ms at 16kHz — required by openwakeword

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model_name = model_name
        self._model = OWWModel(wakeword_models=[model_name])

    def listen_blocking(self) -> bool:
        """Block until wake word is detected. Returns True when detected."""
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
```

- [ ] **Step 2: Manual test — verify wake word detection**

```bash
python -c "
from jarvis.wake_word import WakeWordDetector
print('Say \"Hey Jarvis\"...')
detector = WakeWordDetector()
if detector.listen_blocking():
    print('Wake word detected!')
"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/wake_word.py
git commit -m "feat: add wake word detection with openwakeword"
```

---

### Task 9: Hotkey Listener

**Files:**
- Create: `jarvis/hotkey.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/hotkey.py`:

```python
from __future__ import annotations

import asyncio
import threading

from pynput import keyboard


class HotkeyListener:
    """Listens for a global hotkey and signals an asyncio event."""

    # Maps config strings to pynput keys
    _KEY_MAP = {
        "ctrl": keyboard.Key.ctrl_l,
        "alt": keyboard.Key.alt_l,
        "shift": keyboard.Key.shift,
    }

    def __init__(self, hotkey_str: str, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()
        self._hotkey_str = hotkey_str
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        """Start listening for the hotkey in a background thread."""
        hotkey_combo = self._parse_hotkey(self._hotkey_str)

        self._listener = keyboard.GlobalHotKeys({hotkey_combo: self._on_trigger})
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    async def wait(self) -> None:
        """Await the hotkey being pressed. Resets after each trigger."""
        await self._event.wait()
        self._event.clear()

    def _on_trigger(self) -> None:
        self._loop.call_soon_threadsafe(self._event.set)

    @staticmethod
    def _parse_hotkey(hotkey_str: str) -> str:
        """Convert 'ctrl+alt+j' to '<ctrl>+<alt>+j' for pynput."""
        parts = hotkey_str.lower().split("+")
        result = []
        for part in parts:
            part = part.strip()
            if part in ("ctrl", "alt", "shift"):
                result.append(f"<{part}>")
            else:
                result.append(part)
        return "+".join(result)
```

- [ ] **Step 2: Manual test — verify hotkey works**

```bash
python -c "
import asyncio
from jarvis.hotkey import HotkeyListener

async def main():
    loop = asyncio.get_event_loop()
    hk = HotkeyListener('ctrl+alt+j', loop)
    hk.start()
    print('Press Ctrl+Alt+J...')
    await hk.wait()
    print('Hotkey detected!')
    hk.stop()

asyncio.run(main())
"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/hotkey.py
git commit -m "feat: add global hotkey listener with asyncio integration"
```

---

### Task 10: STT Module (faster-whisper)

**Files:**
- Create: `jarvis/stt.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/stt.py`:

```python
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
        """Transcribe int16 audio array to text.

        Returns the transcribed text or empty string if nothing recognized.
        """
        # faster-whisper expects float32 in [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_float,
            language="pt",
            beam_size=3,
            vad_filter=True,
        )

        text_parts = [segment.text.strip() for segment in segments]
        return " ".join(text_parts).strip()
```

- [ ] **Step 2: Manual test — verify transcription works**

```bash
python -c "
from jarvis.audio_capture import AudioCapture
from jarvis.stt import SpeechToText

print('Loading Whisper model (first time may download ~1.5GB)...')
stt = SpeechToText()
print('Model loaded. Speak something...')

cap = AudioCapture()
audio = cap.record_until_silence()

if audio is not None:
    text = stt.transcribe(audio)
    print(f'Transcribed: \"{text}\"')
else:
    print('No speech detected')
"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/stt.py
git commit -m "feat: add STT module with faster-whisper GPU transcription"
```

---

### Task 11: Audio Player

**Files:**
- Create: `jarvis/audio_player.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/audio_player.py`:

```python
from __future__ import annotations

import asyncio
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
        """Start the playback thread."""
        self._playing = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and drain the queue."""
        self._playing = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, audio: np.ndarray) -> None:
        """Add an audio chunk to the playback queue."""
        self._queue.put(audio)

    def finish(self) -> None:
        """Signal that no more audio will be added. Blocks until done."""
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
```

- [ ] **Step 2: Manual test — verify audio playback**

```bash
python -c "
import numpy as np
from jarvis.audio_player import AudioPlayer

# Generate a simple 440Hz sine wave (1 second)
sr = 22050
t = np.linspace(0, 1.0, sr, dtype=np.float32)
tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

player = AudioPlayer(sample_rate=sr)
player.start()
player.enqueue(tone)
print('Playing 440Hz tone for 1 second...')
player.finish()
print('Done')
"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/audio_player.py
git commit -m "feat: add threaded audio player with queue-based playback"
```

---

### Task 12: Model Download Script

**Files:**
- Create: `setup_models.py`

- [ ] **Step 1: Write the script**

Create `setup_models.py`:

```python
"""Download required models for Jarvis."""

import subprocess
import sys
from pathlib import Path

PIPER_MODEL_DIR = Path("models/piper")
PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium"
PIPER_FILES = [
    "pt_BR-faber-medium.onnx",
    "pt_BR-faber-medium.onnx.json",
]


def download_piper_model() -> None:
    PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename in PIPER_FILES:
        target = PIPER_MODEL_DIR / filename
        if target.exists():
            print(f"  [skip] {filename} already exists")
            continue
        url = f"{PIPER_BASE_URL}/{filename}"
        print(f"  [download] {filename}...")
        subprocess.run(
            ["curl", "-L", "-o", str(target), url],
            check=True,
        )
    print("  Piper model ready.")


def download_openwakeword() -> None:
    print("  Downloading openwakeword models...")
    import openwakeword

    openwakeword.utils.download_models()
    print("  openwakeword models ready.")


def main() -> None:
    print("=== Downloading Piper TTS model ===")
    download_piper_model()
    print()
    print("=== Downloading openwakeword models ===")
    download_openwakeword()
    print()
    print("=== Whisper model will auto-download on first run ===")
    print()
    print("All models ready! Run: python -m jarvis.main")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to download models**

```bash
python setup_models.py
```

- [ ] **Step 3: Commit**

```bash
git add setup_models.py
git commit -m "feat: add model download script for Piper and openwakeword"
```

---

### Task 13: Main Orchestrator

**Files:**
- Create: `jarvis/main.py`

- [ ] **Step 1: Write the implementation**

Create `jarvis/main.py`:

```python
"""Jarvis — Main async orchestrator."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from jarvis.audio_capture import AudioCapture
from jarvis.audio_player import AudioPlayer
from jarvis.config import Config, load_config
from jarvis.conversation import Conversation
from jarvis.hotkey import HotkeyListener
from jarvis.llm import stream_chat
from jarvis.stt import SpeechToText
from jarvis.token_logger import TokenLogger
from jarvis.tts import PiperTTS, SentenceSplitter
from jarvis.wake_word import WakeWordDetector

SYSTEM_PROMPT = (
    "Voce e Jarvis, um assistente pessoal inteligente e prestativo. "
    "Responda sempre em portugues brasileiro. "
    "Seja conciso: frases curtas e diretas, adequadas para fala. "
    "Evite listas, markdown, URLs ou formatacao visual. "
    "Quando nao souber algo, diga honestamente. "
    "Seu tom e educado, levemente formal e com um toque de humor sutil. "
    "Areas de interesse do usuario: ciencia, astronomia, fatos curiosos."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/jarvis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("jarvis")


class Jarvis:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.conversation = Conversation(SYSTEM_PROMPT, max_turns=config.max_history)
        self.token_logger = TokenLogger("logs/token_usage.jsonl") if config.log_tokens else None

        log.info("Loading STT model (%s on %s)...", config.whisper_model, config.whisper_device)
        self.stt = SpeechToText(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        piper_model_path = str(Path(config.piper_model_dir) / f"{config.piper_voice}.onnx")
        log.info("Loading TTS model (%s)...", config.piper_voice)
        self.tts = PiperTTS(piper_model_path)

        log.info("Loading audio capture with VAD...")
        self.audio_capture = AudioCapture(
            vad_threshold=config.vad_threshold,
            silence_duration_ms=config.silence_duration_ms,
        )

        self.player = AudioPlayer(sample_rate=self.tts.sample_rate)

        log.info("Loading wake word detector (%s)...", config.wake_word)
        self.wake_detector = WakeWordDetector(
            model_name=config.wake_word,
            threshold=config.wake_threshold,
        )

        self.hotkey: HotkeyListener | None = None

    async def run(self) -> None:
        """Main loop: listen, transcribe, chat, speak, repeat."""
        loop = asyncio.get_event_loop()

        # Start hotkey listener
        self.hotkey = HotkeyListener(self.cfg.hotkey, loop)
        self.hotkey.start()
        log.info("Hotkey listener started: %s", self.cfg.hotkey)

        log.info("Jarvis pronto! Diga '%s' ou pressione %s.", self.cfg.wake_word, self.cfg.hotkey)

        # Speak startup greeting
        await self._speak("Jarvis online. Como posso ajudar?")

        try:
            while True:
                # Wait for activation (wake word or hotkey)
                activated = await self._wait_for_activation(loop)
                if not activated:
                    continue

                log.info("Ativado! Ouvindo...")

                # Record speech
                audio = await loop.run_in_executor(None, self.audio_capture.record_until_silence)
                if audio is None:
                    log.info("Nenhuma fala detectada, voltando ao standby.")
                    continue

                # Transcribe
                log.info("Transcrevendo...")
                text = await loop.run_in_executor(None, self.stt.transcribe, audio)
                if not text:
                    log.info("Transcricao vazia, voltando ao standby.")
                    continue

                log.info("Usuario: %s", text)

                # Get response from Minimax and speak it
                await self._chat_and_speak(text)

        except KeyboardInterrupt:
            log.info("Encerrando Jarvis...")
        finally:
            if self.hotkey:
                self.hotkey.stop()

    async def _wait_for_activation(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Wait for wake word or hotkey. Returns True when activated."""
        wake_future = loop.run_in_executor(None, self.wake_detector.listen_blocking)
        hotkey_task = asyncio.create_task(self.hotkey.wait())

        done, pending = await asyncio.wait(
            [wake_future, hotkey_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self.wake_detector.reset()
        return True

    async def _chat_and_speak(self, user_text: str) -> None:
        """Send text to Minimax API (streaming), synthesize and play response."""
        messages = self.conversation.get_messages_for_api(user_text)

        splitter = SentenceSplitter()
        self.player.start()
        full_response = []
        usage = None

        try:
            async for content, chunk_usage in stream_chat(
                api_key=self.cfg.minimax_api_key,
                messages=messages,
                model=self.cfg.minimax_model,
            ):
                if chunk_usage:
                    usage = chunk_usage

                if content:
                    full_response.append(content)
                    for char in content:
                        sentence = splitter.feed(char)
                        if sentence:
                            audio = self.tts.synthesize(sentence)
                            self.player.enqueue(audio)

            # Flush remaining text
            for sentence in splitter.flush():
                if sentence:
                    audio = self.tts.synthesize(sentence)
                    self.player.enqueue(audio)

        except Exception as e:
            log.error("Erro na API Minimax: %s", e)
            error_audio = self.tts.synthesize(
                "Nao consegui me conectar. Tente novamente em instantes."
            )
            self.player.enqueue(error_audio)

        self.player.finish()

        response_text = "".join(full_response).strip()
        if response_text:
            self.conversation.add_turn(user_text, response_text)
            log.info("Jarvis: %s", response_text)

        if usage and self.token_logger:
            self.token_logger.log(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            log.info("Tokens: %s (acumulado: %d)", usage, self.token_logger.cumulative)

    async def _speak(self, text: str) -> None:
        """Synthesize and play a simple text message."""
        self.player.start()
        audio = self.tts.synthesize(text)
        self.player.enqueue(audio)
        self.player.finish()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    config = load_config()
    jarvis = Jarvis(config)
    asyncio.run(jarvis.run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify module can be imported without errors**

```bash
python -c "from jarvis.main import Jarvis; print('Import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add jarvis/main.py
git commit -m "feat: add main orchestrator with streaming pipeline"
```

---

### Task 14: Windows Startup Script

**Files:**
- Create: `start_jarvis.bat`

- [ ] **Step 1: Create the batch file**

Create `start_jarvis.bat`:

```batch
@echo off
title Jarvis - Assistente de Voz
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m jarvis.main
pause
```

- [ ] **Step 2: Test the startup script**

Double-click `start_jarvis.bat` or run from terminal:

```bash
./start_jarvis.bat
```

Verify Jarvis starts, loads all models, and announces "Jarvis online".

- [ ] **Step 3: Commit**

```bash
git add start_jarvis.bat
git commit -m "feat: add Windows startup batch script"
```

---

### Task 15: End-to-End Integration Test

- [ ] **Step 1: Full pipeline test**

Run Jarvis and test the complete flow:

```bash
python -m jarvis.main
```

Test checklist:
1. Jarvis says "Jarvis online. Como posso ajudar?"
2. Say "Hey Jarvis" → activation sound/log
3. Ask "O que e um buraco negro?" → Jarvis responds in Portuguese via speech
4. Press Ctrl+Alt+J → activation
5. Ask "Qual a distancia da Terra ao Sol?" → Jarvis responds
6. Check `logs/token_usage.jsonl` has entries
7. Check `logs/jarvis.log` has transcriptions and responses

- [ ] **Step 2: Test error handling**

1. Disconnect internet → ask a question → Jarvis should say the error message
2. Check that Jarvis returns to standby after error

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Jarvis voice assistant v1.0"
```

---

## Task Dependency Order

```
Task 1 (scaffolding)
  ├── Task 2 (config)
  ├── Task 3 (conversation)
  ├── Task 4 (token logger)
  ├── Task 5 (TTS splitter + Piper)
  ├── Task 6 (Minimax API client)
  ├── Task 7 (audio capture + VAD)
  ├── Task 8 (wake word)
  ├── Task 9 (hotkey)
  ├── Task 10 (STT)
  ├── Task 11 (audio player)
  ├── Task 12 (model download)
  └── Tasks 2-12 all feed into:
       ├── Task 13 (main orchestrator)
       ├── Task 14 (startup script)
       └── Task 15 (integration test)
```

Tasks 2-12 are independent of each other and can be implemented in any order. Tasks 13-15 depend on all previous tasks.
