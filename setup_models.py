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
