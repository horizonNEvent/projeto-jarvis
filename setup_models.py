"""Download required models and binaries for Jarvis."""

import subprocess
import sys
import zipfile
from pathlib import Path

PIPER_DIR = Path("models/piper")
PIPER_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium"
PIPER_VOICE_FILES = [
    "pt_BR-faber-medium.onnx",
    "pt_BR-faber-medium.onnx.json",
]
PIPER_BINARY_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
PIPER_BINARY_ZIP = "piper_windows_amd64.zip"


def download_piper_binary() -> None:
    """Download the standalone piper.exe binary for Windows."""
    piper_exe = PIPER_DIR / "piper.exe"
    if piper_exe.exists():
        print("  [skip] piper.exe already exists")
        return

    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PIPER_DIR / PIPER_BINARY_ZIP

    print("  [download] piper_windows_amd64.zip (~30MB)...")
    subprocess.run(
        ["curl", "-L", "-o", str(zip_path), PIPER_BINARY_URL],
        check=True,
    )

    print("  [extract] extracting piper binary...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Files inside zip are in a "piper/" subfolder
        for member in zf.namelist():
            filename = Path(member).name
            if not filename:
                continue
            target = PIPER_DIR / filename
            with zf.open(member) as source, open(target, "wb") as dest:
                dest.write(source.read())

    zip_path.unlink()
    print("  Piper binary ready.")


def download_piper_voice() -> None:
    """Download the pt_BR-faber-medium voice model."""
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    for filename in PIPER_VOICE_FILES:
        target = PIPER_DIR / filename
        if target.exists():
            print(f"  [skip] {filename} already exists")
            continue
        url = f"{PIPER_VOICE_BASE_URL}/{filename}"
        print(f"  [download] {filename}...")
        subprocess.run(
            ["curl", "-L", "-o", str(target), url],
            check=True,
        )
    print("  Piper voice model ready.")


def download_openwakeword() -> None:
    """Download openwakeword pre-trained models."""
    print("  Downloading openwakeword models...")
    import openwakeword

    openwakeword.utils.download_models()
    print("  openwakeword models ready.")


def main() -> None:
    print("=== Downloading Piper TTS binary ===")
    download_piper_binary()
    print()
    print("=== Downloading Piper voice model ===")
    download_piper_voice()
    print()
    print("=== Downloading openwakeword models ===")
    download_openwakeword()
    print()
    print("=== Whisper model will auto-download on first run ===")
    print()
    print("All models ready! Run: python -m jarvis.main")


if __name__ == "__main__":
    main()
