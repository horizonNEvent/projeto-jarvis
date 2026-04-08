import os
import pytest


def test_config_loads_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIMAX_API_KEY=sk-test-123\n"
        "MINIMAX_MODEL=MiniMax-M2.7\n"
        "MAX_HISTORY=3\n"
        "LOG_TOKENS=false\n"
        "SILENCE_DURATION_MS=2000\n"
    )
    monkeypatch.chdir(tmp_path)

    from jarvis.config import load_config

    cfg = load_config(str(env_file))
    assert cfg.minimax_api_key == "sk-test-123"
    assert cfg.minimax_model == "MiniMax-M2.7"
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
