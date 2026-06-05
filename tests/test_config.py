from __future__ import annotations

from pathlib import Path

from server.config import load_config


def test_config_uses_environment_secret(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yml"
    config_file.write_text(Path("config.yml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-from-env")

    config = load_config(config_file)

    assert config.deepseek.api_key == "secret-from-env"
    assert config.models.asr_model_id == "Qwen/Qwen3-ASR-1.7B"
    assert config.models.tts_languages == ("English", "Chinese")
