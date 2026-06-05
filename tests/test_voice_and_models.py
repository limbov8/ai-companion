from __future__ import annotations

import wave
from io import BytesIO

import pytest

from server.models.gpu import SingleGpuGate
from server.models.asr import LocalAsrService
from server.models.tts import QwenTtsService, TtsUnavailableError
from server.voice.session import VoiceSessionManager


def test_barge_in_advances_generation():
    manager = VoiceSessionManager()
    session = manager.get("abc")

    assert session.speaking_generation == 0
    assert manager.barge_in("abc") == 1
    assert session.speaking_generation == 1


def test_asr_provider_is_selected_from_model_id():
    qwen = LocalAsrService("Qwen/Qwen3-ASR-1.7B", "cpu", SingleGpuGate())
    whisper = LocalAsrService("openai/whisper-large-v3", "cpu", SingleGpuGate())

    assert qwen._is_qwen_asr
    assert not whisper._is_qwen_asr


@pytest.mark.asyncio
async def test_tts_unavailable_raises_by_default(monkeypatch):
    monkeypatch.setenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "0")
    monkeypatch.delenv("AI_COMPANION_ALLOW_TTS_FALLBACK_WAV", raising=False)
    service = QwenTtsService("test-tts", "cpu", SingleGpuGate())

    with pytest.raises(TtsUnavailableError):
        await service.synthesize("hello there")


@pytest.mark.asyncio
async def test_tts_fallback_returns_audible_wav(monkeypatch):
    monkeypatch.setenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "0")
    monkeypatch.setenv("AI_COMPANION_ALLOW_TTS_FALLBACK_WAV", "1")
    service = QwenTtsService("test-tts", "cpu", SingleGpuGate())

    audio = await service.synthesize("hello there")

    with wave.open(BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == 12_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0
        assert wav.readframes(wav.getnframes()).strip(b"\x00")


def test_tts_auto_selects_chinese_for_cjk_text():
    service = QwenTtsService(
        "test-tts",
        "cpu",
        SingleGpuGate(),
        language="English",
        languages=("English", "Chinese"),
    )

    assert service._language_for_text("hello") == "English"
    assert service._language_for_text("你好") == "Chinese"


def test_tts_model_id_selects_base_voice_clone_mode():
    base = QwenTtsService("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "cpu", SingleGpuGate())
    custom = QwenTtsService("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "cpu", SingleGpuGate())

    assert base._uses_base_voice_clone()
    assert not custom._uses_base_voice_clone()


@pytest.mark.asyncio
async def test_mlx_tts_requires_macos_when_strict(monkeypatch):
    monkeypatch.setenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "1")
    monkeypatch.setattr("server.models.tts.platform.system", lambda: "Windows")
    service = QwenTtsService(
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        "cuda",
        SingleGpuGate(),
    )

    assert service._uses_mlx_backend()
    with pytest.raises(RuntimeError, match="MLX-format"):
        await service.preload(strict=True)


def test_tts_fast_voice_adds_instruction_hint():
    service = QwenTtsService(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "cpu",
        SingleGpuGate(),
        speed=1.18,
        instruct="Natural.",
    )

    assert "faster" in service._speed_instruct().lower()
