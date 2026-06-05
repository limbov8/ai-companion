from __future__ import annotations

import wave
from io import BytesIO

import pytest

from server.models.gpu import SingleGpuGate
from server.models.tts import QwenTtsService
from server.voice.session import VoiceSessionManager


def test_barge_in_advances_generation():
    manager = VoiceSessionManager()
    session = manager.get("abc")

    assert session.speaking_generation == 0
    assert manager.barge_in("abc") == 1
    assert session.speaking_generation == 1


@pytest.mark.asyncio
async def test_tts_fallback_returns_valid_wav(monkeypatch):
    monkeypatch.setenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "0")
    service = QwenTtsService("test-tts", "cpu", SingleGpuGate())

    audio = await service.synthesize("hello there")

    with wave.open(BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == 12_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0
