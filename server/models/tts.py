from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass

from server.models.gpu import SingleGpuGate


@dataclass
class QwenTtsService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    _engine: object | None = None

    async def synthesize(self, text: str, *, interrupt_token: str | None = None) -> bytes:
        async def work() -> bytes:
            if not text.strip():
                return b""
            if interrupt_token == "cancelled":
                return b""
            if self._engine is None:
                self._engine = await self._load_engine()
            if callable(self._engine):
                return await asyncio.to_thread(self._engine, text)
            return self._fallback_wav(text)

        return await self.gpu_gate.run(self.model_id, work)

    async def _load_engine(self) -> object:
        return None

    @staticmethod
    def _fallback_wav(text: str) -> bytes:
        duration_seconds = min(max(len(text) / 24, 0.35), 2.0)
        sample_rate = 12_000
        frame_count = int(sample_rate * duration_seconds)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()
