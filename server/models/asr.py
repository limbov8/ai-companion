from __future__ import annotations

import asyncio
from dataclasses import dataclass

from server.models.gpu import SingleGpuGate


@dataclass
class WhisperAsrService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    _pipeline: object | None = None

    async def transcribe(self, audio_bytes: bytes, content_type: str = "audio/webm") -> str:
        async def work() -> str:
            if not audio_bytes:
                return ""
            if self._pipeline is None:
                self._pipeline = await self._load_pipeline()
            if callable(self._pipeline):
                return await asyncio.to_thread(self._pipeline, audio_bytes, content_type)
            return "[asr unavailable: install local-models extras]"

        return await self.gpu_gate.run(self.model_id, work)

    async def _load_pipeline(self) -> object:
        try:
            from transformers import pipeline
        except ImportError:
            return None

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1,
        )

        def transcribe(data: bytes, _content_type: str) -> str:
            result = pipe(data)
            return str(result.get("text", "")).strip()

        return transcribe
