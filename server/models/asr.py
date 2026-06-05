from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from server.models.gpu import SingleGpuGate


@dataclass
class WhisperAsrService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    _pipeline: object | None = None

    async def preload(self, *, strict: bool = False) -> bool:
        if self._pipeline is None:
            self._pipeline = await self._load_pipeline(strict=strict)
        return self._pipeline is not None

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

    async def _load_pipeline(self, *, strict: bool = False) -> object:
        if os.getenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "0") != "1":
            if strict:
                raise RuntimeError("Local ASR loading is disabled. Set AI_COMPANION_ENABLE_LOCAL_MODELS=1.")
            return None
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            if strict:
                raise RuntimeError("Install local model dependencies with make install-local-models.") from exc
            return None
        torch_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            framework="pt",
            torch_dtype=torch_dtype,
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"low_cpu_mem_usage": True},
        )

        def transcribe(data: bytes, _content_type: str) -> str:
            result = pipe(data)
            return str(result.get("text", "")).strip()

        return transcribe
