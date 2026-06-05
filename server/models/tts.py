from __future__ import annotations

import asyncio
import io
import os
import wave
from dataclasses import dataclass

from server.models.gpu import SingleGpuGate


@dataclass
class QwenTtsService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    language: str = "English"
    speaker: str = "Ryan"
    instruct: str = "Natural, warm, conversational speech."
    _engine: object | None = None

    async def preload(self, *, strict: bool = False) -> bool:
        if self._engine is None:
            self._engine = await self._load_engine(strict=strict)
        return self._engine is not None

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

    async def _load_engine(self, *, strict: bool = False) -> object | None:
        if os.getenv("AI_COMPANION_ENABLE_LOCAL_MODELS", "0") != "1":
            if strict:
                raise RuntimeError("Local TTS loading is disabled. Set AI_COMPANION_ENABLE_LOCAL_MODELS=1.")
            return None
        os.environ.setdefault("USE_TF", "0")
        try:
            import soundfile as sf
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            if strict:
                raise RuntimeError("Install Qwen TTS dependencies with make install-local-models.") from exc
            return None

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        kwargs = {
            "device_map": "cuda:0" if self.device == "cuda" else self.device,
            "dtype": dtype,
        }
        try:
            model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except Exception:
            model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                attn_implementation="sdpa",
                **kwargs,
            )

        def synthesize(text: str) -> bytes:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                instruct=self.instruct,
            )
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sample_rate, format="WAV")
            return buffer.getvalue()

        return synthesize

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
