from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from server.models.gpu import SingleGpuGate


@dataclass
class LocalAsrService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    language: str | None = None
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
        if self._is_qwen_asr:
            return await self._load_qwen_pipeline(strict=strict)
        return await self._load_whisper_pipeline(strict=strict)

    @property
    def _is_qwen_asr(self) -> bool:
        return "qwen3-asr" in self.model_id.lower()

    async def _load_qwen_pipeline(self, *, strict: bool = False) -> object | None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            if strict:
                raise RuntimeError("Install Qwen ASR dependencies with make install-local-models.") from exc
            return None

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        kwargs = {
            "dtype": dtype,
            "device_map": "cuda:0" if self.device == "cuda" else self.device,
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        try:
            model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except Exception:
            model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                attn_implementation="sdpa",
                **kwargs,
            )

        def transcribe(data: bytes, content_type: str) -> str:
            suffix = self._suffix_for_content_type(content_type)
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(data)
                    temp_path = temp_file.name
                results = model.transcribe(audio=temp_path, language=self.language)
                if not results:
                    return ""
                return str(getattr(results[0], "text", "")).strip()
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

        return transcribe

    async def _load_whisper_pipeline(self, *, strict: bool = False) -> object | None:
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

    @staticmethod
    def _suffix_for_content_type(content_type: str) -> str:
        if "wav" in content_type:
            return ".wav"
        if "mp4" in content_type or "m4a" in content_type:
            return ".m4a"
        if "ogg" in content_type:
            return ".ogg"
        return ".webm"
