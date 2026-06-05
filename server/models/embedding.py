from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass

from server.models.gpu import SingleGpuGate


@dataclass
class EmbeddingService:
    model_id: str
    device: str
    gpu_gate: SingleGpuGate
    dimensions: int = 1024
    _model: object | None = None

    async def embed(self, text: str) -> list[float]:
        async def work() -> list[float]:
            if self._model is None:
                self._model = await self._load_model()
            if self._model is not None:
                vector = await asyncio.to_thread(self._model.encode, text)
                return [float(value) for value in vector]
            return self._hash_embedding(text)

        return await self.gpu_gate.run(self.model_id, work)

    async def _load_model(self) -> object | None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            return SentenceTransformer(self.model_id, device=self.device)
        except Exception:
            return None

    def _hash_embedding(self, text: str) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1 for byte in digest)
            counter += 1
        vector = values[: self.dimensions]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
