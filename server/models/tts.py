from __future__ import annotations

import asyncio
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
            return text.encode("utf-8")

        return await self.gpu_gate.run(self.model_id, work)

    async def _load_engine(self) -> object:
        return None
