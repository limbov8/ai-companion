from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class SingleGpuGate:
    """Serializes heavyweight model work so one GPU is not overcommitted."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_model: str | None = None

    async def run(self, model_name: str, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            self.active_model = model_name
            try:
                return await fn()
            finally:
                self.active_model = None
