from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        ...
