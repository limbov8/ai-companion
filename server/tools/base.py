from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    when_to_use: str = ""
    when_not_to_use: str = ""
    side_effect: bool = False
    requires_confirmation: bool = False
    freshness: str = "static"


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        ...
