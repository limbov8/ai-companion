from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from server.tools.base import Tool, ToolSpec


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def run(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return await self._tools[name].run(**kwargs)
