from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from server.tools.base import ToolSpec


@dataclass
class WebSearchTool:
    spec = ToolSpec(
        name="web_search",
        description="Search the public web for current information and return source snippets.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs["query"])
        limit = int(kwargs.get("limit", 5))
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            response.raise_for_status()
            data = response.json()
        related = data.get("RelatedTopics", [])[:limit]
        results = [
            {
                "title": item.get("Text", "").split(" - ")[0],
                "snippet": item.get("Text", ""),
                "url": item.get("FirstURL", ""),
            }
            for item in related
            if isinstance(item, dict)
        ]
        return {"query": query, "results": results}
