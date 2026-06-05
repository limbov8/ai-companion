from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

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
        if len(results) < min(limit, 3):
            results.extend(await self._html_results(query, limit - len(results)))
        return {"query": query, "results": results}

    async def _html_results(self, query: str, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "ai-companion/0.1"})
            response.raise_for_status()
            body = response.text
        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            body,
            flags=re.DOTALL,
        )
        rows = []
        for raw_url, raw_title, raw_snippet in matches[:limit]:
            rows.append(
                {
                    "title": self._clean_html(raw_title),
                    "snippet": self._clean_html(raw_snippet),
                    "url": self._clean_url(raw_url),
                }
            )
        return rows

    @staticmethod
    def _clean_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()

    @staticmethod
    def _clean_url(url: str) -> str:
        url = html.unescape(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
        if url.startswith("//"):
            return f"https:{url}"
        return url
