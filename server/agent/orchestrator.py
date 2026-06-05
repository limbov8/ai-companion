from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from server.config import ConversationConfig, MemoryConfig
from server.llm.deepseek import ChatClient
from server.memory.rerank import keyword_rerank
from server.memory.store import InMemoryVectorStore, MemoryItem
from server.models.embedding import EmbeddingService
from server.prompts.registry import PromptRegistry
from server.tools.registry import ToolRegistry


@dataclass
class AgentResponse:
    text: str
    memories_used: list[MemoryItem]
    memory_stored: bool
    tool_context: str = ""


@dataclass
class AgentOrchestrator:
    chat: ChatClient
    embeddings: EmbeddingService
    memory_store: InMemoryVectorStore
    prompts: PromptRegistry
    tools: ToolRegistry
    memory_config: MemoryConfig
    conversation_config: ConversationConfig

    async def handle_text(
        self,
        user_text: str,
        history: list[dict[str, str]],
        *,
        source: str = "voice",
    ) -> AgentResponse:
        memories = await self.retrieve_memories(user_text)
        memory_context = "\n".join(f"- {item.text}" for item in memories) or "None."
        tool_result = await self.maybe_run_context_tool(user_text)
        web_context = self.format_tool_context(tool_result) if tool_result else "None."
        system_prompt = self.prompts.render(
            "life_helper_system",
            memory_context=memory_context,
            web_context=web_context,
        )
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_text}]
        answer = await self.chat.complete(messages, purpose="conversation")
        stored = await self.maybe_store_memory(user_text, source=source)
        if tool_result:
            await self.maybe_store_tool_result("web_search", tool_result)
        return AgentResponse(
            text=answer,
            memories_used=memories,
            memory_stored=stored,
            tool_context=web_context if tool_result else "",
        )

    async def maybe_run_context_tool(self, user_text: str) -> dict[str, Any] | None:
        if not self.should_search_web(user_text):
            return None
        try:
            return await self.tools.run("web_search", query=user_text, limit=6)
        except Exception as exc:
            return {"query": user_text, "error": str(exc), "results": []}

    @staticmethod
    def should_search_web(user_text: str) -> bool:
        text = user_text.lower()
        current_markers = (
            "stock",
            "share price",
            "market",
            "ticker",
            "nasdaq",
            "dow jones",
            "s&p",
            "crypto",
            "bitcoin",
            "price of",
            "news",
            "headline",
            "latest",
            "today",
            "current",
            "right now",
            "this morning",
            "this week",
            "weather",
            "forecast",
            "earnings",
        )
        if any(marker in text for marker in current_markers):
            return True
        return bool(re.search(r"\bwhat'?s\s+(happening|new|going on)\b", text))

    @staticmethod
    def format_tool_context(result: dict[str, Any]) -> str:
        if result.get("error"):
            return f"web_search error for {result.get('query', '')}: {result['error']}"
        rows = []
        for index, item in enumerate(result.get("results", []), start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            rows.append(f"{index}. {title}\n   {snippet}\n   Source: {url}")
        if not rows:
            return f"web_search found no useful results for: {result.get('query', '')}"
        return "web_search results:\n" + "\n".join(rows)

    async def retrieve_memories(self, query: str) -> list[MemoryItem]:
        embedding = await self.embeddings.embed(query)
        candidates = await self.memory_store.search(embedding, top_k=self.memory_config.top_k)
        candidates = [
            pair for pair in candidates if pair[1] >= self.memory_config.similarity_floor
        ]
        return keyword_rerank(query, candidates, self.memory_config.rerank_top_k)

    async def maybe_store_memory(self, text: str, *, source: str) -> bool:
        decision = await self.chat.decide_memory(text)
        if not decision.get("remember"):
            return False
        summary = str(decision.get("summary") or text).strip()
        category = str(decision.get("category") or "general")
        embedding = await self.embeddings.embed(summary)
        await self.memory_store.add(summary, embedding, category=category, metadata={"source": source})
        return True

    async def maybe_store_tool_result(self, tool_name: str, result: dict[str, object]) -> bool:
        decision = await self.chat.decide_memory(str(result))
        if not decision.get("remember"):
            return False
        summary = str(decision.get("summary") or result)
        embedding = await self.embeddings.embed(summary)
        await self.memory_store.add(
            summary,
            embedding,
            category=str(decision.get("category") or "web"),
            metadata={"source": "tool", "tool": tool_name},
        )
        return True
