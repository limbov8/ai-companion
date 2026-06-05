from __future__ import annotations

from dataclasses import dataclass

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
        system_prompt = self.prompts.render("life_helper_system", memory_context=memory_context)
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_text}]
        answer = await self.chat.complete(messages, purpose="conversation")
        stored = await self.maybe_store_memory(user_text, source=source)
        return AgentResponse(text=answer, memories_used=memories, memory_stored=stored)

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
