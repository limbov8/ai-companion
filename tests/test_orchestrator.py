from __future__ import annotations

from dataclasses import dataclass

import pytest

from server.agent.orchestrator import AgentOrchestrator
from server.config import ConversationConfig, MemoryConfig
from server.memory.store import InMemoryVectorStore
from server.models.embedding import EmbeddingService
from server.models.gpu import SingleGpuGate
from server.prompts.registry import PromptRegistry
from server.tools.registry import ToolRegistry


@dataclass
class FakeChat:
    async def complete(self, messages, *, purpose="conversation"):
        if purpose == "utility":
            return '{"remember": true, "category": "preference", "summary": "User likes quiet mornings."}'
        assert any("Relevant memory" in message["content"] for message in messages)
        return "Good plan. I will keep it quiet and practical."

    async def decide_memory(self, text):
        return {"remember": "quiet" in text.lower(), "category": "preference", "summary": text}


@pytest.mark.asyncio
async def test_orchestrator_stores_and_retrieves_memory():
    embeddings = EmbeddingService("test-embedding", "cpu", SingleGpuGate(), dimensions=32)
    store = InMemoryVectorStore()
    orchestrator = AgentOrchestrator(
        chat=FakeChat(),
        embeddings=embeddings,
        memory_store=store,
        prompts=PromptRegistry(),
        tools=ToolRegistry(),
        memory_config=MemoryConfig(top_k=4, rerank_top_k=2, similarity_floor=-1.0),
        conversation_config=ConversationConfig(
            proactive_topic_interval_minutes=45,
            allow_random_topics=True,
        ),
    )

    first = await orchestrator.handle_text("I like quiet mornings.", [], source="text")
    second = await orchestrator.handle_text("Plan my morning.", [], source="text")

    assert first.memory_stored is True
    assert len(store.items) == 1
    assert second.text.startswith("Good plan")
    assert second.memories_used
