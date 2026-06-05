from __future__ import annotations

from dataclasses import dataclass

import pytest

from server.agent.orchestrator import AgentOrchestrator
from server.config import ConversationConfig, MemoryConfig
from server.memory.store import InMemoryVectorStore
from server.models.embedding import EmbeddingService
from server.models.gpu import SingleGpuGate
from server.prompts.registry import PromptRegistry
from server.tools.base import ToolSpec
from server.tools.registry import ToolRegistry


@dataclass
class FakeChat:
    messages: list[list[dict[str, str]]] = None

    def __post_init__(self):
        self.messages = []

    async def complete(self, messages, *, purpose="conversation"):
        if purpose == "utility":
            return '{"remember": true, "category": "preference", "summary": "User likes quiet mornings."}'
        assert any("Relevant memory" in message["content"] for message in messages)
        self.messages.append(messages)
        return "Good plan. I will keep it quiet and practical."

    async def stream_complete(self, messages, *, purpose="conversation"):
        await self.complete(messages, purpose=purpose)
        for chunk in ("Good plan. ", "I will keep it quiet and practical."):
            yield chunk

    async def decide_memory(self, text):
        return {"remember": "quiet" in text.lower(), "category": "preference", "summary": text}


@dataclass
class FakeWebSearchTool:
    spec = ToolSpec(
        name="web_search",
        description="Fake web search",
        input_schema={"type": "object"},
    )
    calls: list[str] = None

    def __post_init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(str(kwargs["query"]))
        return {
            "query": kwargs["query"],
            "results": [
                {
                    "title": "Example market headline",
                    "snippet": "A current stock market snippet.",
                    "url": "https://example.com/market",
                }
            ],
        }


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


@pytest.mark.asyncio
async def test_orchestrator_searches_web_for_current_questions():
    embeddings = EmbeddingService("test-embedding", "cpu", SingleGpuGate(), dimensions=32)
    store = InMemoryVectorStore()
    tools = ToolRegistry()
    search = FakeWebSearchTool()
    tools.register(search)
    chat = FakeChat()
    orchestrator = AgentOrchestrator(
        chat=chat,
        embeddings=embeddings,
        memory_store=store,
        prompts=PromptRegistry(),
        tools=tools,
        memory_config=MemoryConfig(top_k=4, rerank_top_k=2, similarity_floor=-1.0),
        conversation_config=ConversationConfig(
            proactive_topic_interval_minutes=45,
            allow_random_topics=True,
        ),
    )

    response = await orchestrator.handle_text("What's the stock news today?", [], source="voice")

    assert search.calls == ["What's the stock news today?"]
    assert "Example market headline" in response.tool_context
    assert "Current web/tool context" in chat.messages[-1][0]["content"]
    assert "Example market headline" in chat.messages[-1][0]["content"]


@pytest.mark.asyncio
async def test_orchestrator_streams_response_and_final_memory_metadata():
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

    events = [
        event
        async for event in orchestrator.handle_text_stream(
            "I like quiet mornings.",
            [],
            source="voice",
        )
    ]

    assert [event["type"] for event in events] == ["text_delta", "text_delta", "done"]
    assert events[-1]["text"].startswith("Good plan")
    assert events[-1]["memory_stored"] is True
