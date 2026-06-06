from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from server.agent.decision import AgentDecision, AgentMode, ToolCall
from server.agent.router import AgentContext, AgentRouter
from server.agent.task_state import ActiveTask
from server.config import ConversationConfig, MemoryConfig
from server.llm.deepseek import ChatClient
from server.memory.rerank import keyword_rerank
from server.memory.store import InMemoryVectorStore, MemoryItem
from server.models.embedding import EmbeddingService
from server.prompts.registry import PromptRegistry
from server.skills.base import SkillContext, SkillState, SkillStepResult
from server.skills.registry import SkillRegistry
from server.tools.registry import ToolRegistry


@dataclass
class AgentResponse:
    text: str
    memories_used: list[MemoryItem]
    memory_stored: bool
    tool_context: str = ""
    active_task: ActiveTask | None = None
    decision: AgentDecision | None = None


@dataclass
class AgentPreparedTurn:
    messages: list[dict[str, str]]
    memories: list[MemoryItem]
    web_context: str
    tool_result: dict[str, Any] | None
    decision: AgentDecision
    active_task: ActiveTask | None = None


@dataclass
class AgentOrchestrator:
    chat: ChatClient
    embeddings: EmbeddingService
    memory_store: InMemoryVectorStore
    prompts: PromptRegistry
    tools: ToolRegistry
    skills: SkillRegistry
    router: AgentRouter
    memory_config: MemoryConfig
    conversation_config: ConversationConfig

    async def handle_text(
        self,
        user_text: str,
        history: list[dict[str, str]],
        *,
        source: str = "voice",
        active_task: ActiveTask | None = None,
    ) -> AgentResponse:
        prepared = await self.prepare_turn(user_text, history, active_task=active_task)
        answer, task = await self.act(prepared, user_text, source=source)
        stored = await self.maybe_store_memory(user_text, source=source)
        if prepared.tool_result:
            await self.maybe_store_tool_result("web_search", prepared.tool_result)
        return AgentResponse(
            text=answer,
            memories_used=prepared.memories,
            memory_stored=stored,
            tool_context=prepared.web_context if prepared.tool_result else "",
            active_task=task,
            decision=prepared.decision,
        )

    async def handle_text_stream(
        self,
        user_text: str,
        history: list[dict[str, str]],
        *,
        source: str = "voice",
        active_task: ActiveTask | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        prepared = await self.prepare_turn(user_text, history, active_task=active_task)
        if prepared.decision.mode in {
            AgentMode.ASK_CLARIFYING,
            AgentMode.START_SKILL,
            AgentMode.CONTINUE_SKILL,
        }:
            answer, task = await self.act(prepared, user_text, source=source)
            yield {"type": "text_delta", "text": answer}
            stored = await self.maybe_store_memory(user_text, source=source)
            yield {
                "type": "done",
                "text": answer,
                "memory_stored": stored,
                "memories_used": [item.text for item in prepared.memories],
                "tool_context": prepared.web_context if prepared.tool_result else "",
                "active_task": task,
            }
            return
        answer_parts: list[str] = []
        async for delta in self.chat.stream_complete(prepared.messages, purpose="conversation"):
            answer_parts.append(delta)
            yield {"type": "text_delta", "text": delta}
        answer = "".join(answer_parts)
        stored = await self.maybe_store_memory(user_text, source=source)
        if prepared.tool_result:
            await self.maybe_store_tool_result("web_search", prepared.tool_result)
        yield {
            "type": "done",
            "text": answer,
            "memory_stored": stored,
            "memories_used": [item.text for item in prepared.memories],
            "tool_context": prepared.web_context if prepared.tool_result else "",
            "active_task": active_task,
        }

    async def prepare_turn(
        self,
        user_text: str,
        history: list[dict[str, str]],
        *,
        active_task: ActiveTask | None = None,
    ) -> AgentPreparedTurn:
        memories = await self.retrieve_memories(user_text)
        ctx = AgentContext(
            user_text=user_text,
            history=history,
            memories=memories,
            active_task=active_task,
            tools=self.tools.list_specs(),
            skills=self.skills.list_specs(),
        )
        decision = await self.router.decide(ctx)
        tool_result = await self.run_decision_tools(decision)
        memory_context = "\n".join(f"- {item.text}" for item in memories) or "None."
        web_context = self.format_tool_context(tool_result) if tool_result else "None."
        system_prompt = self.prompts.render(
            "life_helper_system",
            memory_context=memory_context,
            web_context=web_context,
        )
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_text}]
        return AgentPreparedTurn(
            messages=messages,
            memories=memories,
            web_context=web_context,
            tool_result=tool_result,
            decision=decision,
            active_task=active_task,
        )

    async def act(
        self,
        prepared: AgentPreparedTurn,
        user_text: str,
        *,
        source: str,
    ) -> tuple[str, ActiveTask | None]:
        decision = prepared.decision
        if decision.mode == AgentMode.ASK_CLARIFYING:
            return self.clarifying_question(decision), prepared.active_task
        if decision.mode == AgentMode.START_SKILL and decision.skill_name:
            return await self.start_skill(decision.skill_name, prepared, user_text)
        if decision.mode == AgentMode.CONTINUE_SKILL and prepared.active_task:
            return await self.continue_skill(prepared.active_task, prepared, user_text)
        return await self.chat.complete(prepared.messages, purpose="conversation"), prepared.active_task

    async def run_decision_tools(self, decision: AgentDecision) -> dict[str, Any] | None:
        if decision.mode != AgentMode.USE_TOOL or not decision.tool_calls:
            return None
        results = []
        for call in decision.tool_calls:
            results.append(await self.run_tool_call(call))
        if len(results) == 1:
            return results[0]
        return {"query": "multiple tool calls", "results": results}

    async def run_tool_call(self, call: ToolCall) -> dict[str, Any]:
        try:
            return await self.tools.run(call.tool_name, **call.args)
        except Exception as exc:
            return {"query": call.args.get("query", call.tool_name), "error": str(exc), "results": []}

    def clarifying_question(self, decision: AgentDecision) -> str:
        if decision.reason and ("?" in decision.reason or "？" in decision.reason):
            return decision.reason
        slot = decision.required_slots[0] if decision.required_slots else "detail"
        return f"Before I continue, what {slot} should I use?"

    async def start_skill(
        self,
        skill_name: str,
        prepared: AgentPreparedTurn,
        user_text: str,
    ) -> tuple[str, ActiveTask | None]:
        skill = self.skills.get(skill_name)
        result = await skill.start(self.skill_context(prepared, user_text))
        return self.render_skill_result(skill_name, result, None)

    async def continue_skill(
        self,
        task: ActiveTask,
        prepared: AgentPreparedTurn,
        user_text: str,
    ) -> tuple[str, ActiveTask | None]:
        skill = self.skills.get(task.skill_name)
        state = SkillState(skill_name=task.skill_name, data=task.state, status=task.status)
        result = await skill.step(self.skill_context(prepared, user_text), state, user_text)
        return self.render_skill_result(task.skill_name, result, task)

    def skill_context(self, prepared: AgentPreparedTurn, user_text: str) -> SkillContext:
        return SkillContext(
            user_text=user_text,
            history=prepared.messages,
            memories=prepared.memories,
            active_task=prepared.active_task,
        )

    def render_skill_result(
        self,
        skill_name: str,
        result: SkillStepResult,
        task: ActiveTask | None,
    ) -> tuple[str, ActiveTask | None]:
        if task is None:
            task = ActiveTask.create(skill_name, result.updated_state)
        else:
            task.update(result.updated_state)
        if result.status == "need_user_input":
            task.update(result.updated_state, status="waiting_user")
            return result.question or "What detail should I use next?", task
        if result.status == "final_answer":
            task.update(result.updated_state, status="completed")
            return result.final_answer or "Done.", None
        if result.status == "failed":
            task.update(result.updated_state, status="abandoned")
            return result.final_answer or "I hit a problem with that workflow.", None
        task.update(result.updated_state, status="active")
        return result.final_answer or result.question or "I am working on it.", task

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
