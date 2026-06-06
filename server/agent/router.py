from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from server.agent.decision import AgentDecision, AgentMode, ToolCall
from server.agent.task_state import ActiveTask
from server.llm.deepseek import ChatClient
from server.memory.store import MemoryItem
from server.skills.base import SkillSpec
from server.tools.base import ToolSpec


@dataclass(frozen=True)
class AgentContext:
    user_text: str
    history: list[dict[str, str]]
    memories: list[MemoryItem]
    active_task: ActiveTask | None
    tools: list[ToolSpec]
    skills: list[SkillSpec]


@dataclass
class AgentRouter:
    chat: ChatClient | None = None

    async def decide(self, ctx: AgentContext) -> AgentDecision:
        deterministic = self.decide_rules(ctx)
        if deterministic:
            return deterministic
        if not self.chat:
            return AgentDecision.answer(reason="router fallback without llm")
        try:
            return await self.decide_llm(ctx)
        except Exception:
            return AgentDecision.answer(reason="router llm failed")

    def decide_rules(self, ctx: AgentContext) -> AgentDecision | None:
        text = ctx.user_text.strip()
        if self.is_casual(text):
            return AgentDecision.answer(intent="casual_support", reason="casual conversational message")
        if self.needs_fresh_info(text) and self.has_tool(ctx, "web_search"):
            return AgentDecision(
                mode=AgentMode.USE_TOOL,
                intent="current_info",
                confidence=0.95,
                tool_calls=[
                    ToolCall(
                        tool_name="web_search",
                        args={"query": text, "limit": 6},
                        reason="fresh or external information request",
                    )
                ],
            )
        return None

    async def decide_llm(self, ctx: AgentContext) -> AgentDecision:
        raw = await self.chat.complete(
            [
                {"role": "system", "content": self.router_prompt(ctx)},
                {"role": "user", "content": ctx.user_text},
            ],
            purpose="utility",
        )
        parsed = json.loads(raw)
        return self.parse_decision(parsed)

    def router_prompt(self, ctx: AgentContext) -> str:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "when_to_use": tool.when_to_use,
                "when_not_to_use": tool.when_not_to_use,
                "side_effect": tool.side_effect,
                "requires_confirmation": tool.requires_confirmation,
                "freshness": tool.freshness,
            }
            for tool in ctx.tools
        ]
        skills = [
            {
                "name": skill.name,
                "description": skill.description,
                "when_to_use": skill.when_to_use,
                "required_slots": skill.required_slots,
            }
            for skill in ctx.skills
        ]
        active_task = None
        if ctx.active_task:
            active_task = {
                "skill_name": ctx.active_task.skill_name,
                "status": ctx.active_task.status,
                "state": ctx.active_task.state,
            }
        return (
            "You are the control router for a personal AI companion. "
            "Decide the next action before the assistant answers. "
            "Prefer tools for questions about current, recent, changing, external, or unknown facts. "
            "Prefer skills for multi-turn workflows. "
            "If the user's request is complex or open-ended (planning, organizing, multi-step tasks), "
            "consider start_skill to propose a structured plan before taking action. "
            "Ask one clarifying question only when a critical slot is missing. "
            "Return strict JSON matching: "
            "{mode,intent,confidence,required_slots,tool_calls,skill_name,response_style,reason}. "
            "mode must be one of answer, ask_clarifying, use_tool, start_skill, continue_skill. "
            f"Available tools: {json.dumps(tools, ensure_ascii=False)}. "
            f"Available skills: {json.dumps(skills, ensure_ascii=False)}. "
            f"Active task: {json.dumps(active_task, ensure_ascii=False)}."
        )

    @staticmethod
    def parse_decision(data: dict[str, Any]) -> AgentDecision:
        mode = AgentMode(str(data.get("mode", "answer")).lower())
        tool_calls = [
            ToolCall(
                tool_name=str(item.get("tool_name") or item.get("name") or ""),
                args=dict(item.get("args") or {}),
                reason=item.get("reason"),
            )
            for item in data.get("tool_calls", [])
            if isinstance(item, dict)
        ]
        return AgentDecision(
            mode=mode,
            intent=str(data.get("intent") or mode.value),
            confidence=float(data.get("confidence") or 0.5),
            required_slots=[str(slot) for slot in data.get("required_slots", [])],
            tool_calls=tool_calls,
            skill_name=data.get("skill_name"),
            response_style=data.get("response_style"),
            reason=data.get("reason"),
        )

    @staticmethod
    def has_tool(ctx: AgentContext, name: str) -> bool:
        return any(tool.name == name for tool in ctx.tools)

    @staticmethod
    def has_skill(ctx: AgentContext, name: str) -> bool:
        return any(skill.name == name for skill in ctx.skills)

    @staticmethod
    def is_casual(text: str) -> bool:
        lowered = text.lower()
        markers = ("累", "难过", "开心", "hello", "thanks", "谢谢", "有点")
        question_markers = ("?", "？", "吗", "怎么样", "多少", "什么", "how", "what", "when", "where")
        casual = any(marker in lowered for marker in markers) or bool(re.search(r"\bhi\b", lowered))
        return casual and not any(marker in lowered for marker in question_markers)

    @staticmethod
    def needs_fresh_info(text: str) -> bool:
        lowered = text.lower()
        markers = (
            "stock",
            "market",
            "price",
            "news",
            "latest",
            "today",
            "weather",
            "forecast",
            "current",
            "right now",
            "股票",
            "股市",
            "大盘",
            "行情",
            "新闻",
            "最新",
            "今天",
            "今日",
            "现在",
            "天气",
            "会下雨",
            "价格",
        )
        if any(marker in lowered for marker in markers):
            return True
        return bool(re.search(r"\bwhat'?s\s+(happening|new|going on)\b", lowered))
