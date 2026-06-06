from __future__ import annotations

import pytest

from server.agent.decision import AgentMode
from server.agent.router import AgentContext, AgentRouter
from server.agent.task_state import ActiveTask
from server.skills.planning import PlanningSkill
from server.tools.web_search import WebSearchTool


def ctx(user_text: str, *, active_task: ActiveTask | None = None) -> AgentContext:
    return AgentContext(
        user_text=user_text,
        history=[],
        memories=[],
        active_task=active_task,
        tools=[WebSearchTool().spec],
        skills=[PlanningSkill().spec],
    )


@pytest.mark.asyncio
async def test_router_casual_emotional_message_answers():
    decision = await AgentRouter().decide(ctx("我今天有点累"))

    assert decision.mode == AgentMode.ANSWER


@pytest.mark.asyncio
async def test_router_current_weather_uses_tool():
    decision = await AgentRouter().decide(ctx("今天 Sunnyvale 会下雨吗？"))

    assert decision.mode == AgentMode.USE_TOOL
    assert decision.tool_calls[0].tool_name == "web_search"


@pytest.mark.asyncio
async def test_router_vague_planning_falls_back_to_answer_without_llm():
    decision = await AgentRouter().decide(ctx("帮我计划周末带娃出去玩"))

    assert decision.mode == AgentMode.ANSWER


@pytest.mark.asyncio
async def test_router_followup_continues_active_task():
    task = ActiveTask.create("planning", {"goal": "帮我计划周末带娃出去玩"})

    decision = await AgentRouter().decide(ctx("Sunnyvale 附近，别太累", active_task=task))

    assert decision.mode == AgentMode.CONTINUE_SKILL
    assert decision.skill_name == "planning"


@pytest.mark.asyncio
async def test_router_memory_request_falls_back_to_answer_without_llm():
    decision = await AgentRouter().decide(ctx("帮我记住我爸喜欢喝龙井"))

    assert decision.mode == AgentMode.ANSWER


@pytest.mark.asyncio
async def test_router_static_fact_answers():
    decision = await AgentRouter().decide(ctx("What is a binary search?"))

    assert decision.mode == AgentMode.ANSWER
