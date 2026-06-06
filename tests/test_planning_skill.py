from __future__ import annotations

import pytest

from server.skills.base import SkillContext, SkillState
from server.skills.planning import PlanningSkill


def ctx(text: str) -> SkillContext:
    return SkillContext(user_text=text, history=[], memories=[])


@pytest.mark.asyncio
async def test_planning_skill_starts_vague_request_with_one_question():
    result = await PlanningSkill().start(ctx("帮我计划周末带娃出去玩"))

    assert result.status == "need_user_input"
    assert result.question
    assert result.question.count("？") + result.question.count("?") == 1


@pytest.mark.asyncio
async def test_planning_skill_extracts_location_date_and_constraints():
    result = await PlanningSkill().start(ctx("帮我计划周末在 Sunnyvale 附近带娃出去玩，别太累"))

    assert result.updated_state["location"] == "Sunnyvale"
    assert result.updated_state["date_or_timeframe"] == "周末"
    assert result.updated_state["constraints"] == "轻松、别太累"


@pytest.mark.asyncio
async def test_planning_skill_continues_with_active_state():
    skill = PlanningSkill()
    state = SkillState(
        skill_name="planning",
        data={"goal": "帮我计划周末带娃出去玩", "date_or_timeframe": "周末"},
        status="waiting_user",
    )

    result = await skill.step(ctx("Sunnyvale 附近，别太累"), state, "Sunnyvale 附近，别太累")

    assert result.status == "final_answer"
    assert "Sunnyvale" in result.final_answer


@pytest.mark.asyncio
async def test_planning_skill_produces_plan_when_enough_slots_are_available():
    result = await PlanningSkill().start(ctx("help me plan a weekend trip in Sunnyvale with low budget"))

    assert result.status == "final_answer"
    assert "Sunnyvale" in result.final_answer


@pytest.mark.asyncio
async def test_planning_skill_does_not_ask_multiple_questions():
    result = await PlanningSkill().start(ctx("帮我计划周末出去玩"))

    question_marks = result.question.count("?") + result.question.count("？")
    assert question_marks <= 1
