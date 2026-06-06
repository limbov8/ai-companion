from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from server.agent.clarification import one_question
from server.skills.base import SkillContext, SkillSpec, SkillState, SkillStepResult


@dataclass
class PlanningSkill:
    spec = SkillSpec(
        name="planning",
        description="Stateful planning for outings, trips, projects, and family activities.",
        when_to_use=(
            "Use when the user asks to plan an outing, trip, weekend, baby activity, "
            "project, schedule, or multi-step plan."
        ),
        required_slots=["goal"],
    )

    async def start(self, ctx: SkillContext) -> SkillStepResult:
        data = self.extract_slots(ctx.user_text, {})
        return self.next_result(data)

    async def step(self, ctx: SkillContext, state: SkillState, user_text: str) -> SkillStepResult:
        data = self.extract_slots(user_text, dict(state.data))
        return self.next_result(data)

    def next_result(self, data: dict[str, Any]) -> SkillStepResult:
        missing = self.missing_critical_slots(data)
        if missing:
            question = self.question_for_missing(data, missing[0])
            data["phase"] = "clarifying"
            return SkillStepResult(
                status="need_user_input",
                question=question,
                updated_state=data,
            )
        data["phase"] = "completed"
        return SkillStepResult(
            status="final_answer",
            updated_state=data,
            final_answer=self.render_plan(data),
        )

    def missing_critical_slots(self, data: dict[str, Any]) -> list[str]:
        missing = []
        if not data.get("goal"):
            missing.append("goal")
        goal = str(data.get("goal") or "").lower()
        outing_like = any(marker in goal for marker in ("出去玩", "出门", "trip", "weekend", "周末", "activity"))
        if outing_like and not data.get("location"):
            missing.append("location")
        return missing

    def question_for_missing(self, data: dict[str, Any], slot: str) -> str:
        if slot == "location":
            if self.has_child_context(data):
                return one_question(
                    "可以。我先按轻松、低刺激、适合带小宝宝来规划。你想在 Sunnyvale 附近，还是可以开车远一点？"
                )
            return one_question("可以。你想在附近安排，还是可以开车远一点？")
        return one_question("可以。你最想完成的目标是什么？")

    def render_plan(self, data: dict[str, Any]) -> str:
        goal = data.get("goal") or "这个计划"
        location = data.get("location") or "附近"
        timeframe = data.get("date_or_timeframe") or "这次"
        constraints = data.get("constraints") or "节奏轻松"
        audience = data.get("audience_or_people") or "你们"
        budget = data.get("budget") or "中等预算"
        return (
            f"好，我按{constraints}来安排。{timeframe}可以在{location}做一个简单计划：\n"
            f"1. 先选一个低压力主活动，适合{audience}，控制在 1-2 小时。\n"
            f"2. 中间留出休息、喝水和临时调整时间。\n"
            f"3. 如果状态不错，再加一个很近的备选点；如果累了就直接回家。\n"
            f"4. 预算先按{budget}处理，优先选停车方便、排队少的地方。\n"
            f"我会把目标记为：{goal}。"
        )

    def extract_slots(self, text: str, current: dict[str, Any]) -> dict[str, Any]:
        data = {
            "goal": current.get("goal"),
            "location": current.get("location"),
            "date_or_timeframe": current.get("date_or_timeframe"),
            "budget": current.get("budget"),
            "constraints": current.get("constraints"),
            "audience_or_people": current.get("audience_or_people"),
            "phase": current.get("phase", "active"),
        }
        stripped = text.strip()
        lowered = stripped.lower()
        if not data["goal"] and self.looks_like_planning(stripped):
            data["goal"] = stripped
        elif data["goal"] and stripped:
            data["goal"] = data["goal"]
        for location in ("Sunnyvale", "Cupertino", "Mountain View", "San Jose", "Palo Alto"):
            if location.lower() in lowered:
                data["location"] = location
        if "附近" in stripped and not data["location"]:
            data["location"] = "附近"
        if "周末" in stripped or "weekend" in lowered:
            data["date_or_timeframe"] = "周末"
        if "今天" in stripped:
            data["date_or_timeframe"] = "今天"
        if "明天" in stripped:
            data["date_or_timeframe"] = "明天"
        if any(marker in stripped for marker in ("宝宝", "娃", "孩子", "小孩")):
            data["audience_or_people"] = "带小朋友"
        if any(marker in stripped for marker in ("别太累", "轻松", "低刺激", "不累")):
            data["constraints"] = "轻松、别太累"
        if "便宜" in stripped or "budget" in lowered:
            data["budget"] = "低预算"
        if re.search(r"\$\d+|\d+\s*(元|块|刀)", stripped):
            data["budget"] = re.search(r"\$\d+|\d+\s*(元|块|刀)", stripped).group(0)
        return data

    @staticmethod
    def looks_like_planning(text: str) -> bool:
        lowered = text.lower()
        markers = ("计划", "规划", "安排", "plan", "trip", "project", "周末", "出去玩", "活动")
        return any(marker in lowered for marker in markers)

    @staticmethod
    def has_child_context(data: dict[str, Any]) -> bool:
        return "小" in str(data.get("audience_or_people") or "") or "娃" in str(data.get("goal") or "")
