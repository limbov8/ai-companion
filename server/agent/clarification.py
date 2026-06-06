from __future__ import annotations

VOICE_CLARIFICATION_MAX_CHARS = 140


def should_ask_clarifying(*, missing_slots: list[str], critical_slots: set[str]) -> bool:
    return bool(set(missing_slots) & critical_slots)


def one_question(question: str, *, voice: bool = True) -> str:
    cleaned = " ".join(question.split())
    if "?" not in cleaned and "？" not in cleaned:
        cleaned = f"{cleaned}?"
    if voice and len(cleaned) > VOICE_CLARIFICATION_MAX_CHARS:
        return cleaned[: VOICE_CLARIFICATION_MAX_CHARS - 1].rstrip() + "?"
    return cleaned


def low_risk_assumption_text(assumptions: list[str]) -> str:
    if not assumptions:
        return ""
    return "Assuming " + ", ".join(assumptions) + "."
