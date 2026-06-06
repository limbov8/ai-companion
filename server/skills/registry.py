from __future__ import annotations

from dataclasses import dataclass, field

from server.skills.base import Skill, SkillSpec


@dataclass
class SkillRegistry:
    _skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> None:
        self._skills[skill.spec.name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list_specs(self) -> list[SkillSpec]:
        return [skill.spec for skill in self._skills.values()]

    def has(self, name: str) -> bool:
        return name in self._skills
