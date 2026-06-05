from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    description: str
    template: str


class PromptRegistry:
    def __init__(self, path: str | Path = "prompts/templates.yml") -> None:
        self.path = Path(path)
        self._templates = self._load()

    def get(self, name: str) -> PromptTemplate:
        return self._templates[name]

    def list(self) -> list[PromptTemplate]:
        return list(self._templates.values())

    def render(self, name: str, **values: str) -> str:
        prompt = self.get(name).template
        return prompt.format(**values)

    def _load(self) -> dict[str, PromptTemplate]:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        templates = {}
        for item in raw["prompts"]:
            template = PromptTemplate(**item)
            templates[template.name] = template
        return templates
