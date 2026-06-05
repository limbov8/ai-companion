from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import httpx

from server.config import DeepSeekConfig


Message = dict[str, str]


class ChatClient(Protocol):
    async def complete(self, messages: list[Message], *, purpose: str = "conversation") -> str:
        ...

    async def decide_memory(self, text: str) -> dict[str, object]:
        ...


@dataclass
class DeepSeekClient:
    config: DeepSeekConfig
    timeout_seconds: float = 45.0

    async def complete(self, messages: list[Message], *, purpose: str = "conversation") -> str:
        if not self.config.api_key:
            return self._offline_response(messages, purpose)

        model = (
            self.config.conversation_model
            if purpose == "conversation"
            else self.config.utility_model
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "reasoning_effort": self.config.reasoning_effort,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def decide_memory(self, text: str) -> dict[str, object]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Decide if the user's text contains durable personal memory. "
                    "Return JSON with keys remember:boolean, category:string, summary:string."
                ),
            },
            {"role": "user", "content": text},
        ]
        raw = await self.complete(messages, purpose="utility")
        try:
            parsed = json.loads(raw)
            return {
                "remember": bool(parsed.get("remember")),
                "category": str(parsed.get("category", "general")),
                "summary": str(parsed.get("summary", text)).strip(),
            }
        except json.JSONDecodeError:
            return self._heuristic_memory(text)

    def _offline_response(self, messages: list[Message], purpose: str) -> str:
        if purpose == "utility":
            return json.dumps(self._heuristic_memory(messages[-1]["content"]))
        latest = messages[-1]["content"]
        return f"I heard you. Here is a grounded next step: {latest}"

    @staticmethod
    def _heuristic_memory(text: str) -> dict[str, object]:
        lowered = text.lower()
        durable_markers = ["remember", "i like", "i prefer", "my ", "call me", "important"]
        remember = any(marker in lowered for marker in durable_markers)
        return {
            "remember": remember,
            "category": "preference" if "like" in lowered or "prefer" in lowered else "general",
            "summary": text.strip(),
        }
