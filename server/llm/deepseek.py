from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import httpx

from server.config import DeepSeekConfig


Message = dict[str, str]


class ChatClient(Protocol):
    async def complete(self, messages: list[Message], *, purpose: str = "conversation") -> str:
        ...

    def stream_complete(
        self,
        messages: list[Message],
        *,
        purpose: str = "conversation",
    ) -> AsyncIterator[str]:
        ...

    async def decide_memory(self, text: str) -> dict[str, object]:
        ...

    async def decide_web_search(self, text: str) -> dict[str, object]:
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
        self._print_exchange(
            "deepseek.input",
            {
                "purpose": purpose,
                "model": model,
                "messages": messages,
                "stream": False,
                "reasoning_effort": self.config.reasoning_effort,
            },
        )
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            self._print_exchange(
                "deepseek.output",
                {
                    "purpose": purpose,
                    "model": model,
                    "status_code": response.status_code,
                    "content": content,
                },
            )
            return content

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        purpose: str = "conversation",
    ) -> AsyncIterator[str]:
        if not self.config.api_key:
            yield self._offline_response(messages, purpose)
            return

        model = (
            self.config.conversation_model
            if purpose == "conversation"
            else self.config.utility_model
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "reasoning_effort": self.config.reasoning_effort,
        }
        self._print_exchange(
            "deepseek.input",
            {
                "purpose": purpose,
                "model": model,
                "messages": messages,
                "stream": True,
                "reasoning_effort": self.config.reasoning_effort,
            },
        )
        collected: list[str] = []
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = str(delta.get("content") or "")
                    if content:
                        collected.append(content)
                        yield content
        self._print_exchange(
            "deepseek.output",
            {
                "purpose": purpose,
                "model": model,
                "stream": True,
                "content": "".join(collected),
            },
        )

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

    async def decide_web_search(self, text: str) -> dict[str, object]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Decide whether the assistant must search the web before answering. "
                    "Return JSON only with keys search:boolean, query:string, reason:string. "
                    "Search is required for current or time-sensitive facts: stocks, market, "
                    "news, prices, weather, recent events, release dates, laws, schedules, "
                    "or anything the model may not know confidently."
                ),
            },
            {"role": "user", "content": text},
        ]
        raw = await self.complete(messages, purpose="utility")
        try:
            parsed = json.loads(raw)
            return {
                "search": bool(parsed.get("search")),
                "query": str(parsed.get("query") or text).strip(),
                "reason": str(parsed.get("reason") or "").strip(),
            }
        except json.JSONDecodeError:
            return {"search": False, "query": text, "reason": "tool decision parse failed"}

    def _offline_response(self, messages: list[Message], purpose: str) -> str:
        if purpose == "utility":
            return json.dumps(self._heuristic_memory(messages[-1]["content"]))
        latest = messages[-1]["content"]
        return f"I heard you. Here is a grounded next step: {latest}"

    @staticmethod
    def _print_exchange(label: str, payload: dict[str, object]) -> None:
        print(f"{label} {json.dumps(payload, ensure_ascii=False, indent=2)}", flush=True)

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
