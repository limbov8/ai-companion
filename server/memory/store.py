from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    category: str
    metadata: dict[str, Any]
    embedding: list[float]
    created_at: datetime


@dataclass
class InMemoryVectorStore:
    items: list[MemoryItem] = field(default_factory=list)

    async def add(
        self,
        text: str,
        embedding: list[float],
        *,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            id=str(uuid.uuid4()),
            text=text,
            category=category,
            metadata=metadata or {},
            embedding=embedding,
            created_at=datetime.now(UTC),
        )
        self.items.append(item)
        return item

    async def search(self, query_embedding: list[float], *, top_k: int) -> list[tuple[MemoryItem, float]]:
        scored = [(item, cosine_similarity(item.embedding, query_embedding)) for item in self.items]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(v * v for v in left[:size])) or 1.0
    right_norm = math.sqrt(sum(v * v for v in right[:size])) or 1.0
    return dot / (left_norm * right_norm)
