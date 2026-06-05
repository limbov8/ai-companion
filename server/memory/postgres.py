from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Text, create_engine, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from server.memory.store import MemoryItem


class Base(DeclarativeBase):
    pass


class MemoryRow(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


@dataclass
class PostgresVectorStore:
    dsn: str

    def __post_init__(self) -> None:
        self.engine = create_engine(self.dsn)
        self.session_factory = sessionmaker(self.engine)

    async def add(
        self,
        text_value: str,
        embedding: list[float],
        *,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        memory_id = uuid.uuid4()
        now = datetime.now(UTC)
        vector = "[" + ",".join(str(value) for value in embedding) + "]"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO memories (id, text, category, metadata, embedding, created_at, updated_at)
                    VALUES (:id, :text, :category, CAST(:metadata AS jsonb), CAST(:embedding AS vector), :created_at, :updated_at)
                    """
                ),
                {
                    "id": str(memory_id),
                    "text": text_value,
                    "category": category,
                    "metadata": metadata or {},
                    "embedding": vector,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return MemoryItem(
            id=str(memory_id),
            text=text_value,
            category=category,
            metadata=metadata or {},
            embedding=embedding,
            created_at=now,
        )

    async def search(self, query_embedding: list[float], *, top_k: int) -> list[tuple[MemoryItem, float]]:
        vector = "[" + ",".join(str(value) for value in query_embedding) + "]"
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, text, category, metadata, created_at,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                    FROM memories
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                {"embedding": vector, "top_k": top_k},
            ).mappings()
            return [
                (
                    MemoryItem(
                        id=str(row["id"]),
                        text=row["text"],
                        category=row["category"],
                        metadata=dict(row["metadata"]),
                        embedding=[],
                        created_at=row["created_at"],
                    ),
                    float(row["similarity"]),
                )
                for row in rows
            ]
