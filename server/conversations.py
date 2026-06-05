from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationSummary:
    session_id: str
    title: str
    turn_count: int
    last_role: str
    last_content: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationTurnRecord:
    role: str
    content: str
    created_at: datetime


class ConversationRepository:
    def record_turn(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError

    def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        raise NotImplementedError

    def load_conversation(self, session_id: str) -> list[ConversationTurnRecord]:
        raise NotImplementedError


@dataclass
class PostgresConversationRepository(ConversationRepository):
    dsn: str

    def __post_init__(self) -> None:
        self.engine = create_engine(self.dsn)

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO conversation_turns (session_id, role, content)
                        VALUES (:session_id, :role, :content)
                        """
                    ),
                    {"session_id": session_id, "role": role, "content": content},
                )
        except SQLAlchemyError as exc:
            log.warning("Could not persist conversation turn to database: %s", exc)

    def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        try:
            with self.engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        WITH ranked AS (
                          SELECT
                            session_id,
                            role,
                            content,
                            created_at,
                            row_number() OVER (PARTITION BY session_id ORDER BY created_at ASC, id ASC) AS first_rank,
                            row_number() OVER (
                              PARTITION BY session_id, role ORDER BY created_at ASC, id ASC
                            ) AS role_rank,
                            row_number() OVER (PARTITION BY session_id ORDER BY created_at DESC, id DESC) AS last_rank,
                            count(*) OVER (PARTITION BY session_id) AS turn_count,
                            min(created_at) OVER (PARTITION BY session_id) AS first_at,
                            max(created_at) OVER (PARTITION BY session_id) AS last_at
                          FROM conversation_turns
                        )
                        SELECT
                          latest.session_id,
                          COALESCE(NULLIF(first_user.content, ''), latest.content) AS title,
                          latest.turn_count,
                          latest.role AS last_role,
                          latest.content AS last_content,
                          latest.first_at AS created_at,
                          latest.last_at AS updated_at
                        FROM ranked latest
                        LEFT JOIN ranked first_user
                          ON first_user.session_id = latest.session_id
                         AND first_user.role = 'user'
                         AND first_user.role_rank = 1
                        WHERE latest.last_rank = 1
                        ORDER BY latest.last_at DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                ).mappings()
                return [
                    ConversationSummary(
                        session_id=row["session_id"],
                        title=self._shorten(row["title"]),
                        turn_count=int(row["turn_count"]),
                        last_role=row["last_role"],
                        last_content=self._shorten(row["last_content"], limit=160),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
        except SQLAlchemyError as exc:
            log.warning("Could not list conversations from database: %s", exc)
            return []

    def load_conversation(self, session_id: str) -> list[ConversationTurnRecord]:
        try:
            with self.engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT role, content, created_at
                        FROM conversation_turns
                        WHERE session_id = :session_id
                        ORDER BY created_at ASC, id ASC
                        """
                    ),
                    {"session_id": session_id},
                ).mappings()
                return [
                    ConversationTurnRecord(
                        role=row["role"],
                        content=row["content"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
        except SQLAlchemyError as exc:
            log.warning("Could not load conversation from database: %s", exc)
            return []

    @staticmethod
    def _shorten(value: str, *, limit: int = 72) -> str:
        compact = " ".join(str(value or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1].rstrip() + "..."
