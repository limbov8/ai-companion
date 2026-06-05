from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from server.agent.conversation import ConversationSession


@dataclass
class VoiceSessionManager:
    sessions: dict[str, ConversationSession] = field(default_factory=dict)

    def get(self, session_id: str | None = None) -> ConversationSession:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def barge_in(self, session_id: str) -> int:
        return self.get(session_id).barge_in()
