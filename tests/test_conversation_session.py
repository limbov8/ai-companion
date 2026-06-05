from __future__ import annotations

from server.agent.conversation import ConversationSession


def test_conversation_session_can_replace_loaded_messages():
    session = ConversationSession(session_id="abc")
    session.add("user", "old")

    session.replace_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert session.messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
