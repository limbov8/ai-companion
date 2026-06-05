from __future__ import annotations

from server.main import pop_speakable_chunks


def test_pop_speakable_chunks_prefers_sentence_boundaries():
    chunks, pending = pop_speakable_chunks("Hello there. I am still thinking")

    assert chunks == ["Hello there."]
    assert pending == "I am still thinking"


def test_pop_speakable_chunks_splits_long_clause():
    text = "This is a fairly long thought that should become speakable before the whole answer ends, "
    text += "because waiting for a final period would feel slow in a voice conversation"

    chunks, pending = pop_speakable_chunks(text)

    assert chunks
    assert pending.strip().startswith("because")
