from __future__ import annotations

from server.memory.store import MemoryItem


def keyword_rerank(query: str, candidates: list[tuple[MemoryItem, float]], top_k: int) -> list[MemoryItem]:
    terms = {term.strip(".,!?;:").lower() for term in query.split() if len(term) > 2}

    def score(pair: tuple[MemoryItem, float]) -> float:
        item, similarity = pair
        text_terms = set(item.text.lower().split())
        overlap = len(terms & text_terms) / max(len(terms), 1)
        return similarity + overlap

    return [item for item, _ in sorted(candidates, key=score, reverse=True)[:top_k]]
