"""Vector store interfaces and shared similarity functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None:
        ...

    def search(self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions do not match.")
    return sum(a * b for a, b in zip(left, right))


def payload_matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        current = payload.get(key)
        if isinstance(expected, (list, tuple, set)):
            if current not in expected:
                return False
        elif current != expected:
            return False
    return True
