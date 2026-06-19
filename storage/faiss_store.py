"""Pure-Python FAISS-compatible fallback store.

The class intentionally exposes a FAISS-like local vector index interface while
remaining dependency-free. A production FAISS adapter can replace it later.
"""

from __future__ import annotations

from core.vector_store import SearchResult, VectorRecord, cosine_similarity, payload_matches


class FaissStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.record_id] = record

    def search(self, vector: list[float], top_k: int, filters: dict | None = None) -> list[SearchResult]:
        results = [
            SearchResult(record=record, score=cosine_similarity(vector, record.vector))
            for record in self._records.values()
            if payload_matches(record.payload, filters)
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)
