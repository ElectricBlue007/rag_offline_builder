"""Hybrid reranking for vector + graph + metadata evidence."""

from __future__ import annotations


class Reranker:
    def rerank(self, candidates: list[dict]) -> list[dict]:
        for candidate in candidates:
            graph_distance = candidate.get("graph_distance", 0)
            graph_score = 1.0 / (1.0 + graph_distance)
            importance = float(candidate["payload"].get("importance_score", 0.0))
            semantic = candidate.get("semantic_relevance_score", 0.0)
            chunk_kind_boost = 0.08 if candidate["payload"].get("chunk_kind") == "semantic" else 0.0
            candidate["final_score"] = round(
                candidate.get("vector_score", 0.0) * 0.35
                + graph_score * 0.15
                + importance * 0.2
                + semantic * 0.3
                + chunk_kind_boost,
                6,
            )
        return sorted(candidates, key=lambda item: item["final_score"], reverse=True)
