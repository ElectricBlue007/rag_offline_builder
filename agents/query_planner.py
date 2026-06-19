"""Deterministic query planner."""

from __future__ import annotations


class QueryPlanner:
    def plan(self, query: str) -> dict:
        lowered = query.lower()
        language = "es" if any(word in lowered for word in ("qué", "sistema", "datos", "informe")) else "en"
        expansion_depth = 2 if any(word in lowered for word in ("explain", "why", "cómo", "por qué")) else 1
        return {
            "query": query,
            "language": language,
            "top_k": 5,
            "expansion_depth": expansion_depth,
            "filters": None,
        }
