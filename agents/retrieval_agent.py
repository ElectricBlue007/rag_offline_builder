"""Retrieval controller agent."""

from __future__ import annotations

from core.retrieval_engine import RetrievalEngine


class RetrievalAgent:
    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval_engine = retrieval_engine

    def retrieve(self, plan: dict) -> list[dict]:
        return self.retrieval_engine.retrieve(
            plan["query"],
            top_k=plan["top_k"],
            filters=plan.get("filters"),
            expansion_depth=plan["expansion_depth"],
        )
