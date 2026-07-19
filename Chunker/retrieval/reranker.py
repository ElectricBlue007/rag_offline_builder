"""
retrieval/reranker.py

Cross-encoder re-ranking — a final precision pass over the hybrid
retriever's top candidates. See RerankerConfig in config.py for the
full rationale on why this is a separate stage from dense/sparse
retrieval rather than a replacement for either.

Decoupled from HybridRetriever deliberately: this module only needs a
query string and a list of RetrievedChunk objects (or anything with
.chunk_id/.text/.metadata), so it can rerank output from any retrieval
source, not just the specific HybridRetriever implementation.
"""

from dataclasses import dataclass
from typing import Optional, Union

from Chunker.config import RerankerConfig
from Chunker.retrieval.hybrid_retriever import RetrievedChunk


@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    metadata: dict
    rerank_score: float
    # What the candidate's fusion score and position were BEFORE
    # reranking — kept so callers/evaluators can see how much reranking
    # actually changed the ordering (useful for step 9's evaluation).
    original_rrf_score: float
    original_rank: int


class CrossEncoderReranker:
    """
    Usage:
        reranker = CrossEncoderReranker(config)
        results = reranker.rerank(query, candidates)  # candidates: list[RetrievedChunk]
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.cfg = config or RerankerConfig()
        self._model = None  # lazy-loaded; settable directly in tests

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install sentence-transformers\n"
                "(Qwen3-Reranker models additionally require transformers>=4.51.0)"
            ) from e

        print(f"  [CrossEncoderReranker] Loading model: {self.cfg.model_name}")
        if self.cfg.task_instruction:
            # Qwen3-Reranker's "instruction aware" prompt mechanism: a
            # named prompt is prepended to every (query, document) pair.
            # Harmless to pass for cross-encoders that don't use prompts
            # at all (they simply ignore the kwarg), so this isn't
            # gated behind a Qwen-specific model-name check.
            self._model = CrossEncoder(
                self.cfg.model_name,
                max_length=self.cfg.max_length,
                prompts={"task": self.cfg.task_instruction},
                default_prompt_name="task",
            )
        else:
            self._model = CrossEncoder(self.cfg.model_name, max_length=self.cfg.max_length)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[Union[RetrievedChunk, object]],
    ) -> list[RerankedChunk]:
        """
        Re-score every (query, candidate.text) pair with the cross-
        encoder and return the top final_n_results, sorted by the
        cross-encoder's score (highest first) — NOT by the candidates'
        original fusion order.
        """
        if not query or not query.strip() or not candidates:
            return []

        model = self._get_model()
        pairs = [(query, self._get_text(c)) for c in candidates]
        scores = model.predict(pairs)

        reranked = [
            RerankedChunk(
                chunk_id=self._get_field(candidate, "chunk_id"),
                text=self._get_text(candidate),
                metadata=self._get_field(candidate, "metadata", {}),
                rerank_score=float(score),
                original_rrf_score=self._get_field(candidate, "rrf_score", 0.0),
                original_rank=original_rank,
            )
            for original_rank, (candidate, score) in enumerate(zip(candidates, scores))
        ]

        reranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return reranked[: self.cfg.final_n_results]

    @staticmethod
    def _get_text(candidate) -> str:
        if isinstance(candidate, dict):
            return candidate.get("text", "")
        return getattr(candidate, "text", "")

    @staticmethod
    def _get_field(candidate, field: str, default=""):
        if isinstance(candidate, dict):
            return candidate.get(field, default)
        return getattr(candidate, field, default)
