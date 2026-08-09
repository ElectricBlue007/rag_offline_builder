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

Two backends, selected via RerankerConfig.backend:
  "huggingface" — sentence_transformers.CrossEncoder, a real
    cross-encoder forward pass. See RerankerConfig for why this is the
    default.
  "ollama" — no dedicated cross-encoder architecture involved at all;
    scores come from asking an Ollama-served instruct model a yes/no
    question per pair and reading the yes-vs-no token probability off
    the response's logprobs. See _rerank_ollama for the mechanics.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

from rag_pipeline.config import RerankerConfig
from rag_pipeline.retrieval.hybrid_retriever import RetrievedChunk


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


# Fixed instruction for the "ollama" backend's yes/no prompt. Kept
# separate from RerankerConfig.task_instruction (which is
# huggingface-backend/Qwen3-Reranker-specific prompt wiring) since the
# two backends use different prompt mechanisms entirely.
_OLLAMA_RERANK_SYSTEM_PROMPT = (
    'Judge whether the Document is relevant to the Query. '
    'Answer with exactly one word: "yes" or "no". No other text.'
)

# A logprob this far below zero is effectively "never happens" — used
# as a floor when a candidate token doesn't appear in top_logprobs at
# all, so a completely absent "yes" (or "no") still yields a sane
# probability instead of crashing or silently defaulting to 0.5.
_ABSENT_TOKEN_LOGPROB_FLOOR = -100.0


class CrossEncoderReranker:
    """
    Usage:
        reranker = CrossEncoderReranker(config)
        results = reranker.rerank(query, candidates)  # candidates: list[RetrievedChunk]
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.cfg = config or RerankerConfig()
        self._model = None  # lazy-loaded; settable directly in tests (huggingface backend)
        self._ollama_client = None  # lazy-loaded (ollama backend)

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
        # device=None lets sentence-transformers auto-pick; model_kwargs
        # (e.g. {"torch_dtype": "float16"}) is forwarded to
        # AutoModel.from_pretrained. Empty dict/None are both safe no-ops.
        common_kwargs = {
            "max_length": self.cfg.max_length,
            "device": self.cfg.device,
            "model_kwargs": self.cfg.model_kwargs or {},
        }
        if self.cfg.task_instruction:
            # Qwen3-Reranker's "instruction aware" prompt mechanism: a
            # named prompt is prepended to every (query, document) pair.
            # Harmless to pass for cross-encoders that don't use prompts
            # at all (they simply ignore the kwarg), so this isn't
            # gated behind a Qwen-specific model-name check.
            self._model = CrossEncoder(
                self.cfg.model_name,
                prompts={"task": self.cfg.task_instruction},
                default_prompt_name="task",
                **common_kwargs,
            )
        else:
            self._model = CrossEncoder(self.cfg.model_name, **common_kwargs)
        return self._model

    def _get_ollama_client(self):
        if self._ollama_client is not None:
            return self._ollama_client
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ollama\n"
                "and ensure the Ollama service is running locally at "
                f"{self.cfg.ollama_host} with model '{self.cfg.ollama_model}' pulled."
            ) from e
        print(f"  [CrossEncoderReranker] Using Ollama model: {self.cfg.ollama_model}")
        self._ollama_client = ollama.Client(host=self.cfg.ollama_host)
        return self._ollama_client

    def rerank(
        self,
        query: str,
        candidates: list[Union[RetrievedChunk, object]],
    ) -> list[RerankedChunk]:
        """
        Re-score every (query, candidate.text) pair and return the top
        final_n_results, sorted by score (highest first) — NOT by the
        candidates' original fusion order. Dispatches to the
        huggingface or ollama backend per RerankerConfig.backend.
        """
        if not query or not query.strip() or not candidates:
            return []

        if self.cfg.backend == "ollama":
            scores = self._score_ollama(query, candidates)
        else:
            scores = self._score_huggingface(query, candidates)

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

    def _score_huggingface(self, query: str, candidates: list) -> list[float]:
        model = self._get_model()
        pairs = [(query, self._get_text(c)) for c in candidates]
        return list(model.predict(pairs))

    def _score_ollama(self, query: str, candidates: list) -> list[float]:
        """
        Generative reranking: one Ollama chat call per candidate, asking
        a strict yes/no relevance question and turning the yes-vs-no
        token probability at the answer position into a 0..1 score.

        This is the same trick Qwen3-Reranker's own "generative" scoring
        mode uses, just against Ollama's API instead of loading the
        model directly — works with any instruct model already pulled
        into Ollama, not only dedicated reranker models.
        """
        client = self._get_ollama_client()
        options = {"temperature": 0.0, "num_predict": 1}
        if self.cfg.ollama_num_ctx is not None:
            options["num_ctx"] = self.cfg.ollama_num_ctx

        scores = []
        for candidate in candidates:
            document = self._get_text(candidate)
            user_prompt = f"<Query>: {query}\n<Document>: {document}"
            try:
                response = client.chat(
                    model=self.cfg.ollama_model,
                    messages=[
                        {"role": "system", "content": _OLLAMA_RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    options=options,
                    logprobs=True,
                    top_logprobs=self.cfg.ollama_top_logprobs,
                )
            except TypeError:
                # Older ollama-python versions don't accept
                # logprobs/top_logprobs at all — degrade to a coarse
                # binary score off the sampled word instead of failing
                # the whole rerank pass. Loses score granularity for
                # sorting ties, but keeps reranking usable.
                response = client.chat(
                    model=self.cfg.ollama_model,
                    messages=[
                        {"role": "system", "content": _OLLAMA_RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    options=options,
                )
                scores.append(self._binary_fallback_score(response))
                continue

            scores.append(self._score_from_logprobs(response))
        return scores

    def _score_from_logprobs(self, response) -> float:
        """
        Reads the first generated token's top_logprobs off an Ollama
        chat response and converts "yes" vs "no" logprobs into a
        probability via a 2-way softmax. Falls back to the sampled
        content text if logprobs are missing/empty (e.g. a server that
        silently ignores the logprobs request param).
        """
        message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        logprobs = message.get("logprobs") if isinstance(message, dict) else getattr(message, "logprobs", None)
        if not logprobs:
            return self._binary_fallback_score(response)

        top = logprobs[0]
        top_logprobs = top.get("top_logprobs") if isinstance(top, dict) else getattr(top, "top_logprobs", None)
        if not top_logprobs:
            return self._binary_fallback_score(response)

        yes_logprob = _ABSENT_TOKEN_LOGPROB_FLOOR
        no_logprob = _ABSENT_TOKEN_LOGPROB_FLOOR
        for entry in top_logprobs:
            token = entry.get("token") if isinstance(entry, dict) else getattr(entry, "token", "")
            logprob = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
            if logprob is None:
                continue
            normalized = token.strip().lower()
            if normalized == "yes":
                yes_logprob = max(yes_logprob, logprob)
            elif normalized == "no":
                no_logprob = max(no_logprob, logprob)

        # 2-way softmax: P(yes) = exp(yes) / (exp(yes) + exp(no)).
        # Both logprobs are already <= 0 (they're log-probabilities), so
        # this is numerically safe without extra max-subtraction.
        return math.exp(yes_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))

    @staticmethod
    def _binary_fallback_score(response) -> float:
        """Last-resort scoring when logprobs aren't available at all: 1.0
        if the model's sampled answer starts with "yes", else 0.0."""
        message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        return 1.0 if content.strip().lower().startswith("yes") else 0.0

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
