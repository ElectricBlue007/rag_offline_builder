"""
evaluation/ragas_evaluator.py

Runs the full RAG pipeline (retrieve -> optionally rerank -> generate)
against a curated set of eval questions, then scores the results with
RAGAS's four core metrics — fully offline, using Ollama as both the
judge LLM and the embedding backend. See RagasEvaluatorConfig in
config.py for why this departs from RAGAS's OpenAI-based defaults.

Split into two explicit steps (run_pipeline_on_questions, then score)
rather than one combined call, so a user can inspect/save the raw
pipeline outputs — question, generated answer, retrieved contexts —
before spending judge-LLM calls on scoring them. Each RAGAS metric
costs its own round of LLM calls, so this separation matters in
practice, not just architecturally.

A note on sequential model loading (see rag_pipeline/model_manager.py):
run_pipeline_on_questions() already gets strict one-model-at-a-time
behaviour for free, since it just calls RagAnswerer.answer() per
question (retrieve -> rerank -> generate), and each of those already
claims/releases its own model via model_manager. score(), however,
hands both the judge LLM AND the embedding model to the `ragas`
library's own evaluate() call, which decides internally which metric
needs which model and can interleave calls to both while scoring a
single question (e.g. answer_relevancy needs both). That internal
interleaving isn't something this module can hook into from the
outside, so score() only claims the judge model up front as a
best-effort default — if your GPU can't hold the judge and embedding
models in VRAM at once, either point judge_llm_model and
embedding_model at the SAME model, or run generation
(run_pipeline_on_questions) and scoring (score) as separate process
invocations with different runtime settings.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from rag_pipeline.config import RagasEvaluatorConfig, RuntimeConfig
from rag_pipeline import model_manager
from rag_pipeline.evaluation.eval_dataset import EvalQuestion
from rag_pipeline.generation.rag_answerer import RagAnswerer, RagAnswer


# Splits build_context()'s "\n\n"-joined chunks back into a list of
# individual context strings — matches the "[N] ..." numbering
# context_builder.py always inserts at the start of each chunk.
_RE_CONTEXT_SPLIT = re.compile(r'\n\n(?=\[\d+\])')


@dataclass
class EvaluationSummary:
    metric_means: dict
    raw_result: object = None  # the ragas EvaluationResult, for to_pandas() etc.

    def print_summary(self, config: Optional[RagasEvaluatorConfig] = None) -> None:
        cfg = config or RagasEvaluatorConfig()
        print("\nRAGAS Evaluation Summary")
        print("=" * 50)
        for name, value in self.metric_means.items():
            if value is None:
                print(f"  {name:20s}: N/A")
                continue

            flag = ""
            if name == "context_recall" and value < cfg.context_recall_warning_threshold:
                flag = (
                    f"  ⚠️  below {cfg.context_recall_warning_threshold} — chunking may be "
                    "splitting context that should stay together"
                )
            elif name == "faithfulness" and value < cfg.faithfulness_warning_threshold:
                flag = (
                    f"  ⚠️  below {cfg.faithfulness_warning_threshold} — check the prompt "
                    "template; the answer may not be fully grounded in context"
                )
            print(f"  {name:20s}: {value:.4f}{flag}")
        print("=" * 50)


class RagasEvaluator:
    """
    Usage:
        evaluator = RagasEvaluator(rag_answerer, config)
        questions = load_eval_questions("evaluation/sample_eval_questions.json")
        records = evaluator.run_pipeline_on_questions(questions)   # cheap, no judge LLM yet
        summary = evaluator.score(records)                        # spends judge-LLM calls
        summary.print_summary()

        # or, in one call:
        summary = evaluator.evaluate(questions)
    """

    def __init__(
        self,
        answerer: RagAnswerer,
        config: Optional[RagasEvaluatorConfig] = None,
        runtime: Optional[RuntimeConfig] = None,
    ):
        self.answerer = answerer
        self.cfg = config or RagasEvaluatorConfig()
        self.runtime = runtime or RuntimeConfig()
        self._ragas_llm = None
        self._ragas_embeddings = None

    # ── Step 1: run the pipeline (retrieval + generation), no judge LLM yet ──

    def run_pipeline_on_questions(self, questions: list[EvalQuestion]) -> list[dict]:
        """
        Run the RAG pipeline on each question, collecting the
        (question, answer, contexts, ground_truth) records RAGAS needs.
        Does NOT call the judge LLM — that only happens in score().
        """
        records = []
        for eq in questions:
            result: RagAnswer = self.answerer.answer(eq.question)
            records.append({
                "question": eq.question,
                "answer": result.answer,
                "contexts": self._extract_contexts(result),
                "ground_truth": eq.ground_truth,
                "language": eq.language,
                "category": eq.category,
            })
        return records

    @staticmethod
    def _extract_contexts(rag_answer: RagAnswer) -> list[str]:
        """
        RAGAS wants retrieved_contexts as a list of separate strings,
        not context_builder.py's single joined block — split back
        apart using the "[N] " numbering boundary.
        """
        text = (rag_answer.context_used or "").strip()
        if not text:
            return []
        parts = _RE_CONTEXT_SPLIT.split(text)
        return [p for p in parts if p.strip()]

    # ── Step 2: score with RAGAS (spends judge-LLM calls) ───────────────────

    def score(self, records: list[dict]) -> EvaluationSummary:
        try:
            from ragas import evaluate, EvaluationDataset, SingleTurnSample
            from ragas.run_config import RunConfig
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ragas"
            ) from e

        if not records:
            return EvaluationSummary(metric_means={m: None for m in self.cfg.metrics})

        # Best-effort claim (see module docstring's note on why this
        # can't be a hard guarantee once ragas.evaluate() takes over):
        # if judge and embedding models differ, this at least ensures
        # whatever was active from run_pipeline_on_questions() (the
        # generation model) is unloaded before scoring starts, and the
        # judge model gets first claim on VRAM.
        model_manager.ensure_active(self.cfg.judge_llm_host, self.cfg.judge_llm_model, self.runtime)

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in records
        ]
        dataset = EvaluationDataset(samples=samples)

        # RAGAS defaults to max_workers=16, firing 16 concurrent judge-LLM
        # calls. A single local Ollama instance serializes them instead,
        # so most requests blow past the default timeout and come back as
        # TimeoutError — which is what produced the all-`nan` summary.
        # low max_workers + a generous timeout matches what one local
        # Ollama process can actually keep up with.
        result = evaluate(
            dataset,
            metrics=self._get_metrics(),
            llm=self._get_ragas_llm(),
            embeddings=self._get_ragas_embeddings(),
            run_config=RunConfig(max_workers=self.cfg.judge_max_workers, timeout=self.cfg.judge_timeout_seconds),
        )

        # Best-effort cleanup: force-unload everything on both hosts now
        # that scoring is done, since ragas may have loaded the
        # embedding model alongside the judge model partway through
        # (see the module docstring).
        model_manager.release_all(self.cfg.judge_llm_host, self.runtime)
        if self.cfg.embedding_host != self.cfg.judge_llm_host:
            model_manager.release_all(self.cfg.embedding_host, self.runtime)

        return self._build_summary(result)

    def evaluate(self, questions: list[EvalQuestion]) -> EvaluationSummary:
        """Convenience: run_pipeline_on_questions() + score() in one call."""
        records = self.run_pipeline_on_questions(questions)
        return self.score(records)

    # ── RAGAS wiring: Ollama for both LLM and embeddings, never OpenAI ──────

    def _get_ragas_llm(self):
        if self._ragas_llm is not None:
            return self._ragas_llm
        try:
            from langchain_community.chat_models import ChatOllama
            from ragas.llms import LangchainLLMWrapper
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ragas langchain-community"
            ) from e

        chat_model = ChatOllama(
            model=self.cfg.judge_llm_model,
            base_url=self.cfg.judge_llm_host,
            num_ctx=self.cfg.judge_num_ctx,
        )
        self._ragas_llm = LangchainLLMWrapper(chat_model)
        return self._ragas_llm

    def _get_ragas_embeddings(self):
        if self._ragas_embeddings is not None:
            return self._ragas_embeddings
        try:
            from langchain_community.embeddings.ollama import OllamaEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ragas langchain-community ollama"
            ) from e

        embed_model = OllamaEmbeddings(
            model=self.cfg.embedding_model,
            base_url=self.cfg.embedding_host,
            num_ctx=self.cfg.embedding_num_ctx,
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(embed_model)
        return self._ragas_embeddings

    def _get_metrics(self) -> list:
        from ragas.metrics import (
            context_precision, context_recall, faithfulness, answer_relevancy,
        )
        available = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }
        return [available[name] for name in self.cfg.metrics if name in available]

    def _build_summary(self, result) -> EvaluationSummary:
        """
        result supports __getitem__(metric_name) -> list[float] (one
        score per sample). Compute means ourselves rather than relying
        on ragas's internal/private repr formatting.
        """
        metric_means = {}
        for name in self.cfg.metrics:
            try:
                scores = result[name]
                valid = [s for s in scores if s is not None]
                metric_means[name] = sum(valid) / len(valid) if valid else None
            except (KeyError, TypeError):
                metric_means[name] = None
        return EvaluationSummary(metric_means=metric_means, raw_result=result)
