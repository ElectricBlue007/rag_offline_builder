"""
tests/test_ragas_evaluator.py

Unit tests for RagasEvaluator. Since `ragas` is genuinely installed in
this environment, tests use REAL SingleTurnSample/EvaluationDataset
construction (verifying our code builds them correctly against the
real API) while mocking only the parts that need live model calls:
ragas.evaluate() itself, ChatOllama, and FastEmbedEmbeddings.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import RagasEvaluatorConfig
from rag_pipeline.evaluation.eval_dataset import EvalQuestion
from rag_pipeline.evaluation.ragas_evaluator import RagasEvaluator, EvaluationSummary
from rag_pipeline.generation.rag_answerer import RagAnswer


class FakeAnswerer:
    """Stands in for RagAnswerer — same dependency-injection pattern
    already proven in test_rag_answerer.py."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self.calls = []

    def answer(self, query: str) -> RagAnswer:
        self.calls.append(query)
        if query in self._responses:
            return self._responses[query]
        return RagAnswer(
            answer=f"Generated answer for: {query}",
            sources=[{"chunk_id": "a", "source_file": "f.md", "breadcrumb": ""}],
            context_used="[1] Text — f.md\nSome retrieved context.",
        )


class FakeEvaluationResult:
    """Stands in for ragas's EvaluationResult — implements __getitem__
    returning per-sample score lists, matching the real API."""

    def __init__(self, scores: dict):
        self._scores = scores

    def __getitem__(self, key):
        return self._scores[key]


class TestExtractContexts(unittest.TestCase):

    def test_single_chunk_context(self):
        answer = RagAnswer(answer="A", context_used="[1] Text — f.md\nSome content.")
        result = RagasEvaluator._extract_contexts(answer)
        self.assertEqual(len(result), 1)
        self.assertIn("Some content.", result[0])

    def test_multiple_chunks_split_correctly(self):
        context = (
            "[1] Text — f.md\nFirst chunk.\n\n"
            "[2] Table — f.md\nSecond chunk.\n\n"
            "[3] Screenshot — f.md\nThird chunk."
        )
        answer = RagAnswer(answer="A", context_used=context)
        result = RagasEvaluator._extract_contexts(answer)
        self.assertEqual(len(result), 3)
        self.assertIn("First chunk.", result[0])
        self.assertIn("Second chunk.", result[1])
        self.assertIn("Third chunk.", result[2])

    def test_empty_context_returns_empty_list(self):
        answer = RagAnswer(answer="A", context_used="")
        result = RagasEvaluator._extract_contexts(answer)
        self.assertEqual(result, [])

    def test_whitespace_only_context_returns_empty_list(self):
        answer = RagAnswer(answer="A", context_used="   \n  ")
        result = RagasEvaluator._extract_contexts(answer)
        self.assertEqual(result, [])

    def test_real_build_context_output_splits_correctly(self):
        """Integration check against the REAL context_builder.py output
        format, not a hand-written approximation of it."""
        from rag_pipeline.generation.context_builder import build_context
        from rag_pipeline.retrieval.hybrid_retriever import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id="a", text="First.", metadata={"source_file": "f.md"}, rrf_score=0.01),
            RetrievedChunk(chunk_id="b", text="Second.", metadata={"source_file": "f.md"}, rrf_score=0.01),
        ]
        context = build_context(chunks)
        answer = RagAnswer(answer="A", context_used=context)
        result = RagasEvaluator._extract_contexts(answer)
        self.assertEqual(len(result), 2)


class TestRunPipelineOnQuestions(unittest.TestCase):

    def test_returns_one_record_per_question(self):
        answerer = FakeAnswerer()
        evaluator = RagasEvaluator(answerer)
        questions = [
            EvalQuestion(question="Q1", ground_truth="A1"),
            EvalQuestion(question="Q2", ground_truth="A2"),
        ]
        records = evaluator.run_pipeline_on_questions(questions)
        self.assertEqual(len(records), 2)

    def test_record_contains_expected_fields(self):
        answerer = FakeAnswerer()
        evaluator = RagasEvaluator(answerer)
        questions = [EvalQuestion(question="What is a CMI?", ground_truth="A CMI is...", language="en", category="table")]
        records = evaluator.run_pipeline_on_questions(questions)
        r = records[0]
        self.assertEqual(r["question"], "What is a CMI?")
        self.assertEqual(r["ground_truth"], "A CMI is...")
        self.assertEqual(r["language"], "en")
        self.assertEqual(r["category"], "table")
        self.assertIn("answer", r)
        self.assertIn("contexts", r)

    def test_answerer_called_with_each_question(self):
        answerer = FakeAnswerer()
        evaluator = RagasEvaluator(answerer)
        questions = [
            EvalQuestion(question="Q1", ground_truth="A1"),
            EvalQuestion(question="Q2", ground_truth="A2"),
        ]
        evaluator.run_pipeline_on_questions(questions)
        self.assertEqual(answerer.calls, ["Q1", "Q2"])

    def test_no_judge_llm_called_during_pipeline_run(self):
        """This step must be free of judge-LLM cost — verified by the
        fact that no ragas import/model construction happens at all."""
        answerer = FakeAnswerer()
        evaluator = RagasEvaluator(answerer)
        with patch.object(evaluator, "_get_ragas_llm") as mock_get_llm:
            evaluator.run_pipeline_on_questions([EvalQuestion(question="Q", ground_truth="A")])
            mock_get_llm.assert_not_called()

    def test_empty_questions_returns_empty_list(self):
        evaluator = RagasEvaluator(FakeAnswerer())
        self.assertEqual(evaluator.run_pipeline_on_questions([]), [])


class TestBuildSummary(unittest.TestCase):

    def test_computes_mean_per_metric(self):
        evaluator = RagasEvaluator(FakeAnswerer())
        fake_result = FakeEvaluationResult({
            "context_precision": [1.0, 0.5],
            "context_recall": [0.9, 0.7],
            "faithfulness": [1.0, 1.0],
            "answer_relevancy": [0.8, 0.6],
        })
        summary = evaluator._build_summary(fake_result)
        self.assertAlmostEqual(summary.metric_means["context_precision"], 0.75)
        self.assertAlmostEqual(summary.metric_means["context_recall"], 0.8)
        self.assertAlmostEqual(summary.metric_means["faithfulness"], 1.0)
        self.assertAlmostEqual(summary.metric_means["answer_relevancy"], 0.7)

    def test_none_scores_excluded_from_mean(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"])
        )
        fake_result = FakeEvaluationResult({"faithfulness": [1.0, None, 0.5]})
        summary = evaluator._build_summary(fake_result)
        self.assertAlmostEqual(summary.metric_means["faithfulness"], 0.75)

    def test_all_none_scores_produces_none_mean(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"])
        )
        fake_result = FakeEvaluationResult({"faithfulness": [None, None]})
        summary = evaluator._build_summary(fake_result)
        self.assertIsNone(summary.metric_means["faithfulness"])

    def test_missing_metric_key_produces_none(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness", "context_recall"])
        )
        fake_result = FakeEvaluationResult({"faithfulness": [1.0]})  # context_recall missing
        summary = evaluator._build_summary(fake_result)
        self.assertAlmostEqual(summary.metric_means["faithfulness"], 1.0)
        self.assertIsNone(summary.metric_means["context_recall"])

    def test_raw_result_preserved(self):
        evaluator = RagasEvaluator(FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"]))
        fake_result = FakeEvaluationResult({"faithfulness": [1.0]})
        summary = evaluator._build_summary(fake_result)
        self.assertIs(summary.raw_result, fake_result)


class TestEvaluationSummaryPrintSummary(unittest.TestCase):

    def _captured_print(self, summary: EvaluationSummary, cfg=None) -> str:
        with patch("builtins.print") as mock_print:
            summary.print_summary(cfg)
        return "\n".join(str(c.args[0]) if c.args else "" for c in mock_print.call_args_list)

    def test_prints_all_metric_values(self):
        summary = EvaluationSummary(metric_means={
            "context_precision": 0.95, "faithfulness": 0.90,
        })
        output = self._captured_print(summary)
        self.assertIn("0.9500", output)
        self.assertIn("0.9000", output)

    def test_none_value_shown_as_na(self):
        summary = EvaluationSummary(metric_means={"faithfulness": None})
        output = self._captured_print(summary)
        self.assertIn("N/A", output)

    def test_low_context_recall_flags_warning(self):
        summary = EvaluationSummary(metric_means={"context_recall": 0.6})
        output = self._captured_print(summary)
        self.assertIn("⚠️", output)
        self.assertIn("chunking", output.lower())

    def test_high_context_recall_no_warning(self):
        summary = EvaluationSummary(metric_means={"context_recall": 0.95})
        output = self._captured_print(summary)
        self.assertNotIn("⚠️", output)

    def test_low_faithfulness_flags_warning(self):
        summary = EvaluationSummary(metric_means={"faithfulness": 0.7})
        output = self._captured_print(summary)
        self.assertIn("⚠️", output)
        self.assertIn("prompt", output.lower())

    def test_high_faithfulness_no_warning(self):
        summary = EvaluationSummary(metric_means={"faithfulness": 0.99})
        output = self._captured_print(summary)
        self.assertNotIn("⚠️", output)

    def test_custom_thresholds_respected(self):
        cfg = RagasEvaluatorConfig(faithfulness_warning_threshold=0.99)
        summary = EvaluationSummary(metric_means={"faithfulness": 0.95})
        output = self._captured_print(summary, cfg)
        # 0.95 < 0.99 custom threshold → should warn even though it
        # wouldn't under the default 0.85 threshold
        self.assertIn("⚠️", output)


class TestRagasWiring(unittest.TestCase):
    """
    Tests the Ollama wiring (both judge LLM and embeddings) using REAL
    ragas classes (SingleTurnSample, EvaluationDataset) but mocked model
    backends — proving our code constructs the real RAGAS objects
    correctly without needing a live Ollama service or embedding model.
    """

    def test_ragas_llm_uses_chat_ollama_not_openai(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(),
            RagasEvaluatorConfig(judge_llm_model="llama3.2", judge_llm_host="http://localhost:11434"),
        )
        mock_chat_ollama_cls = MagicMock()
        with patch("langchain_community.chat_models.ChatOllama", mock_chat_ollama_cls):
            evaluator._get_ragas_llm()

        mock_chat_ollama_cls.assert_called_once_with(
            model="llama3.2", base_url="http://localhost:11434"
        )

    def test_ragas_llm_created_lazily_once(self):
        evaluator = RagasEvaluator(FakeAnswerer())
        mock_chat_ollama_cls = MagicMock()
        with patch("langchain_community.chat_models.ChatOllama", mock_chat_ollama_cls):
            evaluator._get_ragas_llm()
            evaluator._get_ragas_llm()
        mock_chat_ollama_cls.assert_called_once()

    def test_ragas_embeddings_uses_ollama_not_openai(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(),
            RagasEvaluatorConfig(
                embedding_model="qwen3-embedding8b:latest",
                embedding_host="http://localhost:11434",
            ),
        )
        mock_ollama_embeddings_cls = MagicMock()
        with patch(
            "langchain_community.embeddings.ollama.OllamaEmbeddings",
            mock_ollama_embeddings_cls,
        ):
            evaluator._get_ragas_embeddings()

        mock_ollama_embeddings_cls.assert_called_once_with(
            model="qwen3-embedding8b:latest", base_url="http://localhost:11434"
        )

    def test_ragas_embeddings_created_lazily_once(self):
        evaluator = RagasEvaluator(FakeAnswerer())
        mock_ollama_embeddings_cls = MagicMock()
        with patch(
            "langchain_community.embeddings.ollama.OllamaEmbeddings",
            mock_ollama_embeddings_cls,
        ):
            evaluator._get_ragas_embeddings()
            evaluator._get_ragas_embeddings()
        mock_ollama_embeddings_cls.assert_called_once()

    def test_get_metrics_returns_configured_metrics_only(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"])
        )
        metrics = evaluator._get_metrics()
        self.assertEqual(len(metrics), 1)

    def test_get_metrics_returns_all_four_by_default(self):
        evaluator = RagasEvaluator(FakeAnswerer())
        metrics = evaluator._get_metrics()
        self.assertEqual(len(metrics), 4)


class TestScoreEndToEnd(unittest.TestCase):
    """
    Full score() flow with REAL SingleTurnSample/EvaluationDataset
    construction but a mocked ragas.evaluate() call (which is the only
    part needing live model calls).
    """

    def test_score_builds_real_ragas_dataset_and_calls_evaluate(self):
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"])
        )
        records = [{
            "question": "What is a CMI?",
            "answer": "A CMI is a course indicator.",
            "contexts": ["[1] Text — f.md\nCMI details here."],
            "ground_truth": "A CMI is a course/manoeuvre indicator.",
            "language": "en", "category": "table",
        }]

        fake_result = FakeEvaluationResult({"faithfulness": [0.9]})
        with patch("ragas.evaluate", return_value=fake_result) as mock_evaluate, \
             patch.object(evaluator, "_get_ragas_llm", return_value=MagicMock()), \
             patch.object(evaluator, "_get_ragas_embeddings", return_value=MagicMock()):
            summary = evaluator.score(records)

        mock_evaluate.assert_called_once()
        self.assertAlmostEqual(summary.metric_means["faithfulness"], 0.9)

    def test_evaluate_dataset_contains_correct_sample_data(self):
        """Verifies the REAL SingleTurnSample objects passed to
        ragas.evaluate() carry our record data correctly."""
        evaluator = RagasEvaluator(
            FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"])
        )
        records = [{
            "question": "UNIQUE_QUESTION_MARKER",
            "answer": "UNIQUE_ANSWER_MARKER",
            "contexts": ["UNIQUE_CONTEXT_MARKER"],
            "ground_truth": "UNIQUE_GROUND_TRUTH_MARKER",
            "language": "en", "category": "",
        }]

        captured = {}
        def capture_evaluate(dataset, **kwargs):
            captured["dataset"] = dataset
            return FakeEvaluationResult({"faithfulness": [1.0]})

        with patch("ragas.evaluate", side_effect=capture_evaluate), \
             patch.object(evaluator, "_get_ragas_llm", return_value=MagicMock()), \
             patch.object(evaluator, "_get_ragas_embeddings", return_value=MagicMock()):
            evaluator.score(records)

        sample = captured["dataset"].samples[0]
        self.assertEqual(sample.user_input, "UNIQUE_QUESTION_MARKER")
        self.assertEqual(sample.response, "UNIQUE_ANSWER_MARKER")
        self.assertEqual(sample.retrieved_contexts, ["UNIQUE_CONTEXT_MARKER"])
        self.assertEqual(sample.reference, "UNIQUE_GROUND_TRUTH_MARKER")

    def test_empty_records_returns_none_summary_without_calling_evaluate(self):
        evaluator = RagasEvaluator(FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"]))
        with patch("ragas.evaluate") as mock_evaluate:
            summary = evaluator.score([])
        mock_evaluate.assert_not_called()
        self.assertIsNone(summary.metric_means["faithfulness"])


class TestEvaluateConvenienceMethod(unittest.TestCase):

    def test_evaluate_calls_run_pipeline_then_score(self):
        evaluator = RagasEvaluator(FakeAnswerer(), RagasEvaluatorConfig(metrics=["faithfulness"]))
        questions = [EvalQuestion(question="Q", ground_truth="A")]

        with patch.object(evaluator, "run_pipeline_on_questions", wraps=evaluator.run_pipeline_on_questions) as mock_run, \
             patch.object(evaluator, "score", return_value=EvaluationSummary(metric_means={"faithfulness": 0.9})) as mock_score:
            summary = evaluator.evaluate(questions)

        mock_run.assert_called_once_with(questions)
        mock_score.assert_called_once()
        self.assertEqual(summary.metric_means["faithfulness"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
