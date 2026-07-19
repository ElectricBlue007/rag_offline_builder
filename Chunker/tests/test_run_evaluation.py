"""
tests/test_run_evaluation.py

Tests for run_evaluation.py's argument parsing and orchestration
wiring, using mocks for the heavy components (matching the pattern
established for run_answer.py's tests).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "run_evaluation", Path(__file__).parent.parent / "run_evaluation.py"
)
run_evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_evaluation)


class TestArgParsing(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["run_evaluation.py"] + argv):
            return run_evaluation.parse_args()

    def test_defaults(self):
        args = self._parse([])
        self.assertTrue(args.questions.endswith("sample_eval_questions.json"))
        self.assertEqual(args.chroma_dir, "./chroma_db")
        self.assertEqual(args.bm25_index, "./bm25_index.pkl")
        self.assertEqual(args.n_results, 5)
        self.assertFalse(args.rerank)
        self.assertEqual(args.llm_model, "llama3.2")
        self.assertEqual(args.judge_model, "llama3.2")
        self.assertEqual(args.metrics,
                         "context_precision,context_recall,faithfulness,answer_relevancy")
        self.assertIsNone(args.save_results)
        self.assertFalse(args.quiet)

    def test_custom_questions_path(self):
        args = self._parse(["--questions", "/my/questions.json"])
        self.assertEqual(args.questions, "/my/questions.json")

    def test_custom_judge_model(self):
        args = self._parse(["--judge-model", "mistral"])
        self.assertEqual(args.judge_model, "mistral")

    def test_custom_judge_host(self):
        args = self._parse(["--judge-host", "http://remote:11434"])
        self.assertEqual(args.judge_host, "http://remote:11434")

    def test_custom_metrics_subset(self):
        args = self._parse(["--metrics", "faithfulness,answer_relevancy"])
        self.assertEqual(args.metrics, "faithfulness,answer_relevancy")

    def test_rerank_flag(self):
        args = self._parse(["--rerank"])
        self.assertTrue(args.rerank)

    def test_save_results_path(self):
        args = self._parse(["--save-results", "/tmp/results.json"])
        self.assertEqual(args.save_results, "/tmp/results.json")

    def test_quiet_flag(self):
        args = self._parse(["-q"])
        self.assertTrue(args.quiet)


class TestBuildAnswerer(unittest.TestCase):

    def _make_args(self, **overrides):
        class Args:
            chroma_dir = "./chroma_db"
            collection = "rag_docs"
            bm25_index = "./bm25_index.pkl"
            model = "qwen3-embedding:8b"
            n_results = 5
            rerank = False
            rerank_model = "Qwen/Qwen3-Reranker-4B"
            candidate_pool = 10
            llm_model = "llama3.2"
            llm_host = "http://localhost:11434"
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_missing_bm25_index_raises(self):
        args = self._make_args(bm25_index="/tmp/definitely_not_there_eval_test.pkl")
        with self.assertRaises(FileNotFoundError):
            run_evaluation.build_answerer(args)

    def test_returns_answerer_without_reranker_by_default(self):
        import tempfile
        from Chunker.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path))
            answerer = run_evaluation.build_answerer(args)
            self.assertIsNone(answerer.reranker)

    def test_returns_answerer_with_reranker_when_requested(self):
        import tempfile
        from Chunker.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), rerank=True)
            answerer = run_evaluation.build_answerer(args)
            self.assertIsNotNone(answerer.reranker)


class TestMainOrchestration(unittest.TestCase):

    def test_missing_questions_file_exits_with_error(self):
        with patch.object(sys, "argv", [
            "run_evaluation.py", "--questions", "/tmp/definitely_not_there_questions.json",
        ]):
            with self.assertRaises(SystemExit) as cm:
                run_evaluation.main()
            self.assertEqual(cm.exception.code, 1)

    def test_missing_bm25_index_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            questions_path = Path(tmp) / "q.json"
            questions_path.write_text(json.dumps([
                {"question": "Q", "ground_truth": "A", "language": "en", "category": ""}
            ]))
            with patch.object(sys, "argv", [
                "run_evaluation.py",
                "--questions", str(questions_path),
                "--bm25-index", "/tmp/definitely_not_there_main.pkl",
            ]):
                with self.assertRaises(SystemExit) as cm:
                    run_evaluation.main()
                self.assertEqual(cm.exception.code, 1)

    def test_full_flow_with_mocked_scoring(self):
        """
        End-to-end main() flow with a real BM25 index (pure Python) and
        a mocked evaluator.score() call (avoiding the need for a live
        Ollama service / judge LLM in tests).
        """
        from Chunker.storage.bm25_index import BM25Index
        from Chunker.evaluation.ragas_evaluator import EvaluationSummary

        with tempfile.TemporaryDirectory() as tmp:
            bm25_path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "CMI content"}])
            idx.save(str(bm25_path))

            questions_path = Path(tmp) / "q.json"
            questions_path.write_text(json.dumps([
                {"question": "What is a CMI?", "ground_truth": "A CMI is...",
                 "language": "en", "category": "table"}
            ]))

            results_path = Path(tmp) / "results.json"

            fake_summary = EvaluationSummary(metric_means={"faithfulness": 0.9})

            with patch.object(sys, "argv", [
                "run_evaluation.py",
                "--questions", str(questions_path),
                "--bm25-index", str(bm25_path),
                "--metrics", "faithfulness",
                "--save-results", str(results_path),
                "--quiet",
            ]), \
            patch("Chunker.evaluation.ragas_evaluator.RagasEvaluator.run_pipeline_on_questions",
                  return_value=[{"question": "What is a CMI?", "answer": "Answer.",
                                 "contexts": ["ctx"], "ground_truth": "A CMI is...",
                                 "language": "en", "category": "table"}]), \
            patch("Chunker.evaluation.ragas_evaluator.RagasEvaluator.score",
                  return_value=fake_summary):

                with self.assertRaises(SystemExit) as cm:
                    run_evaluation.main()
                self.assertEqual(cm.exception.code, 0)

            self.assertTrue(results_path.exists())
            saved = json.loads(results_path.read_text())
            self.assertEqual(saved["metric_means"]["faithfulness"], 0.9)
            self.assertEqual(len(saved["records"]), 1)

    def test_empty_questions_file_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            questions_path = Path(tmp) / "empty.json"
            questions_path.write_text("[]")
            with patch.object(sys, "argv", [
                "run_evaluation.py", "--questions", str(questions_path),
            ]):
                with self.assertRaises(SystemExit) as cm:
                    run_evaluation.main()
                self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
