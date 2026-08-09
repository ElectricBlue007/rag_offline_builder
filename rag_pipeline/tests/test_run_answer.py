"""
tests/test_run_answer.py

Tests for run_answer.py's argument parsing, builder wiring, and answer
formatting. build_answerer() itself is exercised at the "does it wire
components together correctly" level using mocks — the actual
retrieval/generation logic is already covered by test_hybrid_retriever.py,
test_reranker.py, test_llm_client.py, and test_rag_answerer.py.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "run_answer", Path(__file__).parent.parent / "run_answer.py"
)
run_answer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_answer)

from rag_pipeline.generation.rag_answerer import RagAnswer


class TestArgParsing(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["run_answer.py"] + argv):
            return run_answer.parse_args()

    def test_defaults(self):
        args = self._parse([])
        self.assertIsNone(args.query)
        self.assertEqual(args.chroma_dir, "./chroma_db")
        self.assertEqual(args.collection, "rag_docs")
        self.assertEqual(args.bm25_index, "./bm25_index.pkl")
        # Compare against the same settings.yaml-derived object the CLI
        # itself defaults from, not a literal — so this test verifies
        # "arg parsing wires in settings.yaml correctly" and keeps
        # passing no matter which model settings.yaml names.
        self.assertEqual(args.model, run_answer._S.embed_generator.model_name)
        self.assertEqual(args.n_results, 5)
        self.assertFalse(args.rerank)
        self.assertEqual(args.rerank_model, "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(args.candidate_pool, 20)
        self.assertEqual(args.llm_model, run_answer._S.llm_client.model)
        self.assertEqual(args.llm_host, "http://localhost:11434")
        self.assertEqual(args.temperature, 0.1)
        self.assertFalse(args.no_sources)

    def test_positional_query(self):
        args = self._parse(["What is a CMI?"])
        self.assertEqual(args.query, "What is a CMI?")

    def test_custom_llm_model(self):
        args = self._parse(["query", "--llm-model", "mistral"])
        self.assertEqual(args.llm_model, "mistral")

    def test_custom_llm_host(self):
        args = self._parse(["query", "--llm-host", "http://remote:11434"])
        self.assertEqual(args.llm_host, "http://remote:11434")

    def test_custom_temperature(self):
        args = self._parse(["query", "--temperature", "0.5"])
        self.assertEqual(args.temperature, 0.5)

    def test_rerank_flag(self):
        args = self._parse(["query", "--rerank"])
        self.assertTrue(args.rerank)

    def test_no_sources_flag(self):
        args = self._parse(["query", "--no-sources"])
        self.assertTrue(args.no_sources)

    def test_custom_n_results(self):
        args = self._parse(["query", "--n-results", "8"])
        self.assertEqual(args.n_results, 8)


class TestBuildRetriever(unittest.TestCase):

    def _make_args(self, **overrides):
        class Args:
            chroma_dir = "./chroma_db"
            collection = "rag_docs"
            bm25_index = "./bm25_index.pkl"
            model = "qwen3-embedding8b:latest"
            n_results = 5
            rerank = False
            rerank_model = "Qwen/Qwen3-Reranker-0.6B"
            candidate_pool = 20
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_missing_bm25_index_raises(self):
        args = self._make_args(bm25_index="/tmp/definitely_not_there_answer_test.pkl")
        with self.assertRaises(FileNotFoundError):
            run_answer.build_retriever(args)

    def test_final_n_results_widens_when_reranking(self):
        import tempfile
        from rag_pipeline.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), n_results=5, rerank=True, candidate_pool=15)
            retriever = run_answer.build_retriever(args)
            self.assertEqual(retriever.cfg.final_n_results, 15)

    def test_final_n_results_equals_n_results_without_reranking(self):
        import tempfile
        from rag_pipeline.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), n_results=7, rerank=False)
            retriever = run_answer.build_retriever(args)
            self.assertEqual(retriever.cfg.final_n_results, 7)


class TestBuildReranker(unittest.TestCase):

    def _make_args(self, **overrides):
        class Args:
            rerank_model = "Qwen/Qwen3-Reranker-0.6B"
            n_results = 5
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_returns_cross_encoder_reranker(self):
        from rag_pipeline.retrieval.reranker import CrossEncoderReranker
        reranker = run_answer.build_reranker(self._make_args())
        self.assertIsInstance(reranker, CrossEncoderReranker)

    def test_final_n_results_matches_n_results(self):
        reranker = run_answer.build_reranker(self._make_args(n_results=3))
        self.assertEqual(reranker.cfg.final_n_results, 3)


class TestBuildAnswerer(unittest.TestCase):

    def _make_args(self, **overrides):
        class Args:
            chroma_dir = "./chroma_db"
            collection = "rag_docs"
            bm25_index = "./bm25_index.pkl"
            model = "qwen3-embedding8b:latest"
            n_results = 5
            rerank = False
            rerank_model = "Qwen/Qwen3-Reranker-0.6B"
            candidate_pool = 20
            llm_model = "Qwen-AgentWorld-35B-A3B:latest"
            llm_host = "http://localhost:11434"
            temperature = 0.1
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_missing_bm25_index_raises(self):
        args = self._make_args(bm25_index="/tmp/definitely_not_there_build_answerer.pkl")
        with self.assertRaises(FileNotFoundError):
            run_answer.build_answerer(args)

    def test_returns_rag_answerer_with_no_reranker_by_default(self):
        import tempfile
        from rag_pipeline.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), rerank=False)
            answerer = run_answer.build_answerer(args)
            self.assertIsNone(answerer.reranker)

    def test_returns_rag_answerer_with_reranker_when_requested(self):
        import tempfile
        from rag_pipeline.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), rerank=True)
            answerer = run_answer.build_answerer(args)
            self.assertIsNotNone(answerer.reranker)

    def test_llm_client_configured_with_args(self):
        import tempfile
        from rag_pipeline.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), llm_model="mistral", temperature=0.5)
            answerer = run_answer.build_answerer(args)
            self.assertEqual(answerer.llm_client.cfg.model, "mistral")
            self.assertEqual(answerer.llm_client.cfg.temperature, 0.5)


class TestPrintAnswer(unittest.TestCase):

    def test_prints_answer_text(self, ):
        result = RagAnswer(answer="The CMI is a track correction.", sources=[])
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=True)
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertIn("The CMI is a track correction.", printed)

    def test_sources_printed_when_enabled(self):
        result = RagAnswer(
            answer="Answer.",
            sources=[{"chunk_id": "a", "source_file": "Binder2.md", "breadcrumb": "### System reports"}],
        )
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=True)
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertIn("Binder2.md", printed)
        self.assertIn("System reports", printed)

    def test_sources_omitted_when_disabled(self):
        result = RagAnswer(
            answer="Answer.",
            sources=[{"chunk_id": "a", "source_file": "Binder2.md", "breadcrumb": ""}],
        )
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=False)
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertNotIn("Sources:", printed)

    def test_no_sources_section_when_sources_empty(self):
        result = RagAnswer(answer="Answer.", sources=[])
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=True)
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertNotIn("Sources:", printed)

    def test_sources_numbered_matching_citation_order(self):
        """
        Sources are printed in order, numbered [1], [2]... to match the
        bracketed citation numbers context_builder.py assigned to each
        one. Two sources sharing the same file/breadcrumb must BOTH be
        shown — deduplicating would shift the numbering out of sync
        with whatever the model actually cited in the answer text.
        """
        result = RagAnswer(
            answer="Answer cites [1] and also [2].",
            sources=[
                {"chunk_id": "a", "source_file": "Binder2.md", "breadcrumb": "### System reports"},
                {"chunk_id": "b", "source_file": "Binder2.md", "breadcrumb": "### System reports"},
            ],
        )
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=True)
        printed = "\n".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertEqual(printed.count("Binder2.md"), 2)
        self.assertIn("[1]", printed)
        self.assertIn("[2]", printed)

    def test_root_breadcrumb_not_shown(self):
        result = RagAnswer(
            answer="Answer.",
            sources=[{"chunk_id": "a", "source_file": "f.md", "breadcrumb": "(root)"}],
        )
        with patch("builtins.print") as mock_print:
            run_answer.print_answer(result, show_sources=True)
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertNotIn("(root)", printed)


class TestMainErrorHandling(unittest.TestCase):

    def test_main_exits_gracefully_on_missing_bm25_index(self):
        with patch.object(sys, "argv", [
            "run_answer.py", "some query",
            "--bm25-index", "/tmp/definitely_not_there_main_test.pkl",
        ]):
            with self.assertRaises(SystemExit) as cm:
                run_answer.main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
