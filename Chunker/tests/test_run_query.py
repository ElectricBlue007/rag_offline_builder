"""
tests/test_run_query.py

Tests for run_query.py's argument parsing and result formatting.
build_retriever() itself (which wires real ChromaDB/BM25/FastEmbed
components together) is exercised only at the "does it raise the
right error" level here — the actual retrieval logic is already
covered thoroughly by test_hybrid_retriever.py.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "run_query", Path(__file__).parent.parent / "run_query.py"
)
run_query = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_query)

from Chunker.retrieval.hybrid_retriever import RetrievedChunk


class TestArgParsing(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["run_query.py"] + argv):
            return run_query.parse_args()

    def test_defaults(self):
        args = self._parse([])
        self.assertIsNone(args.query)
        self.assertEqual(args.chroma_dir, "./chroma_db")
        self.assertEqual(args.collection, "rag_docs")
        self.assertEqual(args.bm25_index, "./bm25_index.pkl")
        self.assertEqual(args.model, "qwen3-embedding:8b")
        self.assertEqual(args.n_results, 5)
        self.assertFalse(args.rerank)
        self.assertEqual(args.rerank_model, "Qwen/Qwen3-Reranker-4B")
        self.assertEqual(args.candidate_pool, 10)

    def test_positional_query(self):
        args = self._parse(["What is a CMI?"])
        self.assertEqual(args.query, "What is a CMI?")

    def test_no_query_is_none(self):
        args = self._parse(["--chroma-dir", "/custom"])
        self.assertIsNone(args.query)

    def test_custom_chroma_dir(self):
        args = self._parse(["query", "--chroma-dir", "/custom/chroma"])
        self.assertEqual(args.chroma_dir, "/custom/chroma")

    def test_custom_collection(self):
        args = self._parse(["query", "--collection", "my_docs"])
        self.assertEqual(args.collection, "my_docs")

    def test_custom_bm25_index_path(self):
        args = self._parse(["query", "--bm25-index", "/custom/index.pkl"])
        self.assertEqual(args.bm25_index, "/custom/index.pkl")

    def test_custom_n_results(self):
        args = self._parse(["query", "--n-results", "10"])
        self.assertEqual(args.n_results, 10)
        self.assertIsInstance(args.n_results, int)

    def test_rerank_flag(self):
        args = self._parse(["query", "--rerank"])
        self.assertTrue(args.rerank)

    def test_custom_rerank_model(self):
        args = self._parse(["query", "--rerank-model", "custom/model"])
        self.assertEqual(args.rerank_model, "custom/model")

    def test_custom_candidate_pool(self):
        args = self._parse(["query", "--candidate-pool", "20"])
        self.assertEqual(args.candidate_pool, 20)
        self.assertIsInstance(args.candidate_pool, int)


class TestFormatResults(unittest.TestCase):

    def test_empty_results_message(self):
        result = run_query.format_results([])
        self.assertEqual(result, "No results found.")

    def test_single_result_includes_score(self):
        chunk = RetrievedChunk(
            chunk_id="a", text="Some content here.",
            metadata={"source_file": "Binder2.md", "breadcrumb": "### System reports"},
            rrf_score=0.0328, dense_rank=1, sparse_rank=2,
        )
        result = run_query.format_results([chunk])
        self.assertIn("0.0328", result)
        self.assertIn("Binder2.md", result)
        self.assertIn("System reports", result)
        self.assertIn("Some content here.", result)

    def test_dense_and_sparse_ranks_shown(self):
        chunk = RetrievedChunk(
            chunk_id="a", text="Content.", metadata={}, rrf_score=0.01,
            dense_rank=3, sparse_rank=None,
        )
        result = run_query.format_results([chunk])
        self.assertIn("dense_rank=3", result)
        self.assertIn("sparse_rank=None", result)

    def test_multiple_results_numbered(self):
        chunks = [
            RetrievedChunk(chunk_id=str(i), text=f"Content {i}", metadata={},
                          rrf_score=0.01 * (10 - i))
            for i in range(3)
        ]
        result = run_query.format_results(chunks)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertIn("[3]", result)

    def test_missing_metadata_fields_default_gracefully(self):
        chunk = RetrievedChunk(chunk_id="a", text="Content.", metadata={}, rrf_score=0.01)
        result = run_query.format_results([chunk])  # must not raise
        self.assertIn("?", result)  # source_file default

    def test_long_text_truncated_to_200_chars(self):
        long_text = "A" * 500
        chunk = RetrievedChunk(chunk_id="a", text=long_text, metadata={}, rrf_score=0.01)
        result = run_query.format_results([chunk])
        # Original 500-char run of A's must not appear in full
        self.assertNotIn("A" * 500, result)

    def test_newlines_in_text_replaced(self):
        chunk = RetrievedChunk(
            chunk_id="a", text="Line one.\nLine two.", metadata={}, rrf_score=0.01
        )
        result = run_query.format_results([chunk])
        # The preview line itself should have no raw newline breaking formatting
        preview_line = [l for l in result.split("\n") if "Line one" in l][0]
        self.assertIn("Line two", preview_line)


class TestBuildRetriever(unittest.TestCase):

    def test_missing_bm25_index_raises_file_not_found(self):
        args = self._make_args(bm25_index="/tmp/definitely_not_there_12345.pkl")
        with self.assertRaises(FileNotFoundError):
            run_query.build_retriever(args)

    def test_final_n_results_equals_n_results_when_not_reranking(self):
        import tempfile
        from Chunker.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), n_results=7, rerank=False, candidate_pool=10)
            retriever = run_query.build_retriever(args)
            self.assertEqual(retriever.cfg.final_n_results, 7)

    def test_final_n_results_equals_candidate_pool_when_reranking(self):
        """When --rerank is set, retrieval must fetch the WIDER candidate
        pool, not just n_results — the reranker narrows it down later."""
        import tempfile
        from Chunker.storage.bm25_index import BM25Index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            idx = BM25Index()
            idx.build([{"chunk_id": "a", "contextualized_text": "content"}])
            idx.save(str(path))

            args = self._make_args(bm25_index=str(path), n_results=5, rerank=True, candidate_pool=15)
            retriever = run_query.build_retriever(args)
            self.assertEqual(retriever.cfg.final_n_results, 15)

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
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args


class TestBuildReranker(unittest.TestCase):

    def _make_args(self, **overrides):
        class Args:
            rerank_model = "Qwen/Qwen3-Reranker-4B"
            n_results = 5
        args = Args()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_returns_cross_encoder_reranker(self):
        from Chunker.retrieval.reranker import CrossEncoderReranker
        args = self._make_args()
        reranker = run_query.build_reranker(args)
        self.assertIsInstance(reranker, CrossEncoderReranker)

    def test_model_name_propagated(self):
        args = self._make_args(rerank_model="custom/model")
        reranker = run_query.build_reranker(args)
        self.assertEqual(reranker.cfg.model_name, "custom/model")

    def test_final_n_results_propagated_from_n_results(self):
        args = self._make_args(n_results=3)
        reranker = run_query.build_reranker(args)
        self.assertEqual(reranker.cfg.final_n_results, 3)


class TestFormatResultsWithReranking(unittest.TestCase):
    """format_results must handle RerankedChunk (post-reranking) as
    well as plain RetrievedChunk, via duck-typing on rerank_score."""

    def test_reranked_chunk_shows_rerank_score(self):
        from Chunker.retrieval.reranker import RerankedChunk
        chunk = RerankedChunk(
            chunk_id="a", text="Content.", metadata={"source_file": "Binder2.md"},
            rerank_score=0.87, original_rrf_score=0.0164, original_rank=2,
        )
        result = run_query.format_results([chunk])
        self.assertIn("rerank_score=0.8700", result)
        self.assertIn("rrf_score=0.0164", result)
        self.assertIn("original rank #3", result)

    def test_retrieved_chunk_shows_rrf_score_not_rerank(self):
        chunk = RetrievedChunk(
            chunk_id="a", text="Content.", metadata={}, rrf_score=0.02,
            dense_rank=1, sparse_rank=2,
        )
        result = run_query.format_results([chunk])
        self.assertIn("score=0.0200", result)
        self.assertNotIn("rerank_score", result)

    def test_mixed_list_not_expected_but_handles_each_independently(self):
        """Defensive: even an unusual mixed list should format each item
        according to its own type without crashing."""
        from Chunker.retrieval.reranker import RerankedChunk
        retrieved = RetrievedChunk(chunk_id="a", text="A", metadata={}, rrf_score=0.01)
        reranked = RerankedChunk(
            chunk_id="b", text="B", metadata={}, rerank_score=0.9,
            original_rrf_score=0.01, original_rank=0,
        )
        result = run_query.format_results([retrieved, reranked])
        self.assertIn("score=0.0100", result)
        self.assertIn("rerank_score=0.9000", result)


class TestMainErrorHandling(unittest.TestCase):

    def test_main_exits_gracefully_on_missing_bm25_index(self):
        with patch.object(sys, "argv", [
            "run_query.py", "some query",
            "--bm25-index", "/tmp/definitely_not_there_98765.pkl",
        ]):
            with self.assertRaises(SystemExit) as cm:
                run_query.main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
