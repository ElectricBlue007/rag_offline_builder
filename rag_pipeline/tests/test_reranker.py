"""
tests/test_reranker.py

Unit tests for CrossEncoderReranker — using a mocked sentence-
transformers CrossEncoder, since neither sentence-transformers nor
torch is installed in this environment (matching the same lazy-import
+ mock-based testing pattern already used for ChromaDB and Ollama
elsewhere in this pipeline).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import RerankerConfig
from rag_pipeline.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from rag_pipeline.retrieval.hybrid_retriever import RetrievedChunk


def make_candidate(chunk_id: str, text: str, rrf_score: float = 0.01, metadata=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, metadata=metadata or {}, rrf_score=rrf_score,
    )


class TestCrossEncoderRerankerBasics(unittest.TestCase):

    def _make_reranker_with_scores(self, scores: list[float]) -> CrossEncoderReranker:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        return reranker

    def test_empty_query_returns_empty(self):
        reranker = self._make_reranker_with_scores([])
        candidates = [make_candidate("a", "Content A")]
        self.assertEqual(reranker.rerank("", candidates), [])

    def test_whitespace_query_returns_empty(self):
        reranker = self._make_reranker_with_scores([])
        candidates = [make_candidate("a", "Content A")]
        self.assertEqual(reranker.rerank("   ", candidates), [])

    def test_empty_candidates_returns_empty(self):
        reranker = self._make_reranker_with_scores([])
        self.assertEqual(reranker.rerank("some query", []), [])

    def test_returns_reranked_chunk_instances(self):
        reranker = self._make_reranker_with_scores([0.9])
        candidates = [make_candidate("a", "Content A")]
        results = reranker.rerank("query", candidates)
        for r in results:
            self.assertIsInstance(r, RerankedChunk)

    def test_reorders_by_cross_encoder_score_not_original_order(self):
        """The whole point of reranking: cross-encoder scores can flip
        the order the candidates arrived in."""
        reranker = self._make_reranker_with_scores([0.1, 0.9, 0.5])
        candidates = [
            make_candidate("first", "Content first"),
            make_candidate("second", "Content second"),
            make_candidate("third", "Content third"),
        ]
        results = reranker.rerank("query", candidates)
        # "second" had the highest cross-encoder score (0.9) despite
        # arriving in the middle of the original candidate order.
        self.assertEqual(results[0].chunk_id, "second")
        self.assertEqual(results[1].chunk_id, "third")
        self.assertEqual(results[2].chunk_id, "first")

    def test_final_n_results_truncates(self):
        cfg = RerankerConfig(final_n_results=2)
        reranker = CrossEncoderReranker(cfg)
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3]
        reranker._model = mock_model

        candidates = [make_candidate(str(i), f"Content {i}") for i in range(4)]
        results = reranker.rerank("query", candidates)
        self.assertEqual(len(results), 2)

    def test_original_rank_preserved(self):
        reranker = self._make_reranker_with_scores([0.1, 0.9])
        candidates = [
            make_candidate("first", "Content first"),
            make_candidate("second", "Content second"),
        ]
        results = reranker.rerank("query", candidates)
        # "second" was originally at position 1 (0-indexed), even though
        # it now ranks first after reranking.
        top = next(r for r in results if r.chunk_id == "second")
        self.assertEqual(top.original_rank, 1)

    def test_original_rrf_score_preserved(self):
        reranker = self._make_reranker_with_scores([0.5])
        candidates = [make_candidate("a", "Content A", rrf_score=0.0328)]
        results = reranker.rerank("query", candidates)
        self.assertAlmostEqual(results[0].original_rrf_score, 0.0328)

    def test_metadata_preserved(self):
        reranker = self._make_reranker_with_scores([0.5])
        candidates = [make_candidate("a", "Content A", metadata={"source_file": "Binder2.md"})]
        results = reranker.rerank("query", candidates)
        self.assertEqual(results[0].metadata["source_file"], "Binder2.md")

    def test_rerank_score_is_float(self):
        reranker = self._make_reranker_with_scores([0.789])
        candidates = [make_candidate("a", "Content A")]
        results = reranker.rerank("query", candidates)
        self.assertIsInstance(results[0].rerank_score, float)
        self.assertAlmostEqual(results[0].rerank_score, 0.789)


class TestCrossEncoderRerankerPairConstruction(unittest.TestCase):

    def test_predict_called_with_query_text_pairs(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.5]
        reranker._model = mock_model

        candidates = [
            make_candidate("a", "First content"),
            make_candidate("b", "Second content"),
        ]
        reranker.rerank("my query", candidates)

        call_args = mock_model.predict.call_args[0][0]
        self.assertEqual(call_args, [("my query", "First content"), ("my query", "Second content")])

    def test_dict_candidates_supported(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        reranker._model = mock_model

        candidates = [{"chunk_id": "a", "text": "Dict content", "metadata": {}, "rrf_score": 0.01}]
        results = reranker.rerank("query", candidates)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(results[0].text, "Dict content")


class TestCrossEncoderRerankerModelLoading(unittest.TestCase):

    def test_import_error_when_sentence_transformers_missing(self):
        reranker = CrossEncoderReranker()
        candidates = [make_candidate("a", "Content A")]
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(ImportError):
                reranker.rerank("query", candidates)

    def test_model_loaded_lazily_once(self):
        reranker = CrossEncoderReranker()
        mock_module = MagicMock()
        mock_cross_encoder_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.5]
        mock_cross_encoder_cls.return_value = mock_instance
        mock_module.CrossEncoder = mock_cross_encoder_cls

        candidates = [make_candidate("a", "Content A")]
        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            reranker.rerank("query one", candidates)
            reranker.rerank("query two", candidates)

        mock_cross_encoder_cls.assert_called_once()

    def test_model_name_and_max_length_passed_to_constructor(self):
        cfg = RerankerConfig(
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=256
        )
        reranker = CrossEncoderReranker(cfg)
        mock_module = MagicMock()
        mock_cross_encoder_cls = MagicMock()
        mock_cross_encoder_cls.return_value.predict.return_value = [0.5]
        mock_module.CrossEncoder = mock_cross_encoder_cls

        candidates = [make_candidate("a", "Content A")]
        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            reranker.rerank("query", candidates)

        call_args, call_kwargs = mock_cross_encoder_cls.call_args
        self.assertEqual(call_args[0], "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.assertEqual(call_kwargs["max_length"], 256)


class TestCrossEncoderRerankerRealFixtures(unittest.TestCase):
    """
    Integration test using real chunk data (from the pipeline) as
    reranker candidates, with a mocked cross-encoder that scores based
    on simple keyword overlap — proving the reranking mechanics work
    correctly against real chunk_ids/text/metadata, not just synthetic
    strings.
    """

    def _get_real_candidates(self) -> list[RetrievedChunk]:
        from rag_pipeline.tests.fixtures_path import FIXTURES_DIR
        from rag_pipeline.config import PipelineConfig
        from rag_pipeline.pipeline import MarkdownChunkingPipeline

        cfg = PipelineConfig(input_dir=FIXTURES_DIR, verbose=False)
        pipeline = MarkdownChunkingPipeline(cfg)
        chunks = pipeline.process_file(FIXTURES_DIR / "Binder2.md")
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.normalized_text, metadata=c.metadata,
                rrf_score=0.01,
            )
            for c in chunks
        ]

    def test_real_data_reranks_without_error(self):
        candidates = self._get_real_candidates()
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5] * len(candidates)
        reranker._model = mock_model

        results = reranker.rerank("What is a CMI?", candidates)
        self.assertGreater(len(results), 0)
        real_ids = {c.chunk_id for c in candidates}
        for r in results:
            self.assertIn(r.chunk_id, real_ids)

    def test_real_data_keyword_overlap_scoring_promotes_relevant_chunk(self):
        """Simulate a cross-encoder that favors chunks mentioning 'Create
        SOI' — proving score-based reordering works on real content."""
        candidates = self._get_real_candidates()
        reranker = CrossEncoderReranker(RerankerConfig(final_n_results=3))

        def fake_predict(pairs):
            return [
                1.0 if "Create SOI" in text else 0.1
                for _, text in pairs
            ]
        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict
        reranker._model = mock_model

        results = reranker.rerank("Create SOI action", candidates)
        self.assertIn("Create SOI", results[0].text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
