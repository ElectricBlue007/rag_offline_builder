"""
tests/test_hybrid_retriever.py

Unit tests for reciprocal_rank_fusion() (pure function, tested in
isolation with simple ID lists) and HybridRetriever (tested with
injected fake/mock components — a real BM25Index since it's pure
Python, and mocked ChromaDB/EmbedGenerator since those need a real
model/service).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import HybridRetrieverConfig
from Chunker.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)
from Chunker.storage.bm25_index import BM25Index


# ════════════════════════════════════════════════════════════════════
# Pure RRF math — no mocks needed at all
# ════════════════════════════════════════════════════════════════════

class TestReciprocalRankFusion(unittest.TestCase):

    def test_single_list_preserves_rank_order(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        self.assertGreater(scores["a"], scores["b"])
        self.assertGreater(scores["b"], scores["c"])

    def test_top_rank_gets_highest_score(self):
        scores = reciprocal_rank_fusion([["a", "b"]], k=60)
        self.assertAlmostEqual(scores["a"], 1.0 / 61)
        self.assertAlmostEqual(scores["b"], 1.0 / 62)

    def test_doc_in_both_lists_scores_higher_than_doc_in_one(self):
        dense = ["x", "a", "b"]
        sparse = ["a", "y", "z"]
        scores = reciprocal_rank_fusion([dense, sparse], k=60)
        # "a" appears in both lists — must outscore "x" and "b" (dense-only)
        # and "y"/"z" (sparse-only), each of which appears in only one list.
        self.assertGreater(scores["a"], scores["x"])
        self.assertGreater(scores["a"], scores["y"])

    def test_doc_missing_from_a_list_gets_zero_contribution_not_penalty(self):
        """A doc absent from one list contributes exactly 0 from it —
        not some artificial worst-rank penalty."""
        dense = ["a"]
        sparse = ["b"]
        scores = reciprocal_rank_fusion([dense, sparse], k=60)
        # "a" only in dense at rank 1 → score == 1/(60+1) exactly
        self.assertAlmostEqual(scores["a"], 1.0 / 61)
        self.assertAlmostEqual(scores["b"], 1.0 / 61)

    def test_empty_lists_produce_empty_scores(self):
        scores = reciprocal_rank_fusion([[], []], k=60)
        self.assertEqual(scores, {})

    def test_one_empty_one_populated_list(self):
        scores = reciprocal_rank_fusion([[], ["a", "b"]], k=60)
        self.assertIn("a", scores)
        self.assertGreater(scores["a"], scores["b"])

    def test_k_parameter_affects_score_magnitude(self):
        scores_k60 = reciprocal_rank_fusion([["a"]], k=60)
        scores_k10 = reciprocal_rank_fusion([["a"]], k=10)
        # Smaller k → larger contribution for the same rank
        self.assertGreater(scores_k10["a"], scores_k60["a"])

    def test_three_lists_sum_correctly(self):
        """Generalizes beyond 2 lists — sums across all of them."""
        scores = reciprocal_rank_fusion([["a"], ["a"], ["a"]], k=60)
        expected = 3 * (1.0 / 61)
        self.assertAlmostEqual(scores["a"], expected)

    def test_duplicate_doc_within_same_list_only_counted_once_at_first_rank(self):
        """A well-formed ranked list shouldn't have duplicates, but if it
        does, enumerate() naturally uses each position — this test just
        documents that behaviour rather than silently guessing."""
        scores = reciprocal_rank_fusion([["a", "a"]], k=60)
        # Both occurrences contribute (rank 1 and rank 2) — sums together
        expected = 1.0 / 61 + 1.0 / 62
        self.assertAlmostEqual(scores["a"], expected)

    def test_default_k_is_60(self):
        scores_default = reciprocal_rank_fusion([["a"]])
        scores_explicit = reciprocal_rank_fusion([["a"]], k=60)
        self.assertEqual(scores_default, scores_explicit)


# ════════════════════════════════════════════════════════════════════
# Fakes for dense (ChromaDB) and embedding — real BM25 (pure Python)
# ════════════════════════════════════════════════════════════════════

class FakeChromaIngestor:
    """
    Stands in for ChromaIngestor with a fixed, scriptable dense result
    set — avoids needing a real ChromaDB install for these tests while
    exercising the exact response SHAPE (.query()'s nested lists vs.
    .get_by_ids()'s flat structure) that the retriever depends on.
    """

    def __init__(self, dense_ids=None, dense_documents=None, dense_metadatas=None,
                 extra_docs_by_id=None):
        self._dense_ids = dense_ids or []
        self._dense_documents = dense_documents or []
        self._dense_metadatas = dense_metadatas or [{} for _ in self._dense_ids]
        self._extra_docs_by_id = extra_docs_by_id or {}
        self.last_query_kwargs = None

    def query(self, query_vector, n_results=10, where=None):
        self.last_query_kwargs = {
            "query_vector": query_vector, "n_results": n_results, "where": where
        }
        return {
            "ids": [self._dense_ids],
            "documents": [self._dense_documents],
            "metadatas": [self._dense_metadatas],
        }

    def get_by_ids(self, ids):
        docs = [self._extra_docs_by_id.get(i, {"document": "", "metadata": {}}) for i in ids]
        return {
            "ids": ids,
            "documents": [d["document"] for d in docs],
            "metadatas": [d["metadata"] for d in docs],
        }


class FakeEmbedGenerator:
    """Deterministic fake — no real model needed."""
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def make_bm25_index(chunk_texts: dict) -> BM25Index:
    """chunk_texts: {chunk_id: contextualized_text}"""
    chunks = [{"chunk_id": cid, "contextualized_text": text}
              for cid, text in chunk_texts.items()]
    index = BM25Index()
    index.build(chunks)
    return index


# ════════════════════════════════════════════════════════════════════
# HybridRetriever integration tests
# ════════════════════════════════════════════════════════════════════

class TestHybridRetrieverBasics(unittest.TestCase):

    def test_empty_query_returns_empty_list(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        self.assertEqual(retriever.retrieve(""), [])

    def test_whitespace_query_returns_empty_list(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        self.assertEqual(retriever.retrieve("   "), [])

    def test_returns_retrieved_chunk_instances(self):
        chroma = FakeChromaIngestor(
            dense_ids=["a"], dense_documents=["Content A"], dense_metadatas=[{"language": "en"}]
        )
        bm25 = make_bm25_index({"a": "Content A about CMI"})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("CMI")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, RetrievedChunk)

    def test_dense_only_hit_has_sparse_rank_none(self):
        chroma = FakeChromaIngestor(
            dense_ids=["only_dense"], dense_documents=["Dense content"],
            dense_metadatas=[{}]
        )
        bm25 = make_bm25_index({"unrelated": "totally different words entirely"})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("some query")
        dense_hit = next(r for r in results if r.chunk_id == "only_dense")
        self.assertIsNotNone(dense_hit.dense_rank)
        self.assertIsNone(dense_hit.sparse_rank)

    def test_sparse_only_hit_has_dense_rank_none(self):
        chroma = FakeChromaIngestor(dense_ids=[], dense_documents=[], dense_metadatas=[])
        bm25 = make_bm25_index({"only_sparse": "CMI Create SOI restrictions"})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("CMI restrictions")
        self.assertGreater(len(results), 0)
        sparse_hit = results[0]
        self.assertEqual(sparse_hit.chunk_id, "only_sparse")
        self.assertIsNone(sparse_hit.dense_rank)
        self.assertIsNotNone(sparse_hit.sparse_rank)

    def test_sparse_only_hit_backfills_via_get_by_ids(self):
        """The core correctness test: a chunk found ONLY by BM25 must
        still get its document/metadata populated via a direct fetch."""
        chroma = FakeChromaIngestor(
            dense_ids=[], dense_documents=[], dense_metadatas=[],
            extra_docs_by_id={
                "sparse_only": {"document": "Backfilled content", "metadata": {"chunk_type": "table"}}
            },
        )
        bm25 = make_bm25_index({"sparse_only": "CMI Create SOI"})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("CMI Create SOI")
        self.assertEqual(results[0].text, "Backfilled content")
        self.assertEqual(results[0].metadata, {"chunk_type": "table"})

    def test_doc_in_both_dense_and_sparse_ranks_first(self):
        chroma = FakeChromaIngestor(
            dense_ids=["both", "dense_only"],
            dense_documents=["Both content", "Dense only content"],
            dense_metadatas=[{}, {}],
        )
        bm25 = make_bm25_index({
            "both": "CMI action content",
            "sparse_only": "completely different unrelated text",
        })
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("CMI action")
        self.assertEqual(results[0].chunk_id, "both")

    def test_final_n_results_truncates(self):
        chroma = FakeChromaIngestor(
            dense_ids=[f"id{i}" for i in range(20)],
            dense_documents=[f"content {i}" for i in range(20)],
            dense_metadatas=[{} for _ in range(20)],
        )
        bm25 = make_bm25_index({})
        cfg = HybridRetrieverConfig(final_n_results=3)
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator(), cfg)
        results = retriever.retrieve("query")
        self.assertLessEqual(len(results), 3)

    def test_rrf_score_present_on_all_results(self):
        chroma = FakeChromaIngestor(
            dense_ids=["a"], dense_documents=["A"], dense_metadatas=[{}]
        )
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("query")
        for r in results:
            self.assertIsInstance(r.rrf_score, float)
            self.assertGreater(r.rrf_score, 0)


class TestHybridRetrieverLanguageFilter(unittest.TestCase):

    def test_english_query_applies_language_where_filter(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        retriever.retrieve("This is a clear English sentence about operators.")
        self.assertEqual(chroma.last_query_kwargs["where"], {"language": "en"})

    def test_spanish_query_applies_language_where_filter(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        retriever.retrieve("Definición del botón de conjunto de trabajo nueva posición")
        self.assertEqual(chroma.last_query_kwargs["where"], {"language": "es"})

    def test_unknown_language_omits_where_filter(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        retriever.retrieve("xyz")  # too short/ambiguous to detect
        self.assertIsNone(chroma.last_query_kwargs["where"])

    def test_language_filter_disabled_omits_where_even_for_clear_language(self):
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})
        cfg = HybridRetrieverConfig(language_filter=False)
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator(), cfg)
        retriever.retrieve("This is a clear English sentence about operators.")
        self.assertIsNone(chroma.last_query_kwargs["where"])


class TestHybridRetrieverQueryPrefix(unittest.TestCase):

    def test_dense_search_uses_query_prefixed_text_for_embedding(self):
        """The configured query-side instruction prefix (EmbedTextConfig)
        must be applied before embedding — verified by checking what
        actually got passed to embed_texts. Derives the expected value
        from the actual config default rather than hardcoding the
        prefix string, since that's a model-dependent choice that will
        keep changing as the embedding model changes."""
        from Chunker.config import EmbedTextConfig

        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({})

        class RecordingEmbedGenerator:
            def __init__(self):
                self.last_texts = None
            def embed_texts(self, texts):
                self.last_texts = texts
                return [[0.1, 0.2] for _ in texts]

        embedder = RecordingEmbedGenerator()
        retriever = HybridRetriever(chroma, bm25, embedder)
        retriever.retrieve("What is a CMI?")
        expected = EmbedTextConfig().query_prefix + "What is a CMI?"
        self.assertEqual(embedder.last_texts, [expected])

    def test_sparse_search_uses_raw_query_not_prefixed(self):
        """BM25 must search the RAW query — 'query: ' prefix would just
        inject a spurious common token into every single search."""
        chroma = FakeChromaIngestor()
        bm25 = make_bm25_index({"a": "some content about queries and cmi"})

        class RecordingBM25(BM25Index):
            def __init__(self, real_index):
                self._real = real_index
                self.last_query = None
            def search(self, query, n_results=10):
                self.last_query = query
                return self._real.search(query, n_results)

        recording_bm25 = RecordingBM25(bm25)
        retriever = HybridRetriever(chroma, recording_bm25, FakeEmbedGenerator())
        retriever.retrieve("What is a CMI?")
        self.assertEqual(recording_bm25.last_query, "What is a CMI?")
        self.assertNotIn("query:", recording_bm25.last_query)


class TestHybridRetrieverUnwrap(unittest.TestCase):

    def test_unwrap_normal_nested_list(self):
        result = HybridRetriever._unwrap([["a", "b"]])
        self.assertEqual(result, ["a", "b"])

    def test_unwrap_none_returns_empty(self):
        self.assertEqual(HybridRetriever._unwrap(None), [])

    def test_unwrap_empty_list_returns_empty(self):
        self.assertEqual(HybridRetriever._unwrap([]), [])

    def test_unwrap_empty_inner_list(self):
        self.assertEqual(HybridRetriever._unwrap([[]]), [])


class TestHybridRetrieverRealFixtures(unittest.TestCase):
    """
    Integration test against real Binder2/Testing chunk data, using the
    REAL BM25Index (pure Python) and a fake ChromaDB standing in for
    dense search (since we're not running a real embedding model here).
    This proves the fusion logic works correctly on real chunk_ids and
    real contextualized_text, not just synthetic test strings.
    """

    def _get_all_real_chunks(self):
        from Chunker.tests.fixtures_path import FIXTURES_DIR
        from Chunker.config import PipelineConfig
        from Chunker.pipeline import MarkdownChunkingPipeline

        cfg = PipelineConfig(input_dir=FIXTURES_DIR, verbose=False)
        pipeline = MarkdownChunkingPipeline(cfg)
        chunks = []
        for fname in ["Binder2.md", "Testing.md"]:
            chunks.extend(pipeline.process_file(FIXTURES_DIR / fname))
        return chunks

    def test_real_data_cmi_query_returns_results(self):
        chunks = self._get_all_real_chunks()
        bm25 = BM25Index()
        bm25.build(chunks)

        # Fake dense search returning nothing — proves BM25 alone still
        # surfaces the right chunk via the sparse-only backfill path
        cmi_chunk = next(c for c in chunks if "Create SOI" in c.normalized_text)
        chroma = FakeChromaIngestor(
            dense_ids=[], dense_documents=[], dense_metadatas=[],
            extra_docs_by_id={
                cmi_chunk.chunk_id: {
                    "document": cmi_chunk.normalized_text,
                    "metadata": cmi_chunk.metadata,
                }
            },
        )
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("CMI Create SOI restrictions")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk_id, cmi_chunk.chunk_id)

    def test_real_data_all_chunk_ids_valid(self):
        chunks = self._get_all_real_chunks()
        bm25 = BM25Index()
        bm25.build(chunks)
        chroma = FakeChromaIngestor()
        retriever = HybridRetriever(chroma, bm25, FakeEmbedGenerator())
        results = retriever.retrieve("gemini-pro prompt engineering")
        real_ids = {c.chunk_id for c in chunks}
        for r in results:
            self.assertIn(r.chunk_id, real_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
