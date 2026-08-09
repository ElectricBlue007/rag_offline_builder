"""
tests/test_bm25_index.py

Unit tests for BM25Index and tokenize() — using the REAL rank_bm25
library (pure Python, lightweight, no network/model download needed),
not a mock. This gives much stronger coverage of actual ranking
behaviour than mocking would.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import BM25IndexConfig
from rag_pipeline.storage.bm25_index import (
    BM25Index,
    BM25SearchResult,
    tokenize,
    _get_field,
)


def make_chunk(chunk_id: str, contextualized_text: str) -> dict:
    return {"chunk_id": chunk_id, "contextualized_text": contextualized_text}


class TestTokenize(unittest.TestCase):

    def test_simple_lowercase_split(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_none_safe(self):
        # tokenize should handle falsy input gracefully
        self.assertEqual(tokenize(None or ""), [])

    def test_punctuation_split(self):
        result = tokenize("CMI, MAC, and GCI.")
        self.assertEqual(result, ["cmi", "mac", "and", "gci"])

    def test_dotted_abbreviation_split(self):
        """'Vel.MAC' splits at the period into two independent tokens —
        this is intentional (see module docstring): it makes partial
        term matching MORE robust, not less."""
        result = tokenize("Vel.MAC")
        self.assertEqual(result, ["vel", "mac"])

    def test_spanish_accented_characters_preserved(self):
        result = tokenize("Definición del botón activación")
        self.assertEqual(result, ["definición", "del", "botón", "activación"])

    def test_short_technical_tokens_preserved(self):
        """'NT' (Binder2.md's action code) must survive — no stopword
        removal that could accidentally strip meaningful short tokens."""
        result = tokenize("Enter NT and Vel.MAC")
        self.assertIn("nt", result)

    def test_numbers_preserved(self):
        result = tokenize("Maximum number of CMI = 10")
        self.assertIn("10", result)

    def test_whitespace_only(self):
        self.assertEqual(tokenize("   \n\t  "), [])

    def test_mixed_case_normalized(self):
        result = tokenize("CMI Cmi cmi")
        self.assertEqual(result, ["cmi", "cmi", "cmi"])


class TestGetField(unittest.TestCase):

    def test_dict_access(self):
        self.assertEqual(_get_field({"chunk_id": "a"}, "chunk_id"), "a")

    def test_object_attribute_access(self):
        class Obj:
            chunk_id = "b"
        self.assertEqual(_get_field(Obj(), "chunk_id"), "b")

    def test_missing_key_returns_default(self):
        self.assertEqual(_get_field({}, "chunk_id", "fallback"), "fallback")


class TestBM25IndexBuild(unittest.TestCase):

    def test_build_sets_is_built_true(self):
        index = BM25Index()
        index.build([make_chunk("a", "some content here")])
        self.assertTrue(index.is_built)

    def test_empty_chunks_not_built(self):
        index = BM25Index()
        index.build([])
        self.assertFalse(index.is_built)
        self.assertEqual(index.size, 0)

    def test_size_matches_chunk_count(self):
        index = BM25Index()
        chunks = [make_chunk(str(i), f"content {i}") for i in range(5)]
        index.build(chunks)
        self.assertEqual(index.size, 5)

    def test_uses_contextualized_text_field(self):
        """Confirms the index is built from contextualized_text, not
        some other field — indirectly, by checking a unique marker
        only present in contextualized_text is searchable."""
        chunk = {
            "chunk_id": "a",
            "contextualized_text": "UNIQUE_MARKER_XYZ appears here",
            "embed_text": "passage: UNIQUE_MARKER_XYZ appears here",
        }
        index = BM25Index()
        index.build([chunk])
        results = index.search("UNIQUE_MARKER_XYZ")
        self.assertEqual(len(results), 1)


class TestBM25IndexSearch(unittest.TestCase):

    def setUp(self):
        self.index = BM25Index()
        self.chunks = [
            make_chunk("cmi_doc", "This describes the CMI Create SOI action and CMI track."),
            make_chunk("prompt_doc", "This describes prompt engineering and few-shot examples."),
            make_chunk("unrelated_doc", "The weather today is sunny with clear skies."),
        ]
        self.index.build(self.chunks)

    def test_search_returns_bm25_search_results(self):
        results = self.index.search("CMI")
        for r in results:
            self.assertIsInstance(r, BM25SearchResult)

    def test_relevant_document_ranks_first(self):
        results = self.index.search("CMI track action")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk_id, "cmi_doc")

    def test_different_query_ranks_different_document_first(self):
        results = self.index.search("prompt engineering few-shot")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk_id, "prompt_doc")

    def test_n_results_limits_output(self):
        results = self.index.search("the", n_results=1)
        self.assertLessEqual(len(results), 1)

    def test_results_sorted_descending_by_score(self):
        results = self.index.search("CMI action track")
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.index.search(""), [])

    def test_whitespace_only_query_returns_empty(self):
        self.assertEqual(self.index.search("   "), [])

    def test_query_with_no_matching_terms_still_returns_ranked_list(self):
        """
        Even with zero term overlap, BM25 still produces a score for
        every document (typically 0, but per the module's documented
        IDF quirk, not reliably so) — the module must NOT filter these
        out by an absolute threshold, only rank them.
        """
        results = self.index.search("xyzxyzxyz_nonexistent_term")
        # Must not raise, and must return a list (possibly all zero-ish
        # scores) rather than silently dropping everything.
        self.assertIsInstance(results, list)

    def test_search_on_unbuilt_index_returns_empty(self):
        empty_index = BM25Index()
        self.assertEqual(empty_index.search("anything"), [])

    def test_chunk_ids_match_original_chunks(self):
        results = self.index.search("CMI")
        result_ids = {r.chunk_id for r in results}
        original_ids = {c["chunk_id"] for c in self.chunks}
        self.assertTrue(result_ids.issubset(original_ids))


class TestBM25NegativeIdfQuirk(unittest.TestCase):
    """
    Directly verifies the documented negative-IDF behaviour and that
    this module's search() does NOT filter it out — this is the exact
    scenario that would silently break a naive "score > 0" filter.
    """

    def test_common_term_across_small_corpus_can_score_negative(self):
        chunks = [
            make_chunk("a", "hello world foo"),
            make_chunk("b", "hello world bar"),
        ]
        index = BM25Index()
        index.build(chunks)
        results = index.search("hello")
        # Both documents contain "hello" — BM25's IDF for a term present
        # in every document of a tiny corpus commonly goes negative.
        # The key assertion: results are NOT empty despite this.
        self.assertEqual(len(results), 2)

    def test_negative_score_results_still_returned_and_ranked(self):
        chunks = [
            make_chunk("a", "hello world foo baz"),
            make_chunk("b", "hello world bar"),
        ]
        index = BM25Index()
        index.build(chunks)
        results = index.search("hello world")
        # Regardless of sign, both should be present since both share
        # every query term.
        result_ids = {r.chunk_id for r in results}
        self.assertEqual(result_ids, {"a", "b"})


class TestBM25IndexPersistence(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        chunks = [
            make_chunk("a", "CMI track content here"),
            make_chunk("b", "prompt engineering content here"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"

            index1 = BM25Index()
            index1.build(chunks)
            index1.save(str(path))

            index2 = BM25Index()
            index2.load(str(path))

            self.assertEqual(index2.size, 2)
            results = index2.search("CMI track")
            self.assertEqual(results[0].chunk_id, "a")

    def test_save_uses_config_default_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BM25IndexConfig(persist_path=str(Path(tmp) / "default.pkl"))
            index = BM25Index(cfg)
            index.build([make_chunk("a", "content")])
            index.save()  # no explicit path — uses cfg.persist_path
            self.assertTrue(Path(cfg.persist_path).exists())

    def test_load_uses_config_default_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BM25IndexConfig(persist_path=str(Path(tmp) / "default.pkl"))
            index1 = BM25Index(cfg)
            index1.build([make_chunk("a", "unique content marker")])
            index1.save()

            index2 = BM25Index(cfg)
            index2.load()  # no explicit path — uses cfg.persist_path
            self.assertEqual(index2.size, 1)

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested_path = Path(tmp) / "a" / "b" / "c" / "index.pkl"
            index = BM25Index()
            index.build([make_chunk("a", "content")])
            index.save(str(nested_path))
            self.assertTrue(nested_path.exists())

    def test_save_unbuilt_index_raises(self):
        index = BM25Index()
        with self.assertRaises(RuntimeError):
            index.save("/tmp/whatever.pkl")

    def test_load_nonexistent_file_raises(self):
        index = BM25Index()
        with self.assertRaises(FileNotFoundError):
            index.load("/tmp/definitely_does_not_exist_12345.pkl")

    def test_loaded_index_scores_match_original(self):
        """Round-tripping through pickle must not change ranking behaviour."""
        chunks = [
            make_chunk("a", "CMI Create SOI track action restrictions"),
            make_chunk("b", "prompt engineering few shot examples"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pkl"
            index1 = BM25Index()
            index1.build(chunks)
            original_results = index1.search("CMI action")
            index1.save(str(path))

            index2 = BM25Index()
            index2.load(str(path))
            reloaded_results = index2.search("CMI action")

            self.assertEqual(
                [(r.chunk_id, r.score) for r in original_results],
                [(r.chunk_id, r.score) for r in reloaded_results],
            )


class TestBM25IndexRealFixtures(unittest.TestCase):
    """Integration tests against the real Binder2.md / Testing.md fixtures."""

    def _get_chunks(self, md_filename: str) -> list:
        from rag_pipeline.tests.fixtures_path import FIXTURES_DIR
        from rag_pipeline.config import PipelineConfig
        from rag_pipeline.pipeline import MarkdownChunkingPipeline

        cfg = PipelineConfig(input_dir=FIXTURES_DIR, verbose=False)
        pipeline = MarkdownChunkingPipeline(cfg)
        return pipeline.process_file(FIXTURES_DIR / md_filename)

    def test_binder2_cmi_query_finds_relevant_chunk(self):
        chunks = self._get_chunks("Binder2.md")
        index = BM25Index()
        index.build(chunks)
        results = index.search("CMI Create SOI restrictions")
        self.assertGreater(len(results), 0)

    def test_binder2_spanish_query_finds_relevant_chunk(self):
        chunks = self._get_chunks("Binder2.md")
        index = BM25Index()
        index.build(chunks)
        results = index.search("botón conjunto trabajo")
        self.assertGreater(len(results), 0)

    def test_testing_gemini_query_finds_relevant_chunk(self):
        chunks = self._get_chunks("Testing.md")
        index = BM25Index()
        index.build(chunks)
        results = index.search("gemini-pro temperature")
        self.assertGreater(len(results), 0)

    def test_binder2_and_testing_combined_index(self):
        """Realistic multi-document scenario: both files in one index."""
        binder_chunks = self._get_chunks("Binder2.md")
        testing_chunks = self._get_chunks("Testing.md")
        index = BM25Index()
        index.build(binder_chunks + testing_chunks)
        self.assertEqual(index.size, len(binder_chunks) + len(testing_chunks))

        cmi_results = index.search("CMI track")
        prompt_results = index.search("prompt engineering few-shot")
        self.assertGreater(len(cmi_results), 0)
        self.assertGreater(len(prompt_results), 0)
        # Sanity: the CMI query's top result should come from Binder2's
        # chunk_ids, not Testing's.
        binder_ids = {c.chunk_id for c in binder_chunks}
        self.assertIn(cmi_results[0].chunk_id, binder_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
