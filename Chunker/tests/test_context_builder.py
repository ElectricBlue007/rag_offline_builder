"""
tests/test_context_builder.py

Unit tests for context_builder.py — pure formatting logic, no external
dependencies, fully deterministic.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import ContextBuilderConfig
from Chunker.generation.context_builder import (
    build_context,
    _format_chunk,
    _strip_hashes,
    _defensive_strip_html,
    _get_field,
)
from Chunker.retrieval.hybrid_retriever import RetrievedChunk
from Chunker.retrieval.reranker import RerankedChunk


def make_retrieved(chunk_id="a", text="Some content.", metadata=None, rrf_score=0.01):
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, metadata=metadata or {}, rrf_score=rrf_score,
    )


def make_reranked(chunk_id="a", text="Some content.", metadata=None):
    return RerankedChunk(
        chunk_id=chunk_id, text=text, metadata=metadata or {},
        rerank_score=0.9, original_rrf_score=0.01, original_rank=0,
    )


class TestGetField(unittest.TestCase):

    def test_dict_access(self):
        self.assertEqual(_get_field({"text": "hi"}, "text"), "hi")

    def test_object_attribute_access(self):
        chunk = make_retrieved(text="hello")
        self.assertEqual(_get_field(chunk, "text"), "hello")

    def test_missing_key_returns_default(self):
        self.assertEqual(_get_field({}, "text", "fallback"), "fallback")


class TestStripHashes(unittest.TestCase):

    def test_single_level(self):
        self.assertEqual(_strip_hashes("### System reports"), "System reports")

    def test_multi_level(self):
        result = _strip_hashes("# Prompt Engineering > ## Introduction")
        self.assertEqual(result, "Prompt Engineering > Introduction")

    def test_no_hashes_unchanged(self):
        self.assertEqual(_strip_hashes("Plain text"), "Plain text")

    def test_root_passed_through(self):
        self.assertEqual(_strip_hashes("(root)"), "(root)")


class TestDefensiveStripHtml(unittest.TestCase):

    def test_no_html_unchanged(self):
        text = "Create SOI — logical entry: CMI track."
        self.assertEqual(_defensive_strip_html(text), text)

    def test_html_tags_removed(self):
        text = "<table><tr><td>Create SOI</td></tr></table>"
        result = _defensive_strip_html(text)
        self.assertNotIn("<table", result)
        self.assertNotIn("<td>", result)
        self.assertIn("Create SOI", result)

    def test_fast_path_skips_regex_when_no_angle_bracket(self):
        """Optimization check: text with no '<' should short-circuit."""
        text = "Plain text with > greater than sign but no tags"
        result = _defensive_strip_html(text)
        self.assertEqual(result, text)


class TestFormatChunk(unittest.TestCase):

    def setUp(self):
        self.cfg = ContextBuilderConfig()

    def test_includes_index(self):
        chunk = make_retrieved(text="Content.")
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertTrue(result.startswith("[1]"))

    def test_includes_type_label_table(self):
        chunk = make_retrieved(text="Content.", metadata={"chunk_type": "table"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Table", result)

    def test_includes_type_label_screenshot(self):
        chunk = make_retrieved(text="Content.", metadata={"chunk_type": "image_description"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Screenshot", result)

    def test_includes_type_label_text(self):
        chunk = make_retrieved(text="Content.", metadata={"chunk_type": "text"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Text", result)

    def test_unknown_chunk_type_defaults_to_text_label(self):
        chunk = make_retrieved(text="Content.", metadata={"chunk_type": "mystery"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Text", result)

    def test_includes_source_file(self):
        chunk = make_retrieved(text="Content.", metadata={"source_file": "Binder2.md"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Binder2.md", result)

    def test_prefers_source_pdf_over_source_file(self):
        chunk = make_retrieved(text="Content.", metadata={
            "source_pdf": "Binder2.pdf", "source_file": "outputs/Binder2/Binder2.md",
        })
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Binder2.pdf", result)
        self.assertNotIn("outputs/Binder2", result)

    def test_includes_page_number(self):
        chunk = make_retrieved(text="Content.", metadata={
            "source_pdf": "Binder2.pdf", "page_numbers": "1",
        })
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("page 1", result)

    def test_comma_separated_page_numbers_displayed(self):
        """page_numbers arrives as a comma-separated STRING (ChromaDB's
        metadata sanitization already converted the original list)."""
        chunk = make_retrieved(text="Content.", metadata={
            "source_pdf": "Binder2.pdf", "page_numbers": "1,2",
        })
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("page 1,2", result)

    def test_page_numbers_as_raw_list_handled_defensively(self):
        """
        If page_numbers is still a raw Python list (e.g. this function
        called on chunks that bypassed ChromaDB's metadata sanitization),
        it must be converted to a clean comma string — NOT leak Python's
        list repr ("page [1, 2]") into the LLM-facing context.
        """
        chunk = make_retrieved(text="Content.", metadata={
            "source_pdf": "Binder2.pdf", "page_numbers": [1, 2],
        })
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("page 1,2", result)
        self.assertNotIn("[1, 2]", result)
        self.assertNotIn("[1,2]", result)

    def test_single_item_raw_list_page_number(self):
        """Use page 7 (not 1) so there's no ambiguity with the citation
        index [1] that always prefixes the first chunk."""
        chunk = make_retrieved(text="Content.", metadata={
            "source_pdf": "Binder2.pdf", "page_numbers": [7],
        })
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("page 7", result)
        self.assertNotIn("page [7]", result)

    def test_includes_breadcrumb_cleaned(self):
        chunk = make_retrieved(text="Content.", metadata={"breadcrumb": "### System reports"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("System reports", result)
        self.assertNotIn("###", result)

    def test_root_breadcrumb_excluded(self):
        chunk = make_retrieved(text="Content.", metadata={"breadcrumb": "(root)"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertNotIn("(root)", result)

    def test_text_appears_on_own_line(self):
        chunk = make_retrieved(text="UNIQUE_CONTENT_MARKER", metadata={"source_file": "f.md"})
        result = _format_chunk(chunk, 1, self.cfg)
        lines = result.split("\n")
        self.assertIn("UNIQUE_CONTENT_MARKER", lines[-1])

    def test_type_label_disabled(self):
        cfg = ContextBuilderConfig(include_type_label=False)
        chunk = make_retrieved(text="Content.", metadata={"chunk_type": "table"})
        result = _format_chunk(chunk, 1, cfg)
        self.assertNotIn("Table", result)

    def test_source_disabled(self):
        cfg = ContextBuilderConfig(include_source=False)
        chunk = make_retrieved(text="Content.", metadata={"source_file": "Binder2.md"})
        result = _format_chunk(chunk, 1, cfg)
        self.assertNotIn("Binder2.md", result)

    def test_breadcrumb_disabled(self):
        cfg = ContextBuilderConfig(include_breadcrumb=False)
        chunk = make_retrieved(text="Content.", metadata={"breadcrumb": "### System reports"})
        result = _format_chunk(chunk, 1, cfg)
        self.assertNotIn("System reports", result)

    def test_long_text_truncated(self):
        cfg = ContextBuilderConfig(max_chars_per_chunk=20)
        chunk = make_retrieved(text="A" * 100)
        result = _format_chunk(chunk, 1, cfg)
        self.assertIn("...[truncated]", result)
        self.assertNotIn("A" * 100, result)

    def test_truncation_disabled_when_none(self):
        cfg = ContextBuilderConfig(max_chars_per_chunk=None)
        chunk = make_retrieved(text="A" * 5000)
        result = _format_chunk(chunk, 1, cfg)
        self.assertIn("A" * 5000, result)

    def test_defensive_html_stripped_from_table_text(self):
        """Even if normalization was somehow bypassed, raw HTML must not
        leak into the LLM-facing context."""
        chunk = make_retrieved(
            text="<table><tr><td>Create SOI</td></tr></table>",
            metadata={"chunk_type": "table"},
        )
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertNotIn("<table", result)

    def test_works_with_reranked_chunk(self):
        chunk = make_reranked(text="Content.", metadata={"source_file": "f.md"})
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("[1]", result)
        self.assertIn("Content.", result)

    def test_works_with_dict_chunk(self):
        chunk = {"chunk_id": "a", "text": "Dict content.", "metadata": {"source_file": "f.md"}}
        result = _format_chunk(chunk, 1, self.cfg)
        self.assertIn("Dict content.", result)


class TestBuildContext(unittest.TestCase):

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(build_context([]), "")

    def test_multiple_chunks_separated_by_blank_line(self):
        chunks = [
            make_retrieved(chunk_id="a", text="First."),
            make_retrieved(chunk_id="b", text="Second."),
        ]
        result = build_context(chunks)
        self.assertIn("First.", result)
        self.assertIn("Second.", result)
        self.assertIn("\n\n", result)

    def test_indices_sequential(self):
        chunks = [make_retrieved(chunk_id=str(i), text=f"Content {i}") for i in range(3)]
        result = build_context(chunks)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertIn("[3]", result)

    def test_max_chunks_caps_output(self):
        cfg = ContextBuilderConfig(max_chunks=2)
        chunks = [make_retrieved(chunk_id=str(i), text=f"Content {i}") for i in range(5)]
        result = build_context(chunks, cfg)
        self.assertIn("Content 0", result)
        self.assertIn("Content 1", result)
        self.assertNotIn("Content 4", result)
        self.assertNotIn("[3]", result)

    def test_default_config_used_when_none_passed(self):
        chunks = [make_retrieved(text="Content.")]
        result = build_context(chunks)  # no config arg
        self.assertIn("Content.", result)


class TestBuildContextRealFixtures(unittest.TestCase):
    """Integration test against real chunk data from the pipeline."""

    def _get_real_chunks_as_retrieved(self) -> list[RetrievedChunk]:
        from Chunker.tests.fixtures_path import FIXTURES_DIR
        from Chunker.config import PipelineConfig
        from Chunker.pipeline import MarkdownChunkingPipeline

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

    def test_real_table_chunk_produces_no_html_in_context(self):
        chunks = self._get_real_chunks_as_retrieved()
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        context = build_context(table_chunks)
        self.assertNotIn("<table", context)
        self.assertNotIn("<td", context)

    def test_real_data_includes_source_and_breadcrumb(self):
        chunks = self._get_real_chunks_as_retrieved()
        context = build_context(chunks, ContextBuilderConfig(max_chunks=10))
        # source_pdf ("Binder2.pdf") is intentionally preferred over
        # source_file ("Binder2.md") when both are present — see
        # test_prefers_source_pdf_over_source_file.
        self.assertIn("Binder2.pdf", context)
        self.assertIn("Create SOI", context)

    def test_real_data_page_numbers_no_raw_list_brackets(self):
        """
        Real pipeline chunks carry page_numbers as a raw list (ChromaDB's
        sanitization hasn't run since this test bypasses ChromaDB) —
        must not leak as Python's list repr ("page [1]") into context.
        """
        chunks = self._get_real_chunks_as_retrieved()
        context = build_context(chunks, ContextBuilderConfig(max_chunks=10))
        self.assertNotIn("page [", context)

    def test_real_data_full_pipeline_produces_readable_context(self):
        chunks = self._get_real_chunks_as_retrieved()
        context = build_context(chunks, ContextBuilderConfig(max_chunks=3))
        self.assertGreater(len(context), 0)
        self.assertIn("[1]", context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
