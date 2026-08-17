"""
tests/test_metadata_enricher.py

Unit tests for MetadataEnricher — using real _metadata.json structures
from Binder2_metadata.json and Testing_metadata.json as fixtures.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import MetadataConfig
from rag_pipeline.enrichers.metadata_enricher import (
    MetadataEnricher,
    EnrichedChunk,
    _load_page_metadata,
    _estimate_page_numbers,
)
from rag_pipeline.splitters.semantic_chunker import SemanticChunk
from rag_pipeline.tests.fixtures_path import BINDER2_METADATA_JSON, TESTING_METADATA_JSON


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

# Exact structure from Binder2_metadata.json
BINDER2_META = {
    "file_name": "Binder2.pdf",
    "num_pages": 2,
    "total_token_count": 2458,
    "total_chunks": 23,
    "total_images": 1,
    "pages": [
        {"page_num": 0, "page_box": [0, 0, 1632, 2112], "token_count": 862, "num_chunks": 8, "num_images": 1},
        {"page_num": 1, "page_box": [0, 0, 1632, 2112], "token_count": 1596, "num_chunks": 15, "num_images": 0},
    ]
}

# Exact structure from Testing_metadata.json
TESTING_META = {
    "file_name": "Testing.pdf",
    "num_pages": 6,
    "total_token_count": 3555,
    "total_chunks": 49,
    "total_images": 4,
    "pages": [
        {"page_num": 0, "num_chunks": 4, "num_images": 2},
        {"page_num": 1, "num_chunks": 20, "num_images": 1},
        {"page_num": 2, "num_chunks": 3, "num_images": 1},
        {"page_num": 3, "num_chunks": 6, "num_images": 0},
        {"page_num": 4, "num_chunks": 10, "num_images": 0},
        {"page_num": 5, "num_chunks": 6, "num_images": 0},
    ]
}


def make_chunk(
    content: str = "Test content.",
    section_type: str = "text",
    breadcrumb: str = "### Section",
    header_levels=None,
    header_texts=None,
    source_path: str = "test.md",
    chunk_index: int = 0,
) -> SemanticChunk:
    return SemanticChunk(
        content=content,
        section_type=section_type,
        breadcrumb=breadcrumb,
        header_levels=[3] if header_levels is None else header_levels,
        header_texts=["Section"] if header_texts is None else header_texts,
        source_path=source_path,
        chunk_index=chunk_index,
    )


class TestLoadPageMetadata(unittest.TestCase):

    def test_loads_valid_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "Binder2.md"
            meta_path = Path(tmp) / "Binder2_metadata.json"
            md_path.write_text("# Test", encoding="utf-8")
            meta_path.write_text(json.dumps(BINDER2_META), encoding="utf-8")

            result = _load_page_metadata(md_path)
            self.assertIsNotNone(result)
            self.assertEqual(result["file_name"], "Binder2.pdf")
            self.assertEqual(result["num_pages"], 2)

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "NoMeta.md"
            md_path.write_text("# Test", encoding="utf-8")
            result = _load_page_metadata(md_path)
            self.assertIsNone(result)

    def test_returns_none_for_corrupt_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "Corrupt.md"
            meta_path = Path(tmp) / "Corrupt_metadata.json"
            md_path.write_text("# Test", encoding="utf-8")
            meta_path.write_text("{ invalid json !!!", encoding="utf-8")
            result = _load_page_metadata(md_path)
            self.assertIsNone(result)

    def test_sibling_naming_convention(self):
        """Stem of md file + '_metadata.json' must be the lookup path."""
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "Testing.md"
            meta_path = Path(tmp) / "Testing_metadata.json"
            md_path.write_text("# Test", encoding="utf-8")
            meta_path.write_text(json.dumps(TESTING_META), encoding="utf-8")

            result = _load_page_metadata(md_path)
            self.assertIsNotNone(result)
            self.assertEqual(result["num_pages"], 6)


class TestEstimatePageNumbers(unittest.TestCase):
    """Uses Binder2 structure: page 0 has 8 chunks, page 1 has 15 chunks."""

    def test_chunk_zero_is_on_page_one(self):
        result = _estimate_page_numbers(0, 23, BINDER2_META)
        self.assertEqual(result, [1])

    def test_chunk_seven_still_page_one(self):
        """Chunk index 7 is the last chunk on page 0 (0-indexed: chunks 0–7)."""
        result = _estimate_page_numbers(7, 23, BINDER2_META)
        self.assertEqual(result, [1])

    def test_chunk_eight_is_on_page_two(self):
        """Chunk index 8 is the first chunk on page 1."""
        result = _estimate_page_numbers(8, 23, BINDER2_META)
        self.assertEqual(result, [2])

    def test_chunk_beyond_all_pages_returns_last_page(self):
        result = _estimate_page_numbers(999, 23, BINDER2_META)
        self.assertEqual(result, [2])  # last page is page_num=1 → 1-indexed=2

    def test_empty_pages_returns_empty_list(self):
        meta_no_pages = {"file_name": "test.pdf", "num_pages": 0, "pages": []}
        result = _estimate_page_numbers(0, 5, meta_no_pages)
        self.assertEqual(result, [])

    def test_zero_total_chunks_returns_empty(self):
        result = _estimate_page_numbers(0, 0, BINDER2_META)
        self.assertEqual(result, [])

    def test_testing_meta_first_chunk_page_one(self):
        """Testing.pdf: page 0 has 4 chunks (indices 0–3)."""
        result = _estimate_page_numbers(0, 49, TESTING_META)
        self.assertEqual(result, [1])

    def test_testing_meta_chunk_four_page_two(self):
        """Testing.pdf: page 1 starts at chunk index 4."""
        result = _estimate_page_numbers(4, 49, TESTING_META)
        self.assertEqual(result, [2])

    def test_returns_list_type(self):
        result = _estimate_page_numbers(0, 10, BINDER2_META)
        self.assertIsInstance(result, list)


class TestMetadataEnricher(unittest.TestCase):

    def _make_enricher(self, **cfg_kwargs) -> MetadataEnricher:
        cfg = MetadataConfig(**cfg_kwargs)
        return MetadataEnricher(cfg)

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_list_of_enriched_chunks(self):
        enricher = MetadataEnricher()
        chunks = [make_chunk()]
        result = enricher.enrich(chunks, Path("test.md"))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], EnrichedChunk)

    def test_one_chunk_in_one_chunk_out(self):
        enricher = MetadataEnricher()
        chunks = [make_chunk()]
        result = enricher.enrich(chunks, Path("test.md"))
        self.assertEqual(len(result), 1)

    def test_empty_input_returns_empty(self):
        enricher = MetadataEnricher()
        result = enricher.enrich([], Path("test.md"))
        self.assertEqual(result, [])

    # ── source_file ──────────────────────────────────────────────────────────

    def test_source_file_included(self):
        enricher = self._make_enricher(include_source_file=True)
        test_path = Path("my/file.md")
        result = enricher.enrich([make_chunk()], test_path)
        self.assertIn("source_file", result[0].metadata)
        # Compare against str(Path(...)) rather than a hardcoded forward
        # slash — on Windows, Path("my/file.md") stringifies with
        # backslashes ("my\\file.md"), so a literal "my/file.md" check
        # would fail there even though the metadata is correct.
        self.assertEqual(result[0].metadata["source_file"], str(test_path))

    def test_source_file_excluded(self):
        enricher = self._make_enricher(include_source_file=False)
        result = enricher.enrich([make_chunk()], Path("file.md"))
        self.assertNotIn("source_file", result[0].metadata)

    # ── breadcrumb + section_header ──────────────────────────────────────────

    def test_breadcrumb_included(self):
        enricher = self._make_enricher(include_breadcrumb=True)
        chunk = make_chunk(breadcrumb="# Root > ## Sub")
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertEqual(result[0].metadata["breadcrumb"], "# Root > ## Sub")

    def test_section_header_is_deepest(self):
        enricher = self._make_enricher(include_breadcrumb=True)
        chunk = make_chunk(
            header_texts=["Root", "Sub", "Deep"],
            header_levels=[1, 2, 3],
        )
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertEqual(result[0].metadata["section_header"], "Deep")

    def test_section_header_empty_when_no_headers(self):
        enricher = self._make_enricher(include_breadcrumb=True)
        chunk = make_chunk(header_texts=[], header_levels=[], breadcrumb="(root)")
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertEqual(result[0].metadata["section_header"], "")

    def test_breadcrumb_excluded(self):
        enricher = self._make_enricher(include_breadcrumb=False)
        result = enricher.enrich([make_chunk()], Path("f.md"))
        self.assertNotIn("breadcrumb", result[0].metadata)
        self.assertNotIn("section_header", result[0].metadata)

    # ── chunk_type ───────────────────────────────────────────────────────────

    def test_chunk_type_text(self):
        enricher = self._make_enricher(include_chunk_type=True)
        result = enricher.enrich([make_chunk(section_type="text")], Path("f.md"))
        self.assertEqual(result[0].metadata["chunk_type"], "text")

    def test_chunk_type_table(self):
        enricher = self._make_enricher(include_chunk_type=True)
        result = enricher.enrich([make_chunk(section_type="table")], Path("f.md"))
        self.assertEqual(result[0].metadata["chunk_type"], "table")

    def test_chunk_type_image_description(self):
        enricher = self._make_enricher(include_chunk_type=True)
        result = enricher.enrich([make_chunk(section_type="image_description")], Path("f.md"))
        self.assertEqual(result[0].metadata["chunk_type"], "image_description")

    def test_chunk_type_excluded(self):
        enricher = self._make_enricher(include_chunk_type=False)
        result = enricher.enrich([make_chunk()], Path("f.md"))
        self.assertNotIn("chunk_type", result[0].metadata)

    # ── chunk_index ──────────────────────────────────────────────────────────

    def test_chunk_index_preserved(self):
        enricher = self._make_enricher(include_chunk_index=True)
        chunk = make_chunk(chunk_index=7)
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertEqual(result[0].metadata["chunk_index"], 7)

    def test_chunk_index_excluded(self):
        enricher = self._make_enricher(include_chunk_index=False)
        result = enricher.enrich([make_chunk()], Path("f.md"))
        self.assertNotIn("chunk_index", result[0].metadata)

    # ── language ─────────────────────────────────────────────────────────────

    def test_language_detected_english(self):
        enricher = self._make_enricher(include_language=True)
        chunk = make_chunk(
            content=(
                "This service allows the operator to manipulate great circles. "
                "The operator can create and modify CMI track parameters."
            )
        )
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertIn("language", result[0].metadata)
        self.assertEqual(result[0].metadata["language"], "en")

    def test_language_detected_spanish(self):
        enricher = self._make_enricher(include_language=True)
        chunk = make_chunk(
            content=(
                "Si ya existe una definicion de posicion de boton de conjunto "
                "de trabajo nueva, la posicion de todas las definiciones aumentara."
            )
        )
        result = enricher.enrich([chunk], Path("f.md"))
        self.assertEqual(result[0].metadata["language"], "es")

    def test_language_excluded(self):
        enricher = self._make_enricher(include_language=False)
        result = enricher.enrich([make_chunk()], Path("f.md"))
        self.assertNotIn("language", result[0].metadata)

    # ── page metadata ────────────────────────────────────────────────────────

    def test_page_metadata_loaded_from_sibling_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "Binder2.md"
            meta_path = Path(tmp) / "Binder2_metadata.json"
            md_path.write_text("# content", encoding="utf-8")
            meta_path.write_text(json.dumps(BINDER2_META), encoding="utf-8")

            enricher = self._make_enricher(include_page_metadata=True)
            chunk = make_chunk(chunk_index=0)
            result = enricher.enrich([chunk], md_path)

            self.assertIn("source_pdf", result[0].metadata)
            self.assertEqual(result[0].metadata["source_pdf"], "Binder2.pdf")
            self.assertEqual(result[0].metadata["total_pages"], 2)
            self.assertIn("page_numbers", result[0].metadata)

    def test_page_metadata_falls_back_to_derived_values_when_no_json(self):
        """
        With no sidecar file at all, source_pdf/total_pages are no longer
        just silently omitted — they fall back to values derived from the
        markdown filename itself, so citations still have something
        useful to show even for OCR tools that don't emit a metadata
        sidecar (or when a document simply has no page markers).
        """
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "NoMeta.md"
            md_path.write_text("# content", encoding="utf-8")

            enricher = self._make_enricher(include_page_metadata=True)
            result = enricher.enrich([make_chunk()], md_path)
            self.assertEqual(result[0].metadata["source_pdf"], "NoMeta.pdf")
            self.assertEqual(result[0].metadata["total_pages"], 0)
            self.assertEqual(result[0].metadata["page_numbers"], [])

    def test_page_metadata_flag_false_no_page_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "Binder2.md"
            meta_path = Path(tmp) / "Binder2_metadata.json"
            md_path.write_text("# content", encoding="utf-8")
            meta_path.write_text(json.dumps(BINDER2_META), encoding="utf-8")

            enricher = self._make_enricher(include_page_metadata=False)
            result = enricher.enrich([make_chunk()], md_path)
            self.assertNotIn("source_pdf", result[0].metadata)

    # ── page attribution (rewritten in 1.7.0) ───────────────────────
    #
    # These used to assert that chunk_index N mapped to a page by counting
    # the sidecar's per-page num_chunks. That was the bug: the sidecar
    # counts the OCR tool's chunks (23 for Binder2, 49 for Testing) while
    # the pipeline produces its own semantic chunks (11 for Testing), so
    # comparing the two squeezed every chunk onto the first pages.
    #
    # Pages are now derived from POSITION within the document text, so the
    # tests describe position instead of chunk counts.

    def _enrich_with_sidecar(self, chunks, sidecar, stem="Binder2"):
        """Enrich `chunks` against a sidecar written next to the markdown."""
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / f"{stem}.md"
            (Path(tmp) / f"{stem}_metadata.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            md_path.write_text("", encoding="utf-8")
            enricher = self._make_enricher(include_page_metadata=True)
            return enricher.enrich(chunks, md_path)

    def test_first_chunk_lands_on_the_first_page(self):
        chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(10)]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertEqual(result[0].metadata["page_numbers"], [1])

    def test_last_chunk_lands_on_the_last_page(self):
        chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(10)]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertEqual(result[-1].metadata["page_numbers"], [2])

    def test_every_page_is_reachable_not_only_the_first_ones(self):
        """
        REGRESSION: the reported bug. Testing.pdf has 6 pages, but because
        the old code compared 11 pipeline chunks against 49 OCR chunks,
        only pages 1-2 could ever be cited and pages 3-6 were unreachable
        no matter what the user asked.
        """
        chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(11)]
        result = self._enrich_with_sidecar(chunks, TESTING_META, stem="Testing")
        cited = {page for chunk in result for page in chunk.metadata["page_numbers"]}
        self.assertEqual(cited, {1, 2, 3, 4, 5, 6})

    def test_pages_never_go_backwards_through_the_document(self):
        chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(11)]
        result = self._enrich_with_sidecar(chunks, TESTING_META, stem="Testing")
        firsts = [chunk.metadata["page_numbers"][0] for chunk in result]
        self.assertEqual(firsts, sorted(firsts))

    def test_estimated_pages_are_labelled_estimated(self):
        """An inferred page must never be presented as ground truth."""
        chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(4)]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertEqual(result[0].metadata["page_source"], "estimated")

    def test_single_chunk_document_reports_the_full_page_range(self):
        """
        One chunk IS the whole document, so it genuinely covers every page.
        Reporting both pages is honest; picking one would be a guess.
        """
        result = self._enrich_with_sidecar([make_chunk()], BINDER2_META)
        self.assertEqual(result[0].metadata["page_numbers"], [1, 2])

    def test_inline_markers_win_over_estimation_and_are_exact(self):
        chunks = [make_chunk(content='<span id="page-3-0"></span>Body text here.')]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        # page-3 is 0-indexed in the marker, so this is page 4.
        self.assertEqual(result[0].metadata["page_numbers"], [4])
        self.assertEqual(result[0].metadata["page_source"], "markers")

    def test_chunk_without_marker_inherits_the_open_page(self):
        """
        Markers are boundaries, so a chunk with no marker inside it still
        sits on whichever page was open when it started. Previously such a
        chunk got no page at all.
        """
        chunks = [
            make_chunk(content='<span id="page-2-0"></span>First.', chunk_index=0),
            make_chunk(content="Second, no marker of its own.", chunk_index=1),
        ]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertEqual(result[0].metadata["page_numbers"], [3])
        self.assertEqual(result[1].metadata["page_numbers"], [3])

    def test_text_before_a_marker_belongs_to_the_previous_page(self):
        """
        A chunk that straddles a page break covers BOTH pages. Attributing
        all of it to the newly opened page is the off-by-one users see.
        """
        chunks = [
            make_chunk(content='<span id="page-0-0"></span>Page one text.', chunk_index=0),
            make_chunk(
                content='Tail of page one.<span id="page-1-0"></span>Start of page two.',
                chunk_index=1,
            ),
        ]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertEqual(result[0].metadata["page_numbers"], [1])
        self.assertEqual(result[1].metadata["page_numbers"], [1, 2])

    def test_markers_are_stripped_from_the_text_sent_to_the_model(self):
        chunks = [make_chunk(content='<span id="page-3-0"></span>Visible body.')]
        result = self._enrich_with_sidecar(chunks, BINDER2_META)
        self.assertNotIn("page-3-0", result[0].content)
        self.assertEqual(result[0].content, "Visible body.")

    def test_no_sidecar_and_no_markers_reports_no_page_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "NoMeta.md"
            md_path.write_text("# content", encoding="utf-8")
            enricher = self._make_enricher(include_page_metadata=True)
            result = enricher.enrich([make_chunk()], md_path)
            self.assertEqual(result[0].metadata["page_numbers"], [])
            self.assertEqual(result[0].metadata["page_source"], "none")

    # ── to_dict ──────────────────────────────────────────────────────────────

    def test_to_dict_structure(self):
        enricher = MetadataEnricher()
        result = enricher.enrich([make_chunk(content="Hello world.")], Path("f.md"))
        d = result[0].to_dict()
        self.assertIn("content", d)
        self.assertIn("metadata", d)
        self.assertEqual(d["content"], "Hello world.")
        self.assertIsInstance(d["metadata"], dict)

    def test_to_dict_json_serialisable(self):
        enricher = MetadataEnricher()
        result = enricher.enrich([make_chunk()], Path("f.md"))
        # Must not raise
        serialized = json.dumps(result[0].to_dict())
        self.assertIsInstance(serialized, str)

    # ── Real metadata files ──────────────────────────────────────────────────

    def test_real_binder2_metadata_loaded(self):
        path = BINDER2_METADATA_JSON
        if not path.exists():
            self.skipTest("Binder2_metadata.json not available")
        meta = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(meta["num_pages"], 2)
        self.assertEqual(len(meta["pages"]), 2)
        self.assertEqual(meta["pages"][0]["num_chunks"], 8)

    def test_real_testing_metadata_loaded(self):
        path = TESTING_METADATA_JSON
        if not path.exists():
            self.skipTest("Testing_metadata.json not available")
        meta = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(meta["num_pages"], 6)
        self.assertEqual(meta["total_images"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
