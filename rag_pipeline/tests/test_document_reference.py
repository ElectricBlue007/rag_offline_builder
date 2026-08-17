"""
tests/test_document_reference.py

Tests for document_reference.py and its three consumers (metadata_enricher,
context_builder, rag_answerer).

The behaviour being protected: a reader must be able to take a citation out
of the model's answer and find the exact place in the original document --
which file, which page, which chapter, which section.

Two rules matter most and each has its own test:
  1. A corpus ingested BEFORE these fields existed must still produce
     chapter/section references (derived from the old breadcrumb string),
     so nobody is forced to re-ingest.
  2. Stored reference fields must stay flat scalars, because ChromaDB
     rejects list-valued metadata.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

from rag_pipeline.config import ContextBuilderConfig, MetadataConfig
from rag_pipeline.document_reference import (
    LEVEL_LABELS_EN,
    LEVEL_LABELS_FA,
    clean_heading,
    format_reference,
    heading_chain,
    heading_levels,
    label_for_depth,
    pages_display,
    reference_fields,
    source_display,
    structured_reference,
)
from rag_pipeline.enrichers.metadata_enricher import MetadataEnricher
from rag_pipeline.generation.context_builder import build_context
from rag_pipeline.generation.rag_answerer import RagAnswerer
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


FULL_META = {
    "source_pdf": "Binder2.pdf",
    "source_file": "data/Binder2.md",
    "page_numbers": [12],
    "header_texts": ["Prompt Engineering", "Few-shot", "Worked example"],
    "header_levels": [1, 2, 3],
}


# ── The pure helpers ──────────────────────────────────────────────────────

class TestCleanHeading(unittest.TestCase):
    def test_strips_markdown_hashes(self):
        self.assertEqual(clean_heading("### System reports"), "System reports")

    def test_leaves_plain_text_alone(self):
        self.assertEqual(clean_heading("System reports"), "System reports")

    def test_handles_empty(self):
        self.assertEqual(clean_heading(""), "")
        self.assertEqual(clean_heading(None), "")


class TestHeadingChain(unittest.TestCase):
    def test_prefers_explicit_header_texts(self):
        self.assertEqual(
            heading_chain(FULL_META),
            ["Prompt Engineering", "Few-shot", "Worked example"],
        )

    def test_falls_back_to_breadcrumb_for_legacy_chunks(self):
        """A corpus ingested before this feature only has the breadcrumb."""
        legacy = {"breadcrumb": "# Prompt Engineering > ## Few-shot"}
        self.assertEqual(heading_chain(legacy), ["Prompt Engineering", "Few-shot"])

    def test_falls_back_to_heading_path(self):
        self.assertEqual(
            heading_chain({"heading_path": "A > B"}),
            ["A", "B"],
        )

    def test_root_breadcrumb_is_not_a_heading(self):
        self.assertEqual(heading_chain({"breadcrumb": "(root)"}), [])
        self.assertEqual(heading_chain({"breadcrumb": ""}), [])
        self.assertEqual(heading_chain({}), [])

    def test_accepts_comma_string_from_chroma(self):
        """ChromaDB flattens lists into comma-separated strings."""
        self.assertEqual(
            heading_chain({"header_texts": "Chapter One,Section Two"}),
            ["Chapter One", "Section Two"],
        )


class TestLabels(unittest.TestCase):
    def test_labels_by_position(self):
        self.assertEqual(label_for_depth(0), "Chapter")
        self.assertEqual(label_for_depth(1), "Section")
        self.assertEqual(label_for_depth(2), "Subsection")

    def test_deeper_levels_reuse_the_last_label(self):
        self.assertEqual(label_for_depth(7), LEVEL_LABELS_EN[-1])

    def test_persian_labels_available(self):
        self.assertEqual(label_for_depth(0, persian=True), LEVEL_LABELS_FA[0])

    def test_labels_follow_position_not_hash_depth(self):
        """A document whose top heading is '##' still has a chapter."""
        levels = heading_levels({
            "header_texts": ["Manual", "Setup"],
            "header_levels": [2, 4],
        })
        self.assertEqual(levels[0]["label"], "Chapter")
        self.assertEqual(levels[0]["title"], "Manual")
        self.assertEqual(levels[0]["markdown_level"], 2)
        self.assertEqual(levels[1]["label"], "Section")


class TestPagesDisplay(unittest.TestCase):
    def test_single_page(self):
        self.assertEqual(pages_display({"page_numbers": [12]}), "12")

    def test_consecutive_pages_collapse_to_a_range(self):
        self.assertEqual(pages_display({"page_numbers": [4, 5, 6]}), "4-6")

    def test_gaps_are_kept_separate(self):
        self.assertEqual(pages_display({"page_numbers": [4, 9]}), "4, 9")

    def test_accepts_comma_string_from_chroma(self):
        self.assertEqual(pages_display({"page_numbers": "4,5"}), "4-5")

    def test_empty_when_ocr_had_no_page_markers(self):
        self.assertEqual(pages_display({"page_numbers": []}), "")
        self.assertEqual(pages_display({}), "")

    def test_non_numeric_labels_are_preserved(self):
        self.assertEqual(pages_display({"page_numbers": ["iv"]}), "iv")


class TestSourceDisplay(unittest.TestCase):
    def test_prefers_the_original_pdf_name(self):
        self.assertEqual(source_display(FULL_META), "Binder2.pdf")

    def test_falls_back_to_markdown_file(self):
        self.assertEqual(source_display({"source_file": "a/b/Testing.md"}), "Testing.md")

    def test_strips_directories(self):
        self.assertEqual(source_display({"source_pdf": "docs/x/Book.pdf"}), "Book.pdf")


class TestStructuredReference(unittest.TestCase):
    def setUp(self):
        self.reference = structured_reference(FULL_META)

    def test_splits_the_hierarchy_into_named_parts(self):
        self.assertEqual(self.reference["chapter"], "Prompt Engineering")
        self.assertEqual(self.reference["section"], "Few-shot")
        self.assertEqual(self.reference["subsection"], "Worked example")

    def test_includes_source_and_page(self):
        self.assertEqual(self.reference["source"], "Binder2.pdf")
        self.assertEqual(self.reference["pages"], "12")

    def test_full_text_is_readable(self):
        self.assertEqual(
            self.reference["text"],
            "Binder2.pdf, page 12 - Chapter: Prompt Engineering > "
            "Section: Few-shot > Subsection: Worked example",
        )

    def test_short_text_for_cramped_ui(self):
        self.assertEqual(self.reference["short"], "Binder2.pdf p.12 - Worked example")

    def test_missing_pieces_degrade_gracefully(self):
        reference = structured_reference({"source_file": "Notes.md"})
        self.assertEqual(reference["text"], "Notes.md")
        self.assertEqual(reference["chapter"], "")
        self.assertEqual(reference["pages"], "")

    def test_empty_metadata_never_raises(self):
        self.assertEqual(structured_reference({})["text"], "")
        self.assertEqual(structured_reference(None)["text"], "")


class TestFormatReference(unittest.TestCase):
    def test_pages_plural_for_a_range(self):
        text = format_reference({"source_pdf": "B.pdf", "page_numbers": [4, 5]})
        self.assertIn("pages 4-5", text)

    def test_singular_for_one_page(self):
        text = format_reference({"source_pdf": "B.pdf", "page_numbers": [4]})
        self.assertIn("page 4", text)

    def test_parts_can_be_switched_off(self):
        text = format_reference(FULL_META, include_source=False, include_pages=False)
        self.assertNotIn("Binder2.pdf", text)
        self.assertNotIn("12", text)
        self.assertIn("Chapter: Prompt Engineering", text)


class TestReferenceFields(unittest.TestCase):
    def setUp(self):
        self.fields = reference_fields(FULL_META)

    def test_stores_the_named_levels(self):
        self.assertEqual(self.fields["chapter"], "Prompt Engineering")
        self.assertEqual(self.fields["section"], "Few-shot")
        self.assertEqual(self.fields["subsection"], "Worked example")
        self.assertEqual(self.fields["heading_depth"], 3)

    def test_all_values_are_chroma_safe_scalars(self):
        """ChromaDB rejects list/dict metadata values."""
        for key, value in self.fields.items():
            self.assertIsInstance(value, (str, int, float, bool), f"{key} is {type(value)}")


# ── Consumer 1: metadata_enricher ─────────────────────────────────────────

class TestEnricherAttachesReference(unittest.TestCase):
    def _enrich(self, **kwargs):
        chunk = SemanticChunk(
            content=kwargs.get("content", "Body text."),
            section_type="text",
            breadcrumb=kwargs.get("breadcrumb", "# Prompt Engineering > ## Few-shot"),
            header_levels=kwargs.get("header_levels", [1, 2]),
            header_texts=kwargs.get("header_texts", ["Prompt Engineering", "Few-shot"]),
            source_path="Binder2.md",
            chunk_index=0,
        )
        enricher = MetadataEnricher(MetadataConfig())
        return enricher.enrich([chunk], Path("Binder2.md"))[0].metadata

    def test_chapter_and_section_are_stored(self):
        meta = self._enrich()
        self.assertEqual(meta["chapter"], "Prompt Engineering")
        self.assertEqual(meta["section"], "Few-shot")
        self.assertEqual(meta["heading_path"], "Prompt Engineering > Few-shot")

    def test_reference_string_is_ready_to_display(self):
        meta = self._enrich()
        self.assertIn("Chapter: Prompt Engineering", meta["reference"])
        self.assertIn("Section: Few-shot", meta["reference"])

    def test_page_numbers_from_ocr_markers_reach_the_reference(self):
        meta = self._enrich(content='<span id="page-11-0"></span>Body text.')
        self.assertEqual(meta["page_numbers"], [12])
        self.assertEqual(meta["page_display"], "12")
        self.assertIn("page 12", meta["reference"])

    def test_chunk_with_no_headings_still_gets_a_reference(self):
        meta = self._enrich(breadcrumb="(root)", header_levels=[], header_texts=[])
        self.assertEqual(meta["chapter"], "")
        self.assertIn("Binder2", meta["reference"])


# ── Consumer 2: context_builder ───────────────────────────────────────────

class TestContextShowsLocation(unittest.TestCase):
    def _chunk(self, metadata):
        return SimpleNamespace(chunk_id="c1", text="Body text.", metadata=metadata)

    def test_header_labels_chapter_and_section(self):
        context = build_context([self._chunk(FULL_META)], ContextBuilderConfig())
        self.assertIn("Chapter: Prompt Engineering", context)
        self.assertIn("Section: Few-shot", context)
        self.assertIn("page 12", context)
        self.assertIn("Binder2.pdf", context)

    def test_legacy_chunks_still_get_labelled_levels(self):
        legacy = {"source_file": "Binder2.md", "breadcrumb": "# A > ## B"}
        context = build_context([self._chunk(legacy)], ContextBuilderConfig())
        self.assertIn("Chapter: A", context)
        self.assertIn("Section: B", context)

    def test_labels_can_be_disabled(self):
        cfg = ContextBuilderConfig(label_heading_levels=False)
        context = build_context([self._chunk(FULL_META)], cfg)
        self.assertNotIn("Chapter:", context)
        self.assertIn("Prompt Engineering > Few-shot", context)

    def test_citation_number_precedes_the_location(self):
        context = build_context([self._chunk(FULL_META)], ContextBuilderConfig())
        self.assertTrue(context.startswith("[1] "))


# ── Consumer 3: rag_answerer ──────────────────────────────────────────────

class _FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query):
        return self.chunks


class _FakeLlm:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def generate(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "Few-shot uses examples [1] (Binder2.pdf, page 12 - Chapter: " \
               "Prompt Engineering > Section: Few-shot)."


class TestAnswererReportsLocation(unittest.TestCase):
    def setUp(self):
        chunk = SimpleNamespace(chunk_id="c1", text="Body.", metadata=dict(FULL_META))
        self.llm = _FakeLlm()
        self.answerer = RagAnswerer(
            retriever=_FakeRetriever([chunk]),
            llm_client=self.llm,
            context_config=ContextBuilderConfig(),
        )
        self.result = self.answerer.answer("What is few-shot?")

    def test_sources_carry_the_structured_location(self):
        source = self.result.sources[0]
        self.assertEqual(source["chapter"], "Prompt Engineering")
        self.assertEqual(source["section"], "Few-shot")
        self.assertEqual(source["subsection"], "Worked example")
        self.assertEqual(source["pages"], "12")
        self.assertEqual(source["source"], "Binder2.pdf")

    def test_source_knows_its_own_citation_number(self):
        self.assertEqual(self.result.sources[0]["citation_number"], 1)

    def test_levels_are_labelled_for_display(self):
        levels = self.result.sources[0]["levels"]
        self.assertEqual([lvl["label"] for lvl in levels],
                         ["Chapter", "Section", "Subsection"])
        self.assertTrue(all(lvl["label_fa"] for lvl in levels))

    def test_system_prompt_asks_for_the_location(self):
        self.assertIn("Chapter", self.llm.system_prompt)
        self.assertIn("page number", self.llm.system_prompt)

    def test_location_instruction_can_be_disabled(self):
        llm = _FakeLlm()
        chunk = SimpleNamespace(chunk_id="c1", text="Body.", metadata=dict(FULL_META))
        answerer = RagAnswerer(
            retriever=_FakeRetriever([chunk]),
            llm_client=llm,
            context_config=ContextBuilderConfig(cite_document_location=False),
        )
        answerer.answer("What is few-shot?")
        self.assertNotIn("repeat that location", llm.system_prompt)

    def test_legacy_chunk_still_reports_chapter_and_section(self):
        """Works on a chroma_db built before this feature existed."""
        legacy = SimpleNamespace(
            chunk_id="c9",
            text="Body.",
            metadata={"source_file": "Binder2.md", "breadcrumb": "# A > ## B",
                      "page_numbers": [3]},
        )
        answerer = RagAnswerer(
            retriever=_FakeRetriever([legacy]),
            llm_client=_FakeLlm(),
            context_config=ContextBuilderConfig(),
        )
        source = answerer.answer("q").sources[0]
        self.assertEqual(source["chapter"], "A")
        self.assertEqual(source["section"], "B")
        self.assertEqual(source["pages"], "3")


if __name__ == "__main__":
    unittest.main()
