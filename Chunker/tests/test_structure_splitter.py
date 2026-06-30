"""
tests/test_structure_splitter.py

Unit tests for StructureSplitter — derived from real Chandra OCR patterns.
Every guarantee in the docstring is tested:
  - Tables are NEVER split
  - Each section carries the correct breadcrumb
  - Image-description placeholders are tagged correctly
  - Small sections are merged upward into previous text sections
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import StructureSplitterConfig
from Chunker.splitters.structure_splitter import (
    StructureSplitter,
    StructuralSection,
    _build_breadcrumb,
    _pop_to_level,
)
from Chunker.tests.fixtures_path import BINDER2_MD, TESTING_MD

# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

# Binder2.md pattern: file starts at ### with NO parent #/##
BINDER2_STYLE = """\
### SERVICE: Great circle

This service allows the operator to manipulate great circles (abbreviated GCI).

### MMI Disposition

[IMAGE_DESC_PLACEHOLDER_0]

### Operator actions

<table border="1">
<thead><tr><th>Operator action</th><th>Entry</th></tr></thead>
<tbody><tr><td>Create SOI</td><td>Enter NT</td></tr></tbody>
</table>

### System reports

<table border="1">
<thead><tr><th>System report</th><th>Restrictions</th></tr></thead>
<tbody><tr><td>CMI Report</td><td>-</td></tr></tbody>
</table>
"""

# Testing.md pattern: mixed #, ##, ### hierarchy
TESTING_STYLE = """\
# Prompt Engineering

Author: Lee Boonstra

## Introduction

When thinking about a large language model input and output, a text prompt
is the input the model uses to predict a specific output.

<table border="1">
<tr><td>Name</td><td>movie_classification</td></tr>
<tr><td>Goal</td><td>Classify movie reviews</td></tr>
</table>

## One-shot & few-shot

When creating prompts for AI models, it is helpful to provide examples.
"""

# Two tables adjacent in same section (already fixed by preprocessor)
TWO_TABLES_SAME_SECTION = """\
### System reports

<table border="1"><tr><td>EN Table</td></tr></table>

<table border="1"><tr><td>ES Table</td></tr></table>
"""

# Section with text BEFORE the first header
PRE_HEADER_TEXT = """\
Author: Lee Boonstra

## Introduction

Some introduction text here.
"""

# Header with empty body (only next header follows)
EMPTY_BODY_HEADERS = """\
### Header One

### Header Two

Some content under header two.
"""

# Text that should be merged (below min_section_chars)
SMALL_SECTION = """\
### Section A

This is a full and complete section with enough characters to survive.

### Section B

Short.
"""


class TestBuildBreadcrumb(unittest.TestCase):

    def test_empty_returns_root(self):
        self.assertEqual(_build_breadcrumb([], []), "(root)")

    def test_single_level_one(self):
        self.assertEqual(_build_breadcrumb(["Title"], [1]), "# Title")

    def test_single_level_three(self):
        self.assertEqual(_build_breadcrumb(["Sub"], [3]), "### Sub")

    def test_multi_level_chain(self):
        result = _build_breadcrumb(
            ["Prompt Engineering", "Introduction"],
            [1, 2]
        )
        self.assertEqual(result, "# Prompt Engineering > ## Introduction")

    def test_three_level_chain(self):
        result = _build_breadcrumb(
            ["Root", "Section", "SubSection"],
            [1, 2, 3]
        )
        self.assertIn(">", result)
        self.assertIn("### SubSection", result)

    def test_binder2_pattern_starts_at_three(self):
        """Binder2.md starts at ###, no # or ## parent."""
        result = _build_breadcrumb(["SERVICE: Great circle"], [3])
        self.assertEqual(result, "### SERVICE: Great circle")


class TestPopToLevel(unittest.TestCase):

    def test_pop_same_level(self):
        stack = [(3, "Old")]
        result = _pop_to_level(stack, 3)
        self.assertEqual(result, [])  # (3 < 3) is False

    def test_pop_deeper_level(self):
        stack = [(1, "Root"), (2, "Sub"), (3, "SubSub")]
        result = _pop_to_level(stack, 2)
        self.assertEqual(result, [(1, "Root")])

    def test_pop_none_when_all_shallower(self):
        stack = [(1, "Root")]
        result = _pop_to_level(stack, 2)
        self.assertEqual(result, [(1, "Root")])

    def test_pop_all_when_new_is_level_one(self):
        stack = [(1, "Old"), (2, "Sub"), (3, "SubSub")]
        result = _pop_to_level(stack, 1)
        self.assertEqual(result, [])

    def test_empty_stack(self):
        self.assertEqual(_pop_to_level([], 3), [])


class TestStructureSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = StructureSplitter()

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_list_of_structural_sections(self):
        sections = self.splitter.split("### Header\n\nSome content here with text.")
        self.assertIsInstance(sections, list)
        for sec in sections:
            self.assertIsInstance(sec, StructuralSection)

    def test_empty_text_returns_empty_list(self):
        sections = self.splitter.split("")
        self.assertEqual(sections, [])

    def test_source_path_propagated(self):
        sections = self.splitter.split(
            "### Header\n\nContent with enough characters to survive merging.",
            source_path="path/to/file.md"
        )
        for sec in sections:
            self.assertEqual(sec.source_path, "path/to/file.md")

    # ── Section types ────────────────────────────────────────────────────────

    def test_table_section_type(self):
        text = "### Header\n\n<table><tr><td>Data</td></tr></table>"
        sections = self.splitter.split(text)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertGreater(len(table_secs), 0)

    def test_text_section_type(self):
        text = (
            "### Header\n\n"
            "This is a sufficiently long section of text content that will not be merged."
        )
        sections = self.splitter.split(text)
        text_secs = [s for s in sections if s.section_type == "text"]
        self.assertGreater(len(text_secs), 0)

    def test_image_placeholder_section_type(self):
        text = "### MMI Disposition\n\n[IMAGE_DESC_PLACEHOLDER_0]"
        sections = self.splitter.split(text)
        img_secs = [s for s in sections if s.section_type == "image_description"]
        self.assertGreater(len(img_secs), 0)

    def test_valid_section_types_only(self):
        sections = self.splitter.split(BINDER2_STYLE)
        valid_types = {"text", "table", "image_description"}
        for sec in sections:
            self.assertIn(sec.section_type, valid_types)

    # ── Table atomicity ──────────────────────────────────────────────────────

    def test_table_kept_as_single_section(self):
        """A <table> must NEVER be broken into multiple sections."""
        text = (
            "### Actions\n\n"
            "<table border='1'>"
            "<thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody>"
            "<tr><td>Create</td><td>Enter NT</td></tr>"
            "<tr><td>Delete</td><td>Select and remove</td></tr>"
            "<tr><td>Modify</td><td>Select and update</td></tr>"
            "</tbody>"
            "</table>"
        )
        sections = self.splitter.split(text)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertEqual(len(table_secs), 1)
        # All rows must be in the one table section
        self.assertIn("Create", table_secs[0].content)
        self.assertIn("Delete", table_secs[0].content)
        self.assertIn("Modify", table_secs[0].content)

    def test_table_contains_full_html(self):
        text = "### H\n\n<table><tr><td>Row1</td></tr><tr><td>Row2</td></tr></table>"
        sections = self.splitter.split(text)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertEqual(len(table_secs), 1)
        self.assertIn("<table", table_secs[0].content)
        self.assertIn("</table>", table_secs[0].content)

    def test_two_tables_in_section_both_atomic(self):
        """Two tables in same section each get their own table section."""
        sections = self.splitter.split(TWO_TABLES_SAME_SECTION)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertEqual(len(table_secs), 2)
        contents = [s.content for s in table_secs]
        self.assertTrue(any("EN Table" in c for c in contents))
        self.assertTrue(any("ES Table" in c for c in contents))

    def test_table_headers_in_table_not_split_on(self):
        """A markdown-like line inside a table value must not create a new header section."""
        text = (
            "### Actions\n\n"
            "<table><tr><td># Not a real header</td></tr></table>\n\n"
            "Some trailing text here for context."
        )
        sections = self.splitter.split(text)
        # The "# Not a real header" inside the table must not create a level-1 section
        breadcrumbs = [s.breadcrumb for s in sections]
        self.assertNotIn("# Not a real header", " ".join(breadcrumbs))

    # ── Breadcrumb correctness ────────────────────────────────────────────────

    def test_binder2_starts_at_h3_breadcrumb(self):
        """Binder2.md starts at ###, breadcrumb has no parent."""
        sections = self.splitter.split(BINDER2_STYLE)
        # All section breadcrumbs should start with ###
        for sec in sections:
            if sec.breadcrumb != "(root)":
                self.assertIn("###", sec.breadcrumb)

    def test_testing_multi_level_breadcrumb(self):
        sections = self.splitter.split(TESTING_STYLE)
        # The Introduction section should have both # and ## in breadcrumb
        intro_secs = [s for s in sections if "Introduction" in s.breadcrumb]
        self.assertGreater(len(intro_secs), 0)
        bc = intro_secs[0].breadcrumb
        self.assertIn("# Prompt Engineering", bc)
        self.assertIn("## Introduction", bc)

    def test_breadcrumb_resets_at_same_level(self):
        """A second ## header should not carry context from the first ##."""
        text = (
            "# Root\n\n"
            "## Section A\n\nContent A with enough chars to not be merged away.\n\n"
            "## Section B\n\nContent B with enough chars to not be merged away.\n\n"
        )
        sections = self.splitter.split(text)
        b_secs = [s for s in sections if "Section B" in s.breadcrumb]
        self.assertGreater(len(b_secs), 0)
        # Section B breadcrumb must NOT contain Section A
        self.assertNotIn("Section A", b_secs[0].breadcrumb)

    def test_header_texts_list_matches_breadcrumb(self):
        sections = self.splitter.split(TESTING_STYLE)
        for sec in sections:
            self.assertEqual(len(sec.header_texts), len(sec.header_levels))

    def test_pre_header_text_gets_root_breadcrumb(self):
        """Text appearing before the first header → breadcrumb == '(root)'."""
        sections = self.splitter.split(PRE_HEADER_TEXT)
        root_secs = [s for s in sections if s.breadcrumb == "(root)"]
        self.assertGreater(len(root_secs), 0)
        self.assertTrue(any("Author" in s.content for s in root_secs))

    def test_section_header_texts_correct(self):
        text = "### Operator actions\n\nEnough content here to not be merged away from parent."
        sections = self.splitter.split(text)
        secs_with_header = [s for s in sections if s.header_texts]
        self.assertTrue(any("Operator actions" in s.header_texts for s in secs_with_header))

    # ── Small section merging ─────────────────────────────────────────────────

    def test_small_text_section_merged_into_previous(self):
        """A text section below min_section_chars is merged into the previous text section."""
        cfg = StructureSplitterConfig(min_section_chars=50)
        splitter = StructureSplitter(cfg)
        sections = splitter.split(SMALL_SECTION)
        text_secs = [s for s in sections if s.section_type == "text"]
        # "Short." is 6 chars (< 50), so it should merge into "Section A" content
        # Verify "Short." doesn't appear as a standalone section
        standalone_short = [s for s in text_secs if s.content.strip() == "Short."]
        self.assertEqual(len(standalone_short), 0)

    def test_table_never_merged_even_if_small(self):
        """Tables must never be merged into an adjacent text section."""
        text = (
            "### Section\n\n"
            "Long enough text content to be a proper text section on its own.\n\n"
            "<table><tr><td>T</td></tr></table>"
        )
        sections = self.splitter.split(text)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertEqual(len(table_secs), 1)
        # Table must be its own section, not merged
        self.assertIn("<table", table_secs[0].content)

    def test_image_description_never_merged(self):
        """Image description sections must remain separate (not merged into text)."""
        text = (
            "### Section\n\n"
            "Some text content with enough characters to stand alone.\n\n"
            "[IMAGE_DESC_PLACEHOLDER_0]"
        )
        sections = self.splitter.split(text)
        img_secs = [s for s in sections if s.section_type == "image_description"]
        self.assertGreater(len(img_secs), 0)

    def test_empty_list_input(self):
        self.assertEqual(self.splitter.split(""), [])

    # ── Header level filtering ────────────────────────────────────────────────

    def test_header_levels_config_filters(self):
        """With header_levels=[1,2], level-3 headers are treated as body text."""
        cfg = StructureSplitterConfig(header_levels=[1, 2])
        splitter = StructureSplitter(cfg)
        text = (
            "## Top\n\n"
            "Some introductory text here.\n\n"
            "### Sub\n\nSub content here."
        )
        sections = splitter.split(text)
        # "### Sub" must NOT create a new section with a breadcrumb containing "Sub"
        breadcrumbs = [s.breadcrumb for s in sections]
        sub_breadcrumbs = [b for b in breadcrumbs if "### Sub" in b]
        self.assertEqual(len(sub_breadcrumbs), 0,
                         "Level-3 header should not appear in breadcrumbs when filtered")

    # ── Binder2.md real file ─────────────────────────────────────────────────

    def test_binder2_real_file_section_counts(self):
        path = BINDER2_MD
        if not path.exists():
            self.skipTest("Binder2.md not available")
        from Chunker.preprocessors.md_preprocessor import MarkdownPreprocessor
        raw = path.read_text(encoding="utf-8")
        doc = MarkdownPreprocessor().process(raw)
        sections = self.splitter.split(doc.cleaned_text)
        table_secs = [s for s in sections if s.section_type == "table"]
        self.assertGreaterEqual(len(table_secs), 4,
                                "Binder2.md has at least 4 tables")

    def test_testing_real_file_hierarchy(self):
        path = TESTING_MD
        if not path.exists():
            self.skipTest("Testing.md not available")
        from Chunker.preprocessors.md_preprocessor import MarkdownPreprocessor
        raw = path.read_text(encoding="utf-8")
        doc = MarkdownPreprocessor().process(raw)
        sections = self.splitter.split(doc.cleaned_text)
        # Testing.md has # and ## headers — breadcrumbs should contain ">"
        deep_breadcrumbs = [s for s in sections if ">" in s.breadcrumb]
        self.assertGreater(len(deep_breadcrumbs), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
