"""
tests/test_md_preprocessor.py

Unit tests for MarkdownPreprocessor — every case derived from real
Chandra OCR output patterns observed in Binder2.md and Testing.md.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import PreprocessorConfig
from rag_pipeline.preprocessors.md_preprocessor import (
    MarkdownPreprocessor,
    ImageAltChunk,
    PreprocessedDocument,
    _is_toc_table,
    _fix_adjacent_tables,
    _extract_images,
    _remove_toc_tables,
    _strip_superscripts,
    _strip_decorative_lines,
    _collapse_blank_lines,
)
from rag_pipeline.tests.fixtures_path import BINDER2_MD, TESTING_MD

# ════════════════════════════════════════════════════════════════════
# Fixtures — exact patterns from real Chandra OCR output
# ════════════════════════════════════════════════════════════════════

ADJACENT_TABLES = """\
### System reports

<table border="1">
<thead><tr><th>System report</th><th>Restrictions</th></tr></thead>
<tbody><tr><td>CMI Report</td><td>-</td></tr></tbody>
</table><table border="1">
<thead><tr><th>Acción del operador</th><th>Restricciones</th></tr></thead>
<tbody><tr><td>Imprimir</td><td>-</td></tr></tbody>
</table>
"""

TOC_TABLE = """\
# Table of contents

<table><tr><td>Introduction .....</td><td>6</td></tr>
<tr><td>Prompt engineering .....</td><td>7</td></tr></table>

Some text after.
"""

FUSED_IMAGES = (
    "![Google logo](a5c52eb326dc00fc9e10c45812a6fcfb_3_img.webp)"
    "The Google logo is here."
    "![Abstract shape](a5c52eb326dc00fc9e10c45812a6fcfb_4_img.webp)"
    "A large abstract shape."
)

SUPERSCRIPT_TEXT = (
    "El boton no esta disponible (ajustado por parametro del sistema"
    "<sup>d</sup>: Nombre de grupo = LSSCONF)"
)

MANY_BLANK_LINES = "First paragraph.\n\n\n\n\nSecond paragraph."


class TestHelperFunctions(unittest.TestCase):

    # ── _is_toc_table ────────────────────────────────────────────────────────

    def test_toc_detected_by_five_dots(self):
        toc = "<table><tr><td>Introduction .....</td><td>6</td></tr></table>"
        self.assertTrue(_is_toc_table(toc))

    def test_toc_not_detected_by_three_dot_ellipsis(self):
        """
        A bare ellipsis "..." (3 dots) must NOT trigger TOC detection —
        this exact pattern appears in real OCR output inside <pre> code
        blocks (e.g. the pizza-order JSON example in Testing.md) and must
        not cause a legitimate table to be deleted.
        """
        not_toc = "<table><tr><td>Item...</td></tr></table>"
        self.assertFalse(_is_toc_table(not_toc))

    def test_normal_table_not_toc(self):
        normal = "<table><tr><th>Action</th><th>Entry</th></tr></table>"
        self.assertFalse(_is_toc_table(normal))

    def test_toc_many_dot_variants(self):
        for dots in [".....", ".......", "........"]:
            tbl = f"<table><tr><td>Intro {dots}</td></tr></table>"
            self.assertTrue(_is_toc_table(tbl))

    # ── _fix_adjacent_tables ─────────────────────────────────────────────────

    def test_adjacent_tables_get_blank_line(self):
        result = _fix_adjacent_tables(ADJACENT_TABLES)
        self.assertIn("</table>\n\n<table", result)

    def test_fix_adjacent_idempotent(self):
        once = _fix_adjacent_tables(ADJACENT_TABLES)
        twice = _fix_adjacent_tables(once)
        self.assertEqual(once, twice)

    def test_already_separated_tables_unchanged(self):
        text = "<table><tr><td>A</td></tr></table>\n\n<table><tr><td>B</td></tr></table>"
        self.assertEqual(_fix_adjacent_tables(text), text)

    def test_adjacent_tables_case_insensitive(self):
        text = "<TABLE><TR><TD>A</TD></TR></TABLE><table><tr><td>B</td></tr></table>"
        result = _fix_adjacent_tables(text)
        # There should now be a separator
        import re
        self.assertTrue(re.search(r'</TABLE>\s*\n\n\s*<table', result, re.IGNORECASE))

    # ── _extract_images ──────────────────────────────────────────────────────

    def test_strategy_drop_removes_everything(self):
        text = "Before ![Alt text](img.webp) After"
        cleaned, alts = _extract_images(text, "drop")
        self.assertNotIn("![", cleaned)
        self.assertNotIn(".webp", cleaned)
        self.assertNotIn("Alt text", cleaned)
        self.assertEqual(alts, [])

    def test_strategy_keep_inlines_alt(self):
        text = "Before ![Alt text](img.webp) After"
        cleaned, alts = _extract_images(text, "keep")
        self.assertIn("Alt text", cleaned)
        self.assertNotIn("![", cleaned)
        self.assertNotIn(".webp", cleaned)
        self.assertEqual(alts, [])

    def test_strategy_separate_creates_placeholder(self):
        text = "Before ![Alt text](img.webp) After"
        cleaned, alts = _extract_images(text, "separate")
        self.assertIn("[IMAGE_DESC_PLACEHOLDER_0]", cleaned)
        self.assertNotIn(".webp", cleaned)
        self.assertEqual(len(alts), 1)
        self.assertEqual(alts[0].alt_text, "Alt text")

    def test_two_fused_images_both_captured(self):
        """Testing.md pattern: two images concatenated with no separator."""
        cleaned, alts = _extract_images(FUSED_IMAGES, "separate")
        self.assertEqual(len(alts), 2)
        self.assertIn("[IMAGE_DESC_PLACEHOLDER_0]", cleaned)
        self.assertIn("[IMAGE_DESC_PLACEHOLDER_1]", cleaned)
        self.assertEqual(alts[0].alt_text, "Google logo")
        self.assertEqual(alts[1].alt_text, "Abstract shape")

    def test_empty_alt_text_captured(self):
        text = "![](no_alt.webp)"
        cleaned, alts = _extract_images(text, "separate")
        self.assertEqual(len(alts), 1)
        self.assertEqual(alts[0].alt_text, "")

    def test_non_webp_images_not_touched(self):
        text = "![Logo](logo.png) and ![Icon](icon.jpg)"
        cleaned, alts = _extract_images(text, "separate")
        self.assertEqual(cleaned, text)
        self.assertEqual(alts, [])

    def test_alt_text_whitespace_stripped(self):
        text = "![  Spaced alt text  ](img.webp)"
        _, alts = _extract_images(text, "separate")
        self.assertEqual(alts[0].alt_text, "Spaced alt text")

    def test_no_images_returns_unchanged(self):
        text = "Plain text with no images at all."
        cleaned, alts = _extract_images(text, "separate")
        self.assertEqual(cleaned, text)
        self.assertEqual(alts, [])

    def test_image_positions_monotonically_increasing(self):
        text = "AAA ![img1](a.webp) BBB ![img2](b.webp)"
        _, alts = _extract_images(text, "separate")
        self.assertEqual(len(alts), 2)
        self.assertLess(alts[0].position, alts[1].position)

    # ── _remove_toc_tables ───────────────────────────────────────────────────

    def test_toc_table_removed(self):
        result = _remove_toc_tables(TOC_TABLE)
        self.assertNotIn("Introduction .....", result)
        self.assertNotIn("Prompt engineering .....", result)

    def test_normal_table_kept(self):
        text = "<table><tr><td>Operator action</td><td>Entry</td></tr></table>"
        result = _remove_toc_tables(text)
        self.assertIn("Operator action", result)

    def test_surrounding_text_kept(self):
        result = _remove_toc_tables(TOC_TABLE)
        self.assertIn("Some text after.", result)

    def test_mixed_toc_and_real_tables(self):
        mixed = (
            "<table><tr><td>Intro.....</td></tr></table>\n\n"
            "<table><tr><td>Operator action</td></tr></table>"
        )
        result = _remove_toc_tables(mixed)
        self.assertNotIn("Intro.....", result)
        self.assertIn("Operator action", result)

    # ── _strip_superscripts ──────────────────────────────────────────────────

    def test_strip_sup_d(self):
        """<sup>d</sup> from Binder2.md."""
        result = _strip_superscripts(SUPERSCRIPT_TEXT)
        self.assertNotIn("<sup>", result)
        self.assertNotIn("</sup>", result)

    def test_strip_sup_number(self):
        text = "A few-shot prompt <sup>7</sup> provides multiple examples."
        result = _strip_superscripts(text)
        self.assertNotIn("<sup>7</sup>", result)
        self.assertIn("provides multiple examples.", result)

    def test_strip_sup_multiple(self):
        text = "A<sup>1</sup> and B<sup>2</sup>."
        result = _strip_superscripts(text)
        self.assertNotIn("<sup>", result)
        self.assertIn("A and B.", result)

    def test_strip_sup_case_insensitive(self):
        text = "Text<SUP>d</SUP>more."
        result = _strip_superscripts(text)
        self.assertNotIn("SUP", result)

    def test_no_superscripts_unchanged(self):
        text = "Plain text without superscripts."
        self.assertEqual(_strip_superscripts(text), text)

    # ── _strip_decorative_lines ──────────────────────────────────────────────

    def test_strip_continues_next_page_bare(self):
        text = "Real content.\nContinues next page\nMore."
        result = _strip_decorative_lines(text)
        self.assertNotIn("Continues next page", result)
        self.assertIn("Real content.", result)

    def test_strip_continues_with_period(self):
        text = "A.\nContinues next page.\nB."
        result = _strip_decorative_lines(text)
        self.assertNotIn("Continues next page", result)

    def test_strip_continues_uppercase(self):
        text = "A.\nCONTINUES NEXT PAGE\nB."
        result = _strip_decorative_lines(text)
        self.assertNotIn("CONTINUES NEXT PAGE", result)

    def test_strip_press_enter(self):
        text = "Content.\nPress enter or click to view image in full size\nMore."
        result = _strip_decorative_lines(text)
        self.assertNotIn("Press enter", result)
        self.assertIn("Content.", result)

    def test_decorative_no_false_positives(self):
        """Must not strip lines that merely contain those words in context."""
        safe = "This text continues next page numbering for legal reasons."
        result = _strip_decorative_lines(safe)
        self.assertEqual(result, safe)

    # ── _collapse_blank_lines ────────────────────────────────────────────────

    def test_collapse_five_blank_lines(self):
        result = _collapse_blank_lines(MANY_BLANK_LINES)
        self.assertNotIn("\n\n\n", result)
        self.assertIn("First paragraph.", result)

    def test_double_blank_preserved(self):
        text = "A\n\nB"
        self.assertEqual(_collapse_blank_lines(text), text)

    def test_single_blank_preserved(self):
        text = "A\nB"
        self.assertEqual(_collapse_blank_lines(text), text)


class TestMarkdownPreprocessor(unittest.TestCase):

    def setUp(self):
        self.prep = MarkdownPreprocessor()

    # ── Output type and invariants ───────────────────────────────────────────

    def test_returns_preprocessed_document(self):
        doc = self.prep.process("Hello world.", "test.md")
        self.assertIsInstance(doc, PreprocessedDocument)

    def test_original_text_preserved_unmodified(self):
        raw = "![img](x.webp) Some text <sup>1</sup>."
        doc = self.prep.process(raw)
        self.assertEqual(doc.original_text, raw)

    def test_source_path_stored(self):
        doc = self.prep.process("text", "my/path/file.md")
        self.assertEqual(doc.source_path, "my/path/file.md")

    def test_source_path_empty_default(self):
        doc = self.prep.process("text")
        self.assertEqual(doc.source_path, "")

    def test_cleaned_text_is_stripped(self):
        raw = "\n\n  Hello world.  \n\n"
        doc = self.prep.process(raw)
        self.assertEqual(doc.cleaned_text, doc.cleaned_text.strip())

    # ── Default config end-to-end ────────────────────────────────────────────

    def test_adjacent_tables_fixed(self):
        doc = self.prep.process(ADJACENT_TABLES)
        lower = doc.cleaned_text.lower()
        # No direct </table><table concatenation
        self.assertNotIn("</table><table", lower)

    def test_toc_table_removed(self):
        doc = self.prep.process(TOC_TABLE)
        self.assertNotIn("Introduction .....", doc.cleaned_text)

    def test_images_extracted_to_alts(self):
        raw = "Before ![Google logo](logo.webp) After"
        doc = self.prep.process(raw)
        self.assertEqual(len(doc.image_alts), 1)
        self.assertEqual(doc.image_alts[0].alt_text, "Google logo")
        self.assertIn("[IMAGE_DESC_PLACEHOLDER_0]", doc.cleaned_text)

    def test_superscripts_stripped(self):
        doc = self.prep.process(SUPERSCRIPT_TEXT)
        self.assertNotIn("<sup>", doc.cleaned_text)

    def test_decorative_lines_removed(self):
        raw = "Real content.\nContinues next page\nMore content."
        doc = self.prep.process(raw)
        self.assertNotIn("Continues next page", doc.cleaned_text)

    def test_excess_blank_lines_collapsed(self):
        doc = self.prep.process(MANY_BLANK_LINES)
        self.assertNotIn("\n\n\n", doc.cleaned_text)

    def test_content_preserved_after_all_cleaning(self):
        """Real table content must survive all cleaning steps."""
        doc = self.prep.process(ADJACENT_TABLES)
        self.assertIn("CMI Report", doc.cleaned_text)
        self.assertIn("Imprimir", doc.cleaned_text)

    # ── Config flags ─────────────────────────────────────────────────────────

    def test_flag_fix_adjacent_tables_false(self):
        cfg = PreprocessorConfig(fix_adjacent_tables=False)
        doc = MarkdownPreprocessor(cfg).process(ADJACENT_TABLES)
        import re
        lower = doc.cleaned_text.replace("\n", "")
        self.assertIn("</table><table", lower.lower())

    def test_flag_drop_toc_tables_false(self):
        cfg = PreprocessorConfig(drop_toc_tables=False)
        doc = MarkdownPreprocessor(cfg).process(TOC_TABLE)
        self.assertIn("Introduction .....", doc.cleaned_text)

    def test_flag_strip_superscripts_false(self):
        cfg = PreprocessorConfig(strip_superscripts=False)
        doc = MarkdownPreprocessor(cfg).process(SUPERSCRIPT_TEXT)
        self.assertIn("<sup>d</sup>", doc.cleaned_text)

    def test_flag_strip_decorative_false(self):
        cfg = PreprocessorConfig(strip_decorative_lines=False)
        raw = "Content.\nContinues next page\nMore."
        doc = MarkdownPreprocessor(cfg).process(raw)
        self.assertIn("Continues next page", doc.cleaned_text)

    def test_image_strategy_drop(self):
        cfg = PreprocessorConfig(image_alt_strategy="drop")
        raw = "Before ![Logo](logo.webp) After"
        doc = MarkdownPreprocessor(cfg).process(raw)
        self.assertEqual(doc.image_alts, [])
        self.assertNotIn("Logo", doc.cleaned_text)

    def test_image_strategy_keep(self):
        cfg = PreprocessorConfig(image_alt_strategy="keep")
        raw = "Before ![Google logo](logo.webp) After"
        doc = MarkdownPreprocessor(cfg).process(raw)
        self.assertEqual(doc.image_alts, [])
        self.assertIn("Google logo", doc.cleaned_text)
        self.assertNotIn("[IMAGE_DESC_PLACEHOLDER", doc.cleaned_text)

    # ── Real file smoke tests ────────────────────────────────────────────────

    def test_binder2_real_file(self):
        path = BINDER2_MD
        if not path.exists():
            self.skipTest("Binder2.md not available")
        raw = path.read_text(encoding="utf-8")
        doc = self.prep.process(raw, str(path))
        self.assertLess(len(doc.cleaned_text), len(raw))
        self.assertGreater(len(doc.image_alts), 0)
        self.assertNotIn("<sup>", doc.cleaned_text)
        self.assertNotIn("</table><table", doc.cleaned_text.lower())

    def test_testing_real_file_has_four_images(self):
        path = TESTING_MD
        if not path.exists():
            self.skipTest("Testing.md not available")
        raw = path.read_text(encoding="utf-8")
        doc = self.prep.process(raw, str(path))
        self.assertEqual(len(doc.image_alts), 4)
        self.assertNotIn("Introduction .....", doc.cleaned_text)
        self.assertNotIn("<sup>7</sup>", doc.cleaned_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
