"""
tests/test_table_normalizer.py

Unit tests for table_normalizer.py — covering both real table shapes
observed in Chandra OCR output (header-row and key-value), plus edge
cases: <br/> handling, empty markers, colspan, malformed HTML, and
decorative boilerplate leaking into cells.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.normalizers.table_normalizer import (
    normalize_table_html,
    parse_table_html,
    TableNormalizationError,
    TableCell,
    ParsedRow,
    ParsedTable,
    _clean_cell_text,
    _is_bold_only,
    _normalize_header_row_table,
    _normalize_key_value_table,
    _normalize_fallback_table,
)
from Chunker.tests.fixtures_path import BINDER2_MD, TESTING_MD


# ════════════════════════════════════════════════════════════════════
# Fixtures — real table HTML patterns from Binder2.md and Testing.md
# ════════════════════════════════════════════════════════════════════

HEADER_ROW_TABLE = """\
<table border="1">
<thead>
<tr>
<th>Operator action</th>
<th>Logical entry</th>
<th>Restrictions</th>
</tr>
</thead>
<tbody>
<tr>
<td>Create SOI</td>
<td>CMI track</td>
<td>Only for lost surface point tracks. Maximum number of CMI =10</td>
</tr>
<tr>
<td>Eliminate the CMI</td>
<td>CMI track</td>
<td>Only for manually initiated tracks</td>
</tr>
</tbody>
</table>
"""

# Binder2.md pattern: dash means "nothing to report"
DASH_EMPTY_TABLE = """\
<table border="1">
<thead><tr><th>System report</th><th>Restrictions</th></tr></thead>
<tbody>
<tr><td>CMI Report</td><td>-</td></tr>
<tr><td>CMI track</td><td>-</td></tr>
</tbody>
</table>
"""

# Binder2.md pattern: <br/> joining related departures
BR_IN_CELL_TABLE = """\
<table border="1">
<thead><tr><th>Action</th><th>Related departure</th></tr></thead>
<tbody>
<tr><td>Create SOI</td><td>CMI<br/>CMI Report</td></tr>
</tbody>
</table>
"""

# Testing.md pattern: key-value with single colspan value cells
KEY_VALUE_SINGLE_TABLE = """\
<table border="1">
<tr><td><b>Name</b></td><td colspan="3">1_1_movie_classification</td></tr>
<tr><td><b>Goal</b></td><td colspan="3">Classify movie reviews</td></tr>
<tr><td><b>Model</b></td><td colspan="3">gemini-pro</td></tr>
</table>
"""

# Testing.md pattern: key-value with TWO pairs per row (no colspan)
KEY_VALUE_DOUBLE_TABLE = """\
<table border="1">
<tr><td><b>Temperature</b></td><td>0.1</td><td><b>Token Limit</b></td><td>5</td></tr>
<tr><td><b>Top-K</b></td><td>N/A</td><td><b>Top-P</b></td><td>1</td></tr>
</table>
"""

# Decorative boilerplate leaking into a cell's own HTML
DECORATIVE_IN_CELL_TABLE = """\
<table border="1">
<tr><td><b>Prompt</b></td><td colspan="3">
<p>Some real content here.</p>
<p style="text-align: right;"><b>Continues next page...</b></p>
</td></tr>
</table>
"""

MALFORMED_NO_TABLE_TAG = "<div>Not a table at all</div>"
EMPTY_TABLE = "<table></table>"


class TestCleanCellText(unittest.TestCase):

    def _cell(self, html_fragment: str):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(f"<td>{html_fragment}</td>", "html.parser")
        return soup.find("td")

    def test_plain_text(self):
        self.assertEqual(_clean_cell_text(self._cell("Hello world")), "Hello world")

    def test_br_becomes_semicolon_separator(self):
        result = _clean_cell_text(self._cell("CMI<br/>CMI Report"))
        self.assertEqual(result, "CMI; CMI Report")

    def test_multiple_br_tags(self):
        result = _clean_cell_text(self._cell("A<br/>B<br/>C"))
        self.assertEqual(result, "A; B; C")

    def test_no_stray_space_before_semicolon(self):
        """A <br/> after a space must not produce 'A ; B'."""
        result = _clean_cell_text(self._cell("A <br/>B"))
        self.assertNotIn(" ;", result)

    def test_paragraph_tags_joined_with_space(self):
        result = _clean_cell_text(self._cell("<p>First.</p><p>Second.</p>"))
        self.assertIn("First.", result)
        self.assertIn("Second.", result)

    def test_whitespace_collapsed(self):
        result = _clean_cell_text(self._cell("A    B\n\n  C"))
        self.assertEqual(result, "A B C")

    def test_decorative_continues_next_page_stripped(self):
        result = _clean_cell_text(self._cell(
            "<p>Real content.</p><p><b>Continues next page...</b></p>"
        ))
        self.assertNotIn("Continues next page", result)
        self.assertIn("Real content.", result)

    def test_empty_cell(self):
        self.assertEqual(_clean_cell_text(self._cell("")), "")

    def test_dash_only_cell(self):
        self.assertEqual(_clean_cell_text(self._cell("-")), "-")


class TestIsBoldOnly(unittest.TestCase):

    def _cell(self, html_fragment: str):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(f"<td>{html_fragment}</td>", "html.parser")
        return soup.find("td")

    def test_bold_only_true(self):
        self.assertTrue(_is_bold_only(self._cell("<b>Temperature</b>")))

    def test_strong_only_true(self):
        self.assertTrue(_is_bold_only(self._cell("<strong>Model</strong>")))

    def test_bold_plus_extra_text_false(self):
        self.assertFalse(_is_bold_only(self._cell("<b>Bold</b> and more text")))

    def test_plain_text_false(self):
        self.assertFalse(_is_bold_only(self._cell("Plain value")))

    def test_no_bold_tag_false(self):
        self.assertFalse(_is_bold_only(self._cell("0.1")))

    def test_empty_cell_false(self):
        self.assertFalse(_is_bold_only(self._cell("")))

    def test_multiple_bold_tags_false(self):
        """A cell with two <b> tags isn't a clean single label."""
        self.assertFalse(_is_bold_only(self._cell("<b>A</b> and <b>B</b>")))


class TestParseTableHtml(unittest.TestCase):

    def test_header_row_detected(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        self.assertEqual(parsed.table_type, "header_row")

    def test_header_row_headers_extracted(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        self.assertEqual(parsed.headers, ["Operator action", "Logical entry", "Restrictions"])

    def test_header_row_data_rows_count(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        self.assertEqual(len(parsed.rows), 2)

    def test_key_value_detected(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        self.assertEqual(parsed.table_type, "key_value")

    def test_key_value_no_headers(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        self.assertEqual(parsed.headers, [])

    def test_key_value_row_count(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        self.assertEqual(len(parsed.rows), 3)

    def test_key_value_bold_only_flag_set(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        first_cell = parsed.rows[0].cells[0]
        self.assertTrue(first_cell.is_bold_only)
        self.assertEqual(first_cell.text, "Name")

    def test_colspan_captured(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        value_cell = parsed.rows[0].cells[1]
        self.assertEqual(value_cell.colspan, 3)

    def test_malformed_html_raises(self):
        with self.assertRaises(TableNormalizationError):
            parse_table_html(MALFORMED_NO_TABLE_TAG)

    def test_empty_table_raises(self):
        with self.assertRaises(TableNormalizationError):
            parse_table_html(EMPTY_TABLE)

    def test_fallback_type_for_no_th_no_bold_label(self):
        plain = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        parsed = parse_table_html(plain)
        self.assertEqual(parsed.table_type, "fallback")


class TestNormalizeHeaderRowTable(unittest.TestCase):

    def test_subject_from_first_column(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertIn("Create SOI —", result)

    def test_field_names_lowercased(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertIn("logical entry:", result)
        self.assertIn("restrictions:", result)

    def test_each_row_becomes_own_line(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        result = _normalize_header_row_table(parsed)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)

    def test_dash_values_dropped(self):
        parsed = parse_table_html(DASH_EMPTY_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertNotIn("restrictions: -", result)
        self.assertNotIn(": -", result)

    def test_dash_only_row_reduces_to_subject_sentence(self):
        parsed = parse_table_html(DASH_EMPTY_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertIn("CMI Report.", result)

    def test_br_in_cell_becomes_semicolon_in_output(self):
        parsed = parse_table_html(BR_IN_CELL_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertIn("CMI; CMI Report", result)

    def test_no_double_semicolons(self):
        parsed = parse_table_html(BR_IN_CELL_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertNotIn(";;", result)

    def test_all_content_preserved(self):
        parsed = parse_table_html(HEADER_ROW_TABLE)
        result = _normalize_header_row_table(parsed)
        self.assertIn("Create SOI", result)
        self.assertIn("Eliminate the CMI", result)
        self.assertIn("CMI track", result)


class TestNormalizeKeyValueTable(unittest.TestCase):

    def test_single_pair_per_row_colspan(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertIn("Name: 1_1_movie_classification", result)
        self.assertIn("Goal: Classify movie reviews", result)
        self.assertIn("Model: gemini-pro", result)

    def test_two_pairs_per_row(self):
        parsed = parse_table_html(KEY_VALUE_DOUBLE_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertIn("Temperature: 0.1", result)
        self.assertIn("Token Limit: 5", result)
        self.assertIn("Top-K: N/A", result)
        self.assertIn("Top-P: 1", result)

    def test_na_value_preserved_not_dropped(self):
        """N/A is a meaningful config value, unlike a bare '-'."""
        parsed = parse_table_html(KEY_VALUE_DOUBLE_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertIn("N/A", result)

    def test_whole_table_is_one_flat_string(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertNotIn("\n", result)

    def test_pairs_joined_with_semicolons(self):
        parsed = parse_table_html(KEY_VALUE_SINGLE_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertIn(";", result)

    def test_decorative_text_stripped_from_key_value_output(self):
        parsed = parse_table_html(DECORATIVE_IN_CELL_TABLE)
        result = _normalize_key_value_table(parsed)
        self.assertNotIn("Continues next page", result)
        self.assertIn("Some real content here.", result)


class TestNormalizeFallbackTable(unittest.TestCase):

    def test_fallback_preserves_all_cell_text(self):
        plain = "<table><tr><td>A</td><td>B</td></tr></table>"
        parsed = parse_table_html(plain)
        result = _normalize_fallback_table(parsed)
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_fallback_row_per_line(self):
        plain = "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
        parsed = parse_table_html(plain)
        result = _normalize_fallback_table(parsed)
        self.assertEqual(len(result.split("\n")), 2)


class TestNormalizeTableHtmlEndToEnd(unittest.TestCase):
    """Tests for the main public entry point."""

    def test_returns_string(self):
        result = normalize_table_html(HEADER_ROW_TABLE)
        self.assertIsInstance(result, str)

    def test_header_row_routes_correctly(self):
        result = normalize_table_html(HEADER_ROW_TABLE)
        self.assertIn("Create SOI —", result)

    def test_key_value_routes_correctly(self):
        result = normalize_table_html(KEY_VALUE_SINGLE_TABLE)
        self.assertIn("Name: 1_1_movie_classification", result)

    def test_fallback_routes_correctly(self):
        plain = "<table><tr><td>X</td><td>Y</td></tr></table>"
        result = normalize_table_html(plain)
        self.assertIn("X", result)
        self.assertIn("Y", result)

    def test_malformed_html_raises_normalization_error(self):
        with self.assertRaises(TableNormalizationError):
            normalize_table_html(MALFORMED_NO_TABLE_TAG)

    def test_no_html_tags_in_output(self):
        result = normalize_table_html(HEADER_ROW_TABLE)
        self.assertNotIn("<td>", result)
        self.assertNotIn("<tr>", result)
        self.assertNotIn("<table", result)

    def test_original_html_not_mutated(self):
        """normalize_table_html must not alter the caller's HTML string."""
        original = HEADER_ROW_TABLE
        _ = normalize_table_html(original)
        self.assertEqual(original, HEADER_ROW_TABLE)


class TestRealFileTables(unittest.TestCase):
    """Integration tests against the real fixture files."""

    def _get_tables(self, md_path):
        from Chunker.preprocessors.md_preprocessor import MarkdownPreprocessor
        from Chunker.splitters.structure_splitter import StructureSplitter

        raw = md_path.read_text(encoding="utf-8")
        doc = MarkdownPreprocessor().process(raw)
        sections = StructureSplitter().split(doc.cleaned_text)
        return [s for s in sections if s.section_type == "table"]

    def test_binder2_all_tables_normalize_without_error(self):
        tables = self._get_tables(BINDER2_MD)
        self.assertGreater(len(tables), 0)
        for t in tables:
            result = normalize_table_html(t.content)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_binder2_tables_detected_as_header_row(self):
        tables = self._get_tables(BINDER2_MD)
        for t in tables:
            parsed = parse_table_html(t.content)
            self.assertEqual(parsed.table_type, "header_row")

    def test_binder2_no_html_leaks_into_normalized_text(self):
        tables = self._get_tables(BINDER2_MD)
        for t in tables:
            result = normalize_table_html(t.content)
            self.assertNotIn("<td", result)
            self.assertNotIn("<tr", result)

    def test_testing_all_tables_normalize_without_error(self):
        tables = self._get_tables(TESTING_MD)
        self.assertGreater(len(tables), 0)
        for t in tables:
            result = normalize_table_html(t.content)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_testing_tables_detected_as_key_value(self):
        tables = self._get_tables(TESTING_MD)
        for t in tables:
            parsed = parse_table_html(t.content)
            self.assertEqual(parsed.table_type, "key_value")

    def test_testing_no_continues_next_page_leak(self):
        tables = self._get_tables(TESTING_MD)
        for t in tables:
            result = normalize_table_html(t.content)
            self.assertNotIn("Continues next page", result)

    def test_testing_prompt_content_preserved(self):
        tables = self._get_tables(TESTING_MD)
        all_normalized = " ".join(normalize_table_html(t.content) for t in tables)
        self.assertIn("gemini-pro", all_normalized)
        self.assertIn("Temperature", all_normalized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
