import pytest

from core.markdown_normalizer import MarkdownNormalizer
from rag_doc_compiler.errors import NormalizationError


def test_preserves_markdown_without_structural_repairs():
    markdown = "# Title\n\nBody text.\n"
    result = MarkdownNormalizer().normalize(markdown)
    assert result.markdown == markdown
    assert result.flags_by_line == {}


def test_repairs_broken_atx_header_with_trace_flag():
    result = MarkdownNormalizer().normalize("#Title\n")
    assert result.markdown == "# Title\n"
    assert result.flags_by_line == {1: ["atx_heading_space_inserted"]}


def test_splits_ocr_glued_heading_without_rewriting_text():
    result = MarkdownNormalizer().normalize("Image caption text.## Acknowledgements\n")
    assert result.markdown == "Image caption text.\n## Acknowledgements\n"
    assert result.flags_by_line == {2: ["embedded_atx_heading_split"]}


def test_splits_ocr_glued_html_table_without_rewriting_text():
    result = MarkdownNormalizer().normalize("Intro text.<table><tr><td>A</td></tr></table>\n")
    assert result.markdown == "Intro text.\n<table><tr><td>A</td></tr></table>\n"
    assert result.flags_by_line == {2: ["embedded_html_table_split"]}


def test_fails_on_unclosed_code_fence():
    with pytest.raises(NormalizationError):
        MarkdownNormalizer().normalize("```python\nprint('x')\n")


def test_repairs_common_utf8_mojibake():
    broken = "AI \u00e2\u20ac\u201c prompt\u0637\u00a2\u0622\u00a0engineering\n"
    result = MarkdownNormalizer().normalize(broken)
    assert result.markdown == "AI - prompt engineering\n"


def test_removes_embedded_table_of_contents_and_continuation_noise():
    markdown = (
        "# Prompt Engineering\n\n"
        "![Abstract geometric shape](shape.webp)Decorative shape.# Table of contents\n\n"
        "<table><tr><td>Introduction .....</td><td>6</td></tr></table>\n\n"
        "## Introduction\n"
        "<p style=\"text-align: right;\"><b>Continues next page...</b></p>\n"
        "Prompt engineering is useful.\n"
    )

    result = MarkdownNormalizer().normalize(markdown)

    assert "# Table of contents" not in result.markdown
    assert "Introduction ....." not in result.markdown
    assert "Continues next page" not in result.markdown
    assert "## Introduction" in result.markdown
