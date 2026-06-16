"""Unit tests for markdown generation contracts."""
import unittest
from core.contracts.markdown import MarkdownSection, MarkdownGenerationResult
from core.exceptions.base import ContractValidationError

class TestMarkdownSectionContract(unittest.TestCase):
    def test_valid_section(self):
        section = MarkdownSection(
            section_id="sec-1",
            title="Introduction",
            level=1,
            markdown="# Introduction\n\nThis is the intro."
        )
        section.validate()  # Should not raise

    def test_invalid_section_missing_id(self):
        with self.assertRaises(ContractValidationError):
            MarkdownSection(section_id=" ", title="Intro", level=1, markdown="# Intro").validate()

    def test_invalid_section_negative_level(self):
        with self.assertRaises(ContractValidationError):
            MarkdownSection(section_id="sec-1", title="Intro", level=-1, markdown="# Intro").validate()

class TestMarkdownGenerationResultContract(unittest.TestCase):
    def test_valid_result_with_rag_fields(self):
        section = MarkdownSection(section_id="sec-1", title="Intro", level=1, markdown="# Intro")
        result = MarkdownGenerationResult(
            source_path="/tmp/test.docx",
            markdown="# Intro",
            sections=[section],
            frontmatter={"title": "Test Document", "author": "AI"},
            footnotes={"1": "This is a footnote."}
        )
        result.validate()  # Should not raise

    def test_invalid_result_missing_markdown(self):
        section = MarkdownSection(section_id="sec-1", title="Intro", level=1, markdown="# Intro")
        with self.assertRaises(ContractValidationError):
            MarkdownGenerationResult(
                source_path="/tmp/test.docx",
                markdown="   ",  # Invalid: whitespace only
                sections=[section]
            ).validate()