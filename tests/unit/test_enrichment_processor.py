from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.contracts.content import ContentElement
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection
from enrichment.processor import EnrichmentProcessor


class TestEnrichmentProcessor(unittest.TestCase):
    def test_enrich_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            path.write_text("placeholder", encoding="utf-8")
            markdown = MarkdownGenerationResult(
                source_path=str(path),
                markdown="# Title\n\nBody text with [1] reference.",
                sections=[MarkdownSection(section_id="s1", title="Title", level=0, markdown="# Title")],
                content_elements=[ContentElement(element_id="e1", element_type="heading", text="Title", level=0)],
            )

            result = EnrichmentProcessor().enrich(str(path), markdown)

            self.assertTrue(result.summaries)
            self.assertTrue(result.keywords)
            self.assertTrue(result.references)

    def test_validate_and_health_check(self) -> None:
        processor = EnrichmentProcessor()
        self.assertEqual(processor.validate()["status"], "ok")
        self.assertEqual(processor.health_check()["processor"], "enrichment")


if __name__ == "__main__":
    unittest.main()

