from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.contracts.content import ContentElement
from core.contracts.enrichment import EnrichmentResult, SummaryArtifact
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection
from chunking.splitters.splitter import ChunkSplitter


class TestChunker(unittest.TestCase):
    def test_split_markdown_into_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            path.write_text("placeholder", encoding="utf-8")
            markdown = MarkdownGenerationResult(
                source_path=str(path),
                markdown="# Title\n\nFirst sentence. Second sentence.\n\n## Sub\nThird sentence.",
                sections=[
                    MarkdownSection(section_id="s1", title="Title", level=0, markdown="# Title"),
                    MarkdownSection(section_id="s2", title="Sub", level=1, markdown="## Sub"),
                ],
                content_elements=[ContentElement(element_id="e1", element_type="heading", text="Title", level=0)],
            )
            enrichment = EnrichmentResult(
                source_path=str(path),
                markdown=markdown,
                summaries=[SummaryArtifact(summary_id="s1", text="Title")],
            )

            result = ChunkSplitter().split(str(path), "doc-1", markdown, enrichment)

            self.assertTrue(result.chunks)
            self.assertEqual(result.document_id, "doc-1")

    def test_validate_and_health_check(self) -> None:
        splitter = ChunkSplitter()
        self.assertEqual(splitter.validate()["status"], "ok")
        self.assertEqual(splitter.health_check()["splitter"], "chunking")


if __name__ == "__main__":
    unittest.main()

