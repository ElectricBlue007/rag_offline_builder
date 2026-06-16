from __future__ import annotations

import unittest

from core.contracts.chunking import ChunkArtifact, ChunkingResult
from core.contracts.content import ContentElement
from core.contracts.enrichment import EnrichmentResult, SummaryArtifact
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection


class TestChunkingContracts(unittest.TestCase):
    def test_chunk_artifact_validation(self) -> None:
        chunk = ChunkArtifact(chunk_id="c1", document_id="d1", text="Body")
        chunk.validate()
        self.assertEqual(chunk.debug()["chunk_id"], "c1")

    def test_chunking_result_validation(self) -> None:
        markdown = MarkdownGenerationResult(
            source_path="/tmp/sample.docx",
            markdown="# Title\n\nBody",
            sections=[MarkdownSection(section_id="s1", title="Title", level=0, markdown="# Title")],
            content_elements=[ContentElement(element_id="e1", element_type="paragraph", text="Body")],
        )
        enrichment = EnrichmentResult(
            source_path="/tmp/sample.docx",
            markdown=markdown,
            summaries=[SummaryArtifact(summary_id="s1", text="Title")],
        )
        result = ChunkingResult(
            source_path="/tmp/sample.docx",
            document_id="doc-1",
            chunks=[ChunkArtifact(chunk_id="c1", document_id="doc-1", text="Body")],
            markdown=markdown,
            enrichment=enrichment,
        )
        result.validate()


if __name__ == "__main__":
    unittest.main()

