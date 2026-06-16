from __future__ import annotations

import unittest

from core.contracts.document import (
    ChunkRecord,
    ContentBlock,
    DocumentMetadata,
    DocumentSection,
    MarkdownDocument,
)


class TestCoreContractsIntegration(unittest.TestCase):
    def test_contracts_round_trip_through_pipeline_boundary(self) -> None:
        metadata = DocumentMetadata(file_name="book.epub", mime_type="application/epub+zip", source_hash="hash-1")
        block = ContentBlock(block_id="b1", block_type="paragraph", text="Body text", section_id="s1")
        section = DocumentSection(section_id="s1", title="Chapter 1", level=0)
        markdown = MarkdownDocument(
            document_id="doc-1",
            markdown="# Chapter 1\n\nBody text",
            section_map=[section],
            source_metadata=metadata,
        )
        chunk = ChunkRecord(chunk_id="chunk-1", document_id="doc-1", text="Body text", section_path=["Chapter 1"])

        metadata.validate()
        block.validate()
        markdown.validate()
        chunk.validate()

        self.assertEqual(markdown.source_metadata.file_name, "book.epub")
        self.assertEqual(chunk.section_path, ["Chapter 1"])


if __name__ == "__main__":
    unittest.main()
