from __future__ import annotations

from datetime import datetime
import unittest

from core.contracts.document import (
    ChunkRecord,
    ContentBlock,
    DocumentMetadata,
    DocumentSection,
    MarkdownDocument,
    TableArtifact,
    TableCell,
)
from core.exceptions.base import ContractValidationError


class TestCoreContracts(unittest.TestCase):
    def test_document_metadata_validation_and_debug(self) -> None:
        metadata = DocumentMetadata(
            file_name="sample.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            source_hash="abc123",
            title="Sample",
            created_at=datetime(2026, 1, 1),
        )

        metadata.validate()
        payload = metadata.debug()

        self.assertEqual(payload["file_name"], "sample.docx")
        self.assertEqual(payload["title"], "Sample")
        self.assertEqual(metadata.health_check()["status"], "ok")

    def test_document_metadata_rejects_missing_required_fields(self) -> None:
        metadata = DocumentMetadata(file_name=" ", mime_type="", source_hash=" ")

        with self.assertRaises(ContractValidationError):
            metadata.validate()

    def test_table_artifact_validation_requires_cells(self) -> None:
        table = TableArtifact(table_id="t1", cells=[TableCell(row_index=0, column_index=0, text="A")])

        table.validate()
        self.assertEqual(table.debug()["table_id"], "t1")

    def test_table_artifact_rejects_invalid_cell(self) -> None:
        table = TableArtifact(table_id="t1", cells=[TableCell(row_index=-1, column_index=0, text="A")])

        with self.assertRaises(ContractValidationError):
            table.validate()

    def test_markdown_document_validation(self) -> None:
        document = MarkdownDocument(
            document_id="doc-1",
            markdown="# Title\n\nBody",
            section_map=[DocumentSection(section_id="s1", title="Title", level=0)],
            source_metadata=DocumentMetadata(file_name="sample.pdf", mime_type="application/pdf", source_hash="hash"),
        )

        document.validate()

    def test_chunk_record_validation(self) -> None:
        record = ChunkRecord(chunk_id="c1", document_id="doc1", text="content")
        record.validate()
        self.assertEqual(record.debug()["chunk_id"], "c1")

    def test_content_block_validation(self) -> None:
        block = ContentBlock(block_id="b1", block_type="paragraph", text="hello", page_number=1)
        block.validate()
        self.assertEqual(block.debug()["block_type"], "paragraph")


if __name__ == "__main__":
    unittest.main()
