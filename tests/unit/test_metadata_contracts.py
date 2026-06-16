from __future__ import annotations

import unittest

from core.contracts.document import DocumentMetadata
from core.contracts.extraction import MetadataExtractionResult, SourceDocumentRequest
from core.exceptions.base import ContractValidationError


class TestMetadataExtractionContracts(unittest.TestCase):
    def test_request_validation(self) -> None:
        request = SourceDocumentRequest(file_path="/tmp/sample.docx", source_type="docx")
        request.validate()
        self.assertEqual(request.debug()["file_path"], "/tmp/sample.docx")

    def test_request_rejects_empty_path(self) -> None:
        request = SourceDocumentRequest(file_path=" ")
        with self.assertRaises(ContractValidationError):
            request.validate()

    def test_result_validation(self) -> None:
        result = MetadataExtractionResult(
            request=SourceDocumentRequest(file_path="/tmp/sample.docx"),
            metadata=DocumentMetadata(
                file_name="sample.docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                source_hash="hash-1",
            ),
        )
        result.validate()
        self.assertEqual(result.debug()["extractor_name"], "metadata")


if __name__ == "__main__":
    unittest.main()

