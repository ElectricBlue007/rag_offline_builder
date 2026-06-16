from __future__ import annotations

import unittest

from core.contracts.content import ContentElement, ContentExtractionResult, TextSpan
from core.exceptions.base import ContractValidationError


class TestContentContracts(unittest.TestCase):
    def test_text_span_validation(self) -> None:
        span = TextSpan(text="hello")
        span.validate()
        self.assertEqual(span.debug()["text"], "hello")

    def test_content_element_validation(self) -> None:
        element = ContentElement(
            element_id="e1",
            element_type="paragraph",
            text="Body",
            spans=[TextSpan(text="Body")],
        )
        element.validate()
        self.assertEqual(element.debug()["element_type"], "paragraph")

    def test_content_extraction_result_validation(self) -> None:
        result = ContentExtractionResult(
            source_path="/tmp/sample.txt",
            document_type="txt",
            elements=[ContentElement(element_id="e1", element_type="paragraph", text="Body")],
        )
        result.validate()

    def test_content_extraction_rejects_empty_elements(self) -> None:
        result = ContentExtractionResult(source_path="/tmp/sample.txt", document_type="txt", elements=[])
        with self.assertRaises(ContractValidationError):
            result.validate()


if __name__ == "__main__":
    unittest.main()

