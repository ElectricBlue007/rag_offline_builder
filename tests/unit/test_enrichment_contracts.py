from __future__ import annotations

import unittest

from core.contracts.content import ContentElement
from core.contracts.enrichment import EnrichmentResult, KeywordArtifact, ReferenceArtifact, SummaryArtifact
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection


class TestEnrichmentContracts(unittest.TestCase):
    def test_summary_validation(self) -> None:
        summary = SummaryArtifact(summary_id="s1", text="Short summary")
        summary.validate()
        self.assertEqual(summary.debug()["text"], "Short summary")

    def test_keyword_validation(self) -> None:
        keyword = KeywordArtifact(keyword_id="k1", keyword="RAG", score=0.9)
        keyword.validate()

    def test_reference_validation(self) -> None:
        reference = ReferenceArtifact(reference_id="r1", citation="Doe 2024")
        reference.validate()

    def test_enrichment_result_validation(self) -> None:
        result = EnrichmentResult(
            source_path="/tmp/sample.docx",
            markdown=MarkdownGenerationResult(
                source_path="/tmp/sample.docx",
                markdown="# Title",
                sections=[MarkdownSection(section_id="s1", title="Title", level=0, markdown="# Title")],
                content_elements=[ContentElement(element_id="e1", element_type="heading", text="Title", level=0)],
            ),
            summaries=[SummaryArtifact(summary_id="s1", text="Short summary")],
            keywords=[KeywordArtifact(keyword_id="k1", keyword="RAG")],
            references=[ReferenceArtifact(reference_id="r1", citation="Doe 2024")],
        )
        result.validate()


if __name__ == "__main__":
    unittest.main()

