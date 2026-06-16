"""Local enrichment over markdown outputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.contracts.enrichment import EnrichmentResult, KeywordArtifact, ReferenceArtifact, SummaryArtifact
from core.contracts.markdown import MarkdownGenerationResult


class EnrichmentProcessor:
    processor_name = "enrichment"

    def enrich(self, source_path: str, markdown: MarkdownGenerationResult) -> EnrichmentResult:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        markdown.validate()
        summaries = self._build_summaries(markdown)
        keywords = self._build_keywords(markdown.markdown)
        references = self._build_references(markdown.markdown)
        result = EnrichmentResult(
            source_path=str(path),
            markdown=markdown,
            summaries=summaries,
            keywords=keywords,
            references=references,
            debug_info={"processor": self.processor_name},
        )
        result.validate()
        return result

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "processor": self.processor_name}

    def debug(self) -> dict[str, Any]:
        return {"processor": self.processor_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "processor": self.processor_name}

    def _build_summaries(self, markdown: MarkdownGenerationResult) -> list[SummaryArtifact]:
        summaries: list[SummaryArtifact] = []
        for index, section in enumerate(markdown.sections):
            summaries.append(
                SummaryArtifact(
                    summary_id=f"summary-{index}",
                    text=section.title,
                    section_id=section.section_id,
                    metadata={"source": "heading_title"},
                )
            )
        if not summaries:
            summaries.append(SummaryArtifact(summary_id="summary-0", text=markdown.markdown[:200] or "Document"))
        return summaries

    def _build_keywords(self, markdown_text: str) -> list[KeywordArtifact]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", markdown_text.lower())
        counts = Counter(word for word in words if word not in {"the", "and", "for", "with", "from", "this", "that"})
        keywords: list[KeywordArtifact] = []
        for index, (word, count) in enumerate(counts.most_common(10)):
            keywords.append(KeywordArtifact(keyword_id=f"keyword-{index}", keyword=word, score=float(count)))
        return keywords

    def _build_references(self, markdown_text: str) -> list[ReferenceArtifact]:
        references: list[ReferenceArtifact] = []
        for index, line in enumerate(markdown_text.splitlines()):
            if "[" in line and "]" in line:
                references.append(
                    ReferenceArtifact(
                        reference_id=f"ref-{index}",
                        citation=line.strip(),
                        metadata={"source": "inline_markdown"},
                    )
                )
        return references


def enrich_markdown(source_path: str, markdown: MarkdownGenerationResult) -> EnrichmentResult:
    return EnrichmentProcessor().enrich(source_path, markdown)


def result_to_json(result: EnrichmentResult) -> str:
    return json.dumps(
        {
            "source_path": result.source_path,
            "summaries": [summary.debug() for summary in result.summaries],
            "keywords": [keyword.debug() for keyword in result.keywords],
            "references": [reference.debug() for reference in result.references],
            "debug_info": result.debug_info,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )

