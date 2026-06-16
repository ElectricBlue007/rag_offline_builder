"""Enrichment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.markdown import MarkdownGenerationResult
from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class SummaryArtifact(ContractMixin):
    summary_id: str
    text: str
    section_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.summary_id.strip()), "summary_id is required")
        _require(bool(self.text.strip()), "text is required")


@dataclass(slots=True)
class KeywordArtifact(ContractMixin):
    keyword_id: str
    keyword: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.keyword_id.strip()), "keyword_id is required")
        _require(bool(self.keyword.strip()), "keyword is required")
        if self.score is not None:
            _require(self.score >= 0.0, "score must be non-negative")


@dataclass(slots=True)
class ReferenceArtifact(ContractMixin):
    reference_id: str
    citation: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.reference_id.strip()), "reference_id is required")
        _require(bool(self.citation.strip()), "citation is required")


@dataclass(slots=True)
class EnrichmentResult(ContractMixin):
    source_path: str
    markdown: MarkdownGenerationResult
    summaries: list[SummaryArtifact] = field(default_factory=list)
    keywords: list[KeywordArtifact] = field(default_factory=list)
    references: list[ReferenceArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        self.markdown.validate()
        for summary in self.summaries:
            summary.validate()
        for keyword in self.keywords:
            keyword.validate()
        for reference in self.references:
            reference.validate()

