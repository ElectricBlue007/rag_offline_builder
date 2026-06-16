"""Markdown generation contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)

@dataclass(slots=True)
class MarkdownSection(ContractMixin):
    """Represents a distinct section of the generated Markdown."""
    section_id: str
    title: str
    level: int
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.section_id.strip()), "section_id is required")
        _require(bool(self.title.strip()), "title is required")
        _require(self.level >= 0, "level must be non-negative")
        _require(bool(self.markdown.strip()), "markdown is required")

@dataclass(slots=True)
class MarkdownGenerationResult(ContractMixin):
    """The final output of the RAG-optimized Markdown generator."""
    source_path: str
    markdown: str
    sections: list[MarkdownSection]
    
    # RAG Optimization Fields
    frontmatter: dict[str, Any] = field(default_factory=dict)
    footnotes: dict[str, str] = field(default_factory=dict)
    
    # Legacy fields (kept for backward compatibility)
    content_elements: list[Any] = field(default_factory=list)
    reading_order: Any | None = None
    tables: Any | None = None
    stream: Any | None = None
    
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.markdown.strip()), "markdown is required")
        for section in self.sections:
            section.validate()