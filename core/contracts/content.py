"""Content extraction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class TextSpan(ContractMixin):
    text: str
    style_name: str | None = None
    hyperlink: str | None = None

    def validate(self) -> None:
        _require(bool(self.text.strip()), "text is required")


@dataclass(slots=True)
class ContentElement(ContractMixin):
    element_id: str
    element_type: Literal["heading", "paragraph", "list_item", "quote", "code", "table", "image", "hyperlink"]
    text: str
    level: int | None = None
    page_number: int | None = None
    spans: list[TextSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.element_id.strip()), "element_id is required")
        _require(bool(self.text.strip()), "text is required")
        if self.level is not None:
            _require(self.level >= 0, "level must be non-negative")
        if self.page_number is not None:
            _require(self.page_number >= 0, "page_number must be non-negative")
        for span in self.spans:
            span.validate()


@dataclass(slots=True)
class ContentExtractionResult(ContractMixin):
    source_path: str
    document_type: str
    elements: list[ContentElement]
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.document_type.strip()), "document_type is required")
        _require(bool(self.elements), "elements are required")
        for element in self.elements:
            element.validate()

