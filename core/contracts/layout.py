"""Layout analysis contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class LayoutNode(ContractMixin):
    node_id: str
    node_type: Literal["paragraph", "heading", "table", "textbox", "sidebar", "column_break", "image"]
    text: str | None = None
    page_number: int | None = None
    order_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.node_id.strip()), "node_id is required")
        _require(bool(self.node_type), "node_type is required")
        _require(self.order_index >= 0, "order_index must be non-negative")
        if self.page_number is not None:
            _require(self.page_number >= 0, "page_number must be non-negative")


@dataclass(slots=True)
class ReadingOrderResult(ContractMixin):
    source_path: str
    nodes: list[LayoutNode]
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.nodes), "nodes are required")
        for node in self.nodes:
            node.validate()

