# filepath: core/contracts/tables.py

"""Table extraction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class TableCellSpan(ContractMixin):
    row_index: int
    column_index: int
    rowspan: int = 1
    colspan: int = 1

    def validate(self) -> None:
        _require(self.row_index >= 0, "row_index must be non-negative")
        _require(self.column_index >= 0, "column_index must be non-negative")
        _require(self.rowspan >= 1, "rowspan must be at least 1")
        _require(self.colspan >= 1, "colspan must be at least 1")


@dataclass(slots=True)
class TableCellData(ContractMixin):
    row_index: int
    column_index: int
    text: str
    span: TableCellSpan = field(init=False)
    is_header: bool = False

    def __post_init__(self) -> None:
        self.span = TableCellSpan(row_index=self.row_index, column_index=self.column_index)

    def validate(self) -> None:
        # ALLOW empty strings, just ensure it's not None
        _require(self.text is not None, "text cannot be None")
        self.span.validate()


@dataclass(slots=True)
class TableExtractionResult(ContractMixin):
    source_path: str
    tables: list[list[TableCellData]]
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        for table in self.tables:
            # Note: We allow empty lists if the document has bizarre empty table definitions,
            # but usually we want at least cell structures to exist.
            for cell in table:
                cell.validate()