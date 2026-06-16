"""Table formatting utilities for strict Markdown table generation."""
from typing import Any
from core.contracts.tables import TableCellData
from core.logging import configure_logging, log_event
from .text_sanitizer import TextSanitizer

class TableFormatter:
    """Formats table cells into strict, pipe-delimited Markdown tables."""
    
    def __init__(self):
        self.logger = configure_logging()
        self.sanitizer = TextSanitizer()

    def format_table(self, table_cells: list[TableCellData]) -> list[str]:
        """Convert a list of TableCellData into Markdown table rows."""
        log_event(self.logger, 10, "format_table_start", "formatting table", cell_count=len(table_cells))
        
        if not table_cells:
            return []

        # Group cells by row_index
        rows: dict[int, list[TableCellData]] = {}
        max_cols = 0
        
        for cell in table_cells:
            row_idx = getattr(cell, "row_index", 0)
            col_idx = getattr(cell, "column_index", 0)
            rows.setdefault(row_idx, []).append(cell)
            max_cols = max(max_cols, col_idx + 1)

        if not rows:
            return []

        # Sort rows and columns
        ordered_rows = [rows[idx] for idx in sorted(rows.keys())]
        normalized_rows = [
            sorted(row, key=lambda c: getattr(c, "column_index", 0)) for row in ordered_rows
        ]

        # Ensure all rows have the same number of columns (pad with empty cells)
        padded_rows = []
        for row in normalized_rows:
            padded_row = []
            col_map = {getattr(c, "column_index", 0): c for c in row}
            for col_idx in range(max_cols):
                # Fallback to empty text if a cell is missing in the grid
                padded_row.append(col_map.get(col_idx, TableCellData(row_index=0, column_index=col_idx, text="")))
            padded_rows.append(padded_row)

        # Build Markdown
        markdown_rows = []
        
        # Header row (Row 0)
        header_cells = [self._escape_markdown(self.sanitizer.sanitize(getattr(c, "text", ""))) for c in padded_rows[0]]
        markdown_rows.append("| " + " | ".join(header_cells) + " |")
        
        # Separator row
        markdown_rows.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Data rows (Row 1+)
        for row in padded_rows[1:]:
            data_cells = [self._escape_markdown(self.sanitizer.sanitize(getattr(c, "text", ""))) for c in row]
            markdown_rows.append("| " + " | ".join(data_cells) + " |")

        log_event(self.logger, 10, "format_table_end", "table formatted", row_count=len(padded_rows))
        return markdown_rows

    def _escape_markdown(self, text: str) -> str:
        """Escape pipe characters to prevent breaking Markdown table structure."""
        return text.replace("|", "\\|")

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "module": "TableFormatter"}

    def debug(self) -> dict[str, Any]:
        return {"module": "TableFormatter"}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "module": "TableFormatter"}