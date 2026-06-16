from __future__ import annotations

import unittest

from core.contracts.tables import TableCellData, TableCellSpan, TableExtractionResult


class TestTableContracts(unittest.TestCase):
    def test_table_cell_span_validation(self) -> None:
        span = TableCellSpan(row_index=0, column_index=1, rowspan=2, colspan=3)
        span.validate()
        self.assertEqual(span.debug()["colspan"], 3)

    def test_table_cell_data_validation(self) -> None:
        cell = TableCellData(row_index=0, column_index=0, text="A")
        cell.validate()
        self.assertEqual(cell.debug()["text"], "A")

    def test_table_extraction_result_validation(self) -> None:
        result = TableExtractionResult(
            source_path="/tmp/sample.docx",
            tables=[[TableCellData(row_index=0, column_index=0, text="A")]],
        )
        result.validate()

    def test_table_extraction_allows_empty_tables(self) -> None:
        result = TableExtractionResult(source_path="/tmp/sample.docx", tables=[])
        result.validate()


if __name__ == "__main__":
    unittest.main()
