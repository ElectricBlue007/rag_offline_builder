# filepath: extractors/tables/extractor.py

"""DOCX table extraction with basic merged-cell awareness."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from core.contracts.tables import TableCellData, TableExtractionResult

XML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


class TableExtractor:
    extractor_name = "tables"

    def extract(self, file_path: str) -> TableExtractionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        if path.suffix.lower() != ".docx":
            raise ValueError("table extractor currently supports DOCX only")

        tables = self._extract_docx_tables(path)
        result = TableExtractionResult(
            source_path=str(path),
            tables=tables,
            debug_info={"extractor": self.extractor_name},
        )
        result.validate()
        return result

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def debug(self) -> dict[str, Any]:
        return {"extractor": self.extractor_name, "supported_types": ["docx"]}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def _extract_docx_tables(self, path: Path) -> list[list[TableCellData]]:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        tables = []
        for table_node in root.findall(".//w:tbl", XML_NS):
            tables.append(self._extract_single_table(table_node))
        return tables

    def _extract_single_table(self, table_node: ET.Element) -> list[TableCellData]:
        cells: list[TableCellData] = []
        for row_index, row in enumerate(table_node.findall("w:tr", XML_NS)):
            column_index = 0
            for cell in row.findall("w:tc", XML_NS):
                text = self._cell_text(cell)
                grid_span = self._grid_span(cell)
                cells.append(TableCellData(row_index=row_index, column_index=column_index, text=text))
                column_index += grid_span
        return cells

    def _cell_text(self, cell: ET.Element) -> str:
        paragraphs = []
        # Respect paragraph boundaries (<w:p>) inside cells
        for p in cell.findall(".//w:p", XML_NS):
            texts = [node.text for node in p.findall(".//w:t", XML_NS) if node.text]
            # Use "" to join w:t tags so we don't break formatted words apart
            p_text = "".join(texts).strip()
            if p_text:
                paragraphs.append(p_text)
                
        # Fallback for malformed cells without <w:p>
        if not paragraphs:
            texts = [node.text for node in cell.findall(".//w:t", XML_NS) if node.text]
            p_text = "".join(texts).strip()
            if p_text:
                return p_text
                
        # Preserve paragraph breaks as newlines
        return "\n".join(paragraphs)

    def _grid_span(self, cell: ET.Element) -> int:
        span = cell.find("w:tcPr/w:gridSpan", XML_NS)
        if span is None:
            return 1
        value = span.attrib.get(f"{{{XML_NS['w']}}}val", "1")
        try:
            return max(1, int(value))
        except ValueError:
            return 1


def extract_tables(file_path: str) -> TableExtractionResult:
    return TableExtractor().extract(file_path)


def result_to_json(result: TableExtractionResult) -> str:
    return json.dumps(result.debug_info, indent=2, sort_keys=True, default=str)