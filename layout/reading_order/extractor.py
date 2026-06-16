"""DOCX reading order reconstruction with robust heading detection."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from core.contracts.layout import LayoutNode, ReadingOrderResult
from core.utils.style_classifier import StyleClassifier

XML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


class ReadingOrderExtractor:
    extractor_name = "reading_order"

    def extract(self, file_path: str) -> ReadingOrderResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        if path.suffix.lower() != ".docx":
            raise ValueError("reading order extractor currently supports DOCX only")

        nodes = self._extract_docx_reading_order(path)
        result = ReadingOrderResult(source_path=str(path), nodes=nodes, debug_info={"extractor": self.extractor_name})
        result.validate()
        return result

    def _extract_docx_reading_order(self, path: Path) -> list[LayoutNode]:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find("w:body", XML_NS)
        nodes: list[LayoutNode] = []
        if body is None:
            return nodes
            
        order_index = 0
        image_index = 0
        
        for child in list(body):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                text = self._paragraph_text(child)
                if text:
                    style_name = self._paragraph_style(child)
                    is_heading, level = StyleClassifier.classify(style_name, text)
                    node_type = "heading" if is_heading else "paragraph"
                    
                    nodes.append(
                        LayoutNode(
                            node_id=f"{path.stem}-{order_index}",
                            node_type=node_type,
                            text=text,
                            order_index=order_index,
                            metadata={"style_name": style_name, "heading_level": level},
                        )
                    )
                    order_index += 1
                
                # Detect inline images
                drawings = child.findall(".//w:drawing", XML_NS)
                for _ in drawings:
                    nodes.append(
                        LayoutNode(
                            node_id=f"{path.stem}-{order_index}",
                            node_type="image",
                            text=None,
                            order_index=order_index,
                            metadata={"image_index": image_index}
                        )
                    )
                    order_index += 1
                    image_index += 1
                    
            elif tag == "tbl":
                nodes.append(
                    LayoutNode(
                        node_id=f"{path.stem}-{order_index}",
                        node_type="table",
                        text=self._table_text(child),
                        order_index=order_index,
                    )
                )
                order_index += 1
        return nodes

    def _paragraph_text(self, paragraph: ET.Element) -> str:
        texts = [node.text for node in paragraph.findall(".//w:t", XML_NS) if node.text]
        return " ".join(texts).strip()

    def _paragraph_style(self, paragraph: ET.Element) -> str:
        style = paragraph.find("w:pPr/w:pStyle", XML_NS)
        if style is None:
            return "Normal"
        return style.attrib.get(f"{{{XML_NS['w']}}}val", "Normal")

    def _table_text(self, table: ET.Element) -> str:
        cells = []
        for cell in table.findall(".//w:tc", XML_NS):
            texts = [node.text for node in cell.findall(".//w:t", XML_NS) if node.text]
            text = " ".join(texts).strip()
            if text:
                cells.append(text)
        return " | ".join(cells)

    def validate(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}
    def debug(self) -> dict[str, Any]: return {"extractor": self.extractor_name}
    def health_check(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}

def extract_reading_order(file_path: str) -> ReadingOrderResult:
    return ReadingOrderExtractor().extract(file_path)

def result_to_json(result: ReadingOrderResult) -> str:
    return json.dumps(result.debug_info, indent=2, sort_keys=True, default=str)