"""Content extraction for DOCX, PDF, and TXT with robust heading detection."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from core.contracts.content import ContentElement, ContentExtractionResult, TextSpan
from core.logging import configure_logging, log_event
from core.utils.style_classifier import StyleClassifier

XML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


class ContentExtractor:
    extractor_name = "content"

    def extract(self, file_path: str) -> ContentExtractionResult:
        logger = configure_logging()
        log_event(logger, 20, "content_extract_start", "extracting content", file_path=file_path)
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        elements = []
        if path.suffix.lower() == ".txt":
            elements = self._extract_txt(path)
        elif path.suffix.lower() == ".docx":
            elements = self._extract_docx(path)
        elif path.suffix.lower() == ".pdf":
            elements = self._extract_pdf(path)
        else:
            raise ValueError(f"unsupported document type: {path.suffix}")

        result = ContentExtractionResult(
            source_path=str(path),
            document_type=path.suffix.lower().replace(".", ""),
            elements=elements,
            debug_info={"extractor": self.extractor_name}
        )
        result.validate()
        log_event(logger, 20, "content_extract_end", "content extraction completed", count=len(elements))
        return result

    def _extract_txt(self, path: Path) -> list[ContentElement]:
        elements = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                is_heading, level = StyleClassifier.classify("", text)
                elements.append(ContentElement(
                    element_id=f"txt-{idx}",
                    element_type="heading" if is_heading else "paragraph",
                    text=text,
                    level=level if is_heading else None,
                    spans=[TextSpan(text=text)]
                ))
        return elements

    def _extract_docx(self, path: Path) -> list[ContentElement]:
        elements = []
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find("w:body", XML_NS)
        if body is None:
            return elements

        for idx, paragraph in enumerate(body.findall("w:p", XML_NS)):
            text = self._paragraph_text(paragraph)
            if not text:
                continue
            
            style_name = self._paragraph_style(paragraph)
            is_heading, level = StyleClassifier.classify(style_name, text)
            
            elements.append(ContentElement(
                element_id=f"docx-{idx}",
                element_type="heading" if is_heading else "paragraph",
                text=text,
                level=level if is_heading else None,
                metadata={"style_name": style_name},
                spans=[TextSpan(text=text, style_name=style_name)]
            ))
        return elements

    def _extract_pdf(self, path: Path) -> list[ContentElement]:
        if not HAS_PYPDF:
            raise ImportError("pypdf is required for PDF extraction")
        
        elements = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    for idx, line in enumerate(text.split('\n')):
                        line = line.strip()
                        if not line:
                            continue
                        is_heading, level = StyleClassifier.classify("", line)
                        elements.append(ContentElement(
                            element_id=f"pdf-p{page_num}-l{idx}",
                            element_type="heading" if is_heading else "paragraph",
                            text=line,
                            level=level if is_heading else None,
                            page_number=page_num + 1,
                            spans=[TextSpan(text=line)]
                        ))
        return elements

    def _paragraph_text(self, paragraph: ET.Element) -> str:
        texts = [node.text for node in paragraph.findall(".//w:t", XML_NS) if node.text]
        return " ".join(texts).strip()

    def _paragraph_style(self, paragraph: ET.Element) -> str:
        style = paragraph.find("w:pPr/w:pStyle", XML_NS)
        if style is None:
            return "Normal"
        return style.attrib.get(f"{{{XML_NS['w']}}}val", "Normal")

    def validate(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}
    def debug(self) -> dict[str, Any]: return {"extractor": self.extractor_name}
    def health_check(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}


def extract_content(file_path: str) -> ContentExtractionResult:
    return ContentExtractor().extract(file_path)

def result_to_json(result: ContentExtractionResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True, default=str)