"""Image extraction with vision model integration and strict document ordering."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from core.contracts.document import ImageArtifact
from core.logging import configure_logging, log_event

try:
    from embeddings.vision_provider import TransformersVisionProvider
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

XML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


class ImageExtractor:
    extractor_name = "images"

    def extract_images(self, file_path: str) -> list[ImageArtifact]:
        logger = configure_logging()
        log_event(logger, 20, "image_extract_start", "extracting images", file_path=file_path)
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        if path.suffix.lower() != ".docx":
            return []

        images = self._extract_docx_images(path)
        log_event(logger, 20, "image_extract_end", "image extraction completed", count=len(images))
        return images

    def _extract_docx_images(self, path: Path) -> list[ImageArtifact]:
        with zipfile.ZipFile(path) as archive:
            archive_names = archive.namelist()

            # 1. Map Relationship IDs (rId) to actual filenames
            # Guard: rels file may be absent in minimal/test documents
            rid_to_file: dict[str, str] = {}
            rels_path = "word/_rels/document.xml.rels"
            if rels_path in archive_names:
                rels_xml = archive.read(rels_path)
                rels_root = ET.fromstring(rels_xml)
                for rel in rels_root.findall("rel:Relationship", XML_NS):
                    rid = rel.attrib.get("Id")
                    target = rel.attrib.get("Target", "")
                    rtype = rel.attrib.get("Type", "")
                    if "image" in rtype and target and rid:
                        rid_to_file[rid] = target.replace("media/", "")

            # If no image relationships, return early — nothing to extract
            if not rid_to_file:
                return []

            # 2. Parse document.xml to find the exact chronological order of images
            document_xml = archive.read("word/document.xml")
            doc_root = ET.fromstring(document_xml)
            blips = doc_root.findall(".//a:blip", XML_NS)

            images: list[ImageArtifact] = []
            seen_rids: set[str] = set()

            for idx, blip in enumerate(blips):
                rid = blip.attrib.get(f"{{{XML_NS['r']}}}embed")
                if rid and rid in rid_to_file and rid not in seen_rids:
                    seen_rids.add(rid)
                    filename = rid_to_file[rid]

                    # Load image data for vision model
                    image_data = None
                    media_path = f"word/media/{filename}"
                    if media_path in archive_names:
                        image_data = archive.read(media_path)

                    description = ""
                    if HAS_VISION and image_data:
                        try:
                            from PIL import Image
                            img = Image.open(io.BytesIO(image_data))
                            provider = TransformersVisionProvider()
                            description = provider.describe_image(img)
                        except Exception as e:
                            description = f"[Vision model failed: {e}]"

                    images.append(
                        ImageArtifact(
                            id=filename,
                            alt_text="",
                            caption="",
                            description=description,
                            ocr_text="",
                            source_anchors=[rid],
                            metadata={"rId": rid, "index": idx},
                        )
                    )
        return images

    def validate(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}
    def debug(self) -> dict[str, Any]: return {"extractor": self.extractor_name}
    def health_check(self) -> dict[str, Any]: return {"status": "ok", "extractor": self.extractor_name}


def extract_images(file_path: str) -> list[ImageArtifact]:
    return ImageExtractor().extract_images(file_path)


def result_to_json(result: list[ImageArtifact]) -> str:
    return json.dumps([asdict(img) for img in result], indent=2, sort_keys=True, default=str)