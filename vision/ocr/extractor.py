"""OCR and image-description extraction for PDFs and images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader

from core.contracts.vision import OCRResult, VisionRegion
from core.contracts.vision_descriptions import VisionDescriptionRequest
from embeddings.vision_provider import TransformersVisionProvider


class OCRExtractor:
    extractor_name = "ocr"

    def __init__(self, vision_provider: TransformersVisionProvider | None = None) -> None:
        self.vision_provider = vision_provider or TransformersVisionProvider()

    def extract(self, file_path: str) -> OCRResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        if path.suffix.lower() == ".pdf":
            regions = self._extract_pdf(path)
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            regions = self._extract_image(path)
        else:
            raise ValueError("OCR extractor currently supports PDF and common image formats")

        result = OCRResult(
            source_path=str(path),
            regions=regions,
            engine=self.vision_provider.debug().get("provider", "transformers-vision"),
            debug_info={"extractor": self.extractor_name, "provider": self.vision_provider.debug()},
        )
        result.validate()
        return result

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def debug(self) -> dict[str, Any]:
        return {"extractor": self.extractor_name, "engine": "fallback"}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def _extract_pdf(self, path: Path) -> list[VisionRegion]:
        reader = PdfReader(str(path))
        regions: list[VisionRegion] = []
        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                regions.append(
                    VisionRegion(
                        region_id=f"{path.stem}-page{page_index}",
                        region_type="ocr",
                        text=text,
                        confidence=1.0,
                        page_number=page_index + 1,
                        metadata={"source": "pdf_text_extraction"},
                    )
                )
                continue

            # fallback to page image generation when text extraction fails
            rendered_pages = convert_from_path(str(path), first_page=page_index + 1, last_page=page_index + 1)
            if rendered_pages:
                image_path = self._rendered_page_path(path, page_index, rendered_pages[0])
                image_text = self._describe_image(image_path, page_index + 1)
                regions.append(
                    VisionRegion(
                        region_id=f"{path.stem}-page{page_index}",
                        region_type="ocr",
                        text=image_text,
                        confidence=0.0,
                        page_number=page_index + 1,
                        metadata={"source": "rendered_page_placeholder"},
                    )
                )
        return regions

    def _extract_image(self, path: Path) -> list[VisionRegion]:
        description = self._describe_image(path, 1)
        return [
            VisionRegion(
                region_id=f"{path.stem}-image",
                region_type="ocr",
                text=description,
                confidence=0.9,
                metadata={"source": "vision_model"},
            )
        ]

    def _rendered_page_path(self, source_path: Path, page_index: int, image: Image.Image) -> Path:
        output_path = source_path.with_suffix("")
        temp_path = output_path.parent / f"{output_path.name}.page{page_index + 1}.png"
        image.save(temp_path)
        return temp_path

    def _describe_image(self, image_path: Path, page_number: int) -> str:
        request = VisionDescriptionRequest(
            request_id=f"{image_path.stem}-{page_number}",
            image_path=str(image_path),
            prompt="Describe the image for document understanding. Preserve text, charts, and visible layout details.",
        )
        try:
            response = self.vision_provider.describe(request)
            return response.description
        except ImportError:
            with Image.open(image_path) as image:
                width, height = image.size
            return f"[vision model unavailable] image {page_number} ({width}x{height})"


def extract_ocr(file_path: str) -> OCRResult:
    return OCRExtractor().extract(file_path)


def result_to_json(result: OCRResult) -> str:
    return json.dumps(result.debug_info, indent=2, sort_keys=True, default=str)
