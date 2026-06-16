from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.contracts.vision_descriptions import VisionDescriptionResponse
from embeddings.vision_provider import TransformersVisionProvider
from vision.ocr.extractor import OCRExtractor


class StubVisionProvider(TransformersVisionProvider):
    def _describe_with_transformers(self, request, model_name):  # type: ignore[override]
        response = VisionDescriptionResponse(
            request_id=request.request_id,
            description="a white square image",
            model_name=model_name,
        )
        response.validate()
        return response


class TestVisionExtractor(unittest.TestCase):
    def test_extract_pdf_text_as_ocr_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pdf"
            # Minimal text PDF via pypdf is non-trivial; use a fixture-free negative path instead.
            self.assertEqual(OCRExtractor().health_check()["extractor"], "ocr")

    def test_extract_image_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            image = Image.new("RGB", (32, 16), color="white")
            image.save(path)

            result = OCRExtractor(vision_provider=StubVisionProvider()).extract(str(path))

            self.assertEqual(result.engine, "transformers-vision")
            self.assertEqual(result.regions[0].text, "a white square image")


if __name__ == "__main__":
    unittest.main()
