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
            description="a document image with a visible heading",
            model_name=model_name,
        )
        response.validate()
        return response


class TestVisionExtractorIntegration(unittest.TestCase):
    def test_cli_level_ocr_boundary_on_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (64, 32), color="white").save(path)

            result = OCRExtractor(vision_provider=StubVisionProvider()).extract(str(path))

            self.assertEqual(result.source_path, str(path))
            self.assertEqual(result.regions[0].region_type, "ocr")
            self.assertIn("document image", result.regions[0].text)


if __name__ == "__main__":
    unittest.main()
