from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.contracts.vision_descriptions import VisionDescriptionRequest
from embeddings.vision_provider import TransformersVisionProvider


class StubVisionProvider(TransformersVisionProvider):
    def _describe_with_transformers(self, request, model_name):  # type: ignore[override]
        from core.contracts.vision_descriptions import VisionDescriptionResponse

        response = VisionDescriptionResponse(
            request_id=request.request_id,
            description="a simple photo of a page with text blocks",
            model_name=model_name,
        )
        response.validate()
        return response


class TestVisionProviderIntegration(unittest.TestCase):
    def test_pipeline_level_image_description_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (64, 48), color="white").save(path)

            response = StubVisionProvider().describe(
                VisionDescriptionRequest(request_id="r1", image_path=str(path), prompt="Describe the image.")
            )

            self.assertEqual(response.model_name, "Qwen/Qwen2-VL-2B-Instruct")


if __name__ == "__main__":
    unittest.main()

