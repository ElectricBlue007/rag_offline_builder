from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.contracts.vision_descriptions import VisionDescriptionRequest
from embeddings.vision_provider import TransformersVisionProvider


class DummyVisionProvider(TransformersVisionProvider):
    def _describe_with_transformers(self, request, model_name):  # type: ignore[override]
        from core.contracts.vision_descriptions import VisionDescriptionResponse

        response = VisionDescriptionResponse(
            request_id=request.request_id,
            description="diagram showing a flowchart with three boxes",
            model_name=model_name,
        )
        response.validate()
        return response


class TestVisionProvider(unittest.TestCase):
    def test_describe_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 32), color="white").save(path)

            response = DummyVisionProvider().describe(
                VisionDescriptionRequest(request_id="r1", image_path=str(path), prompt="Describe this image.")
            )

            self.assertIn("flowchart", response.description)


if __name__ == "__main__":
    unittest.main()

