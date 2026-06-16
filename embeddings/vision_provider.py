"""Vision-language provider for image descriptions and OCR-style captions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import uuid

from PIL import Image

from core.contracts.vision_descriptions import VisionDescriptionRequest, VisionDescriptionResponse


class TransformersVisionProvider:
    """Local open-source vision-language provider using Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",   # change here
        fallback_model_name: str = "llava-hf/llava-1.5-7b-hf",
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name

    def describe(self, request: VisionDescriptionRequest) -> VisionDescriptionResponse:
        request.validate()
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        try:
            return self._describe_with_transformers(request, self.model_name)
        except ImportError:
            return self._describe_with_transformers(request, self.fallback_model_name)

    def describe_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Convenience method to describe a PIL image directly."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            from core.contracts.vision_descriptions import VisionDescriptionRequest
            request = VisionDescriptionRequest(
                request_id="image_" + str(uuid.uuid4()),
                image_path=tmp_path,
                prompt=prompt,
            )
            response = self.describe(request)
            return response.description
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "transformers-vision", "model": self.model_name}

    def debug(self) -> dict[str, Any]:
        return {"provider": "transformers-vision", "model": self.model_name, "fallback_model": self.fallback_model_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "transformers-vision"}

    def _describe_with_transformers(self, request: VisionDescriptionRequest, model_name: str) -> VisionDescriptionResponse:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Transformers vision backend is unavailable. Install `transformers`, `torch`, and `accelerate`."
            ) from exc

        image = Image.open(request.image_path).convert("RGB")
        model = AutoModelForImageTextToText.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)

        prompt = request.prompt.strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        input_len = len(inputs.input_ids[0])
        description = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        response = VisionDescriptionResponse(
            request_id=request.request_id,
            description=description,
            model_name=model_name,
            metadata={"image_path": request.image_path, "prompt": request.prompt},
        )
        response.validate()
        return response