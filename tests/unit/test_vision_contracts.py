from __future__ import annotations

import unittest

from core.contracts.vision import OCRResult, VisionRegion


class TestVisionContracts(unittest.TestCase):
    def test_vision_region_validation(self) -> None:
        region = VisionRegion(region_id="r1", region_type="ocr", text="Hello", confidence=0.9)
        region.validate()
        self.assertEqual(region.debug()["region_type"], "ocr")

    def test_ocr_result_validation(self) -> None:
        result = OCRResult(source_path="/tmp/sample.pdf", regions=[VisionRegion(region_id="r1", region_type="ocr")])
        result.validate()


if __name__ == "__main__":
    unittest.main()

