#!/usr/bin/env python3
"""CLI to extract images from a DOCX and generate descriptions using vision model."""

import sys
import json
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extractors.images.extractor import ImageExtractor


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: extract-images <docx_file>", file=sys.stderr)
        return 1

    docx_path = args[0]
    try:
        extractor = ImageExtractor()
        images = extractor.extract_images(docx_path)
        print(json.dumps(extractor.result_to_json(images), indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())