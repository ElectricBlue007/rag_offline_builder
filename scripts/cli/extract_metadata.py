"""Standalone metadata extraction CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.metadata.extractor import MetadataExtractor
from core.contracts.extraction import SourceDocumentRequest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: extract_metadata.py <file-path>\n")
        return 2

    try:
        request = SourceDocumentRequest(file_path=args[0])
        result = MetadataExtractor().extract(request)
        sys.stdout.write(
            json.dumps(result.debug_info | {"metadata": result.metadata.debug()}, indent=2, sort_keys=True, default=str) + "\n"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
