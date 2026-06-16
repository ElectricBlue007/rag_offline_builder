"""Standalone content extraction CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.content.extractor import ContentExtractor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: extract_content.py <file-path>\n")
        return 2

    try:
        result = ContentExtractor().extract(args[0])
        sys.stdout.write(
            json.dumps(
                {
                    "source_path": result.source_path,
                    "document_type": result.document_type,
                    "elements": [element.debug() for element in result.elements],
                    "debug_info": result.debug_info,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

