"""Standalone enrichment CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enrichment.processor import EnrichmentProcessor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: enrich_markdown.py <file-path>\n")
        return 2

    sys.stderr.write("error: enrichment CLI expects a markdown contract from code, not a raw file path\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

