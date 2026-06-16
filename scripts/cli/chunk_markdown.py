"""Standalone chunking CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    sys.stderr.write("error: chunking CLI expects markdown/enrichment contracts from code, not a raw file path\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

