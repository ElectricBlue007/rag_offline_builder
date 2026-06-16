"""Basic health check for the current project scaffold."""

from __future__ import annotations

import json
import sys


def health_check() -> dict[str, str]:
    return {"module": "cli", "status": "ok", "detail": "project scaffold is available"}


def main() -> int:
    sys.stdout.write(json.dumps(health_check(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
