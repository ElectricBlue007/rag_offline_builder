"""Standalone pipeline CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.logging import configure_logging, log_event  # noqa: E402
from pipeline.orchestrator import PipelineOrchestrator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: run_pipeline.py <file-path>\n")
        return 2

    try:
        logger = configure_logging()
        log_event(logger, 20, "cli_start", "pipeline cli started", file_path=args[0])
        result = PipelineOrchestrator().run(args[0])
        sys.stdout.write(
            json.dumps(
                {
                    "source_path": result.source_path,
                    "document_id": result.document_id,
                    "records": [record.debug() for record in result.records],
                    "debug_info": result.debug_info,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        log_event(logger, 20, "cli_end", "pipeline cli completed", file_path=args[0])
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
