"""Batch runner for Chandra OCR."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChandraRunResult:
    pdf_path: Path
    output_dir: Path


class ChandraRunner:
    def __init__(self, cache_dir: str = "models/chandra", method: str = "hf") -> None:
        self.cache_dir = cache_dir
        self.method = method

    def run_directory(self, input_dir: str | Path, output_dir: str | Path) -> list[ChandraRunResult]:
        source = Path(input_dir)
        target = Path(output_dir)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF directory does not exist: {source}")
        pdfs = sorted(source.rglob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under: {source}")
        target.mkdir(parents=True, exist_ok=True)
        results = []
        env = os.environ.copy()
        env["HF_HOME"] = self.cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        for pdf_path in pdfs:
            relative_path = pdf_path.relative_to(source)
            out_relative = relative_path.with_name(relative_path.stem + "_markdown")
            out_path = target / out_relative
            out_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["chandra", str(pdf_path), str(out_path), "--method", self.method],
                check=True,
                env=env,
            )
            results.append(ChandraRunResult(pdf_path=pdf_path, output_dir=out_path))
        return results
