#!/usr/bin/env python3
"""
Batch OCR processor using Chandra.
Recursively finds all PDFs inside ./input and runs Chandra OCR on each one.
Results are saved under ./outputs/<relative_path_of_pdf>/
"""

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).parent

os.environ["HF_HOME"] = str(BASE_DIR / "chandra_models")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def find_all_pdfs(input_dir: Path) -> list[Path]:
    """Recursively find all PDF files inside input_dir."""
    return sorted(input_dir.rglob("*.pdf"))


def run_chandra(pdf_path: Path, output_base: Path, method: str = "hf") -> bool:
    """
    Run Chandra OCR on a single PDF file.
    Returns True if successful, False otherwise.
    """
    cmd = ["chandra", str(pdf_path), str(output_base), "--method", method]

    print(f"\n{'='*60}")
    print(f"  Processing: {pdf_path}")
    print(f"  Output dir: {output_base}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            print(f"  ✅ Done: {pdf_path.name}")
            return True
        else:
            print(f"  ❌ Failed (exit code {result.returncode}): {pdf_path.name}")
            return False
    except FileNotFoundError:
        print("  ❌ ERROR: 'chandra' command not found.")
        print("     Make sure your virtual environment is activated.")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Unexpected error for {pdf_path.name}: {e}")
        return False


def main():
    # ── Configurable paths ──────────────────────────────────────────────────
    base_dir    = Path(__file__).parent          # folder where this script lives
    input_dir   = base_dir / "input"
    output_dir  = base_dir / "outputs"
    method      = "hf"                           # change to "tesseract" etc. if needed
    # ────────────────────────────────────────────────────────────────────────

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    pdfs = find_all_pdfs(input_dir)

    if not pdfs:
        print(f"⚠️  No PDF files found inside: {input_dir}")
        sys.exit(0)

    print(f"\n📂 Input directory : {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔍 Found {len(pdfs)} PDF(s) to process:\n")
    for p in pdfs:
        print(f"   • {p.relative_to(input_dir)}")

    succeeded = []
    failed    = []

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"\n[{i}/{len(pdfs)}]", end="")
        ok = run_chandra(pdf_path, output_dir, method=method)
        (succeeded if ok else failed).append(pdf_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  ✅ Succeeded : {len(succeeded)}")
    print(f"  ❌ Failed    : {len(failed)}")
    if failed:
        print("\n  Failed files:")
        for f in failed:
            print(f"    • {f.relative_to(input_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()