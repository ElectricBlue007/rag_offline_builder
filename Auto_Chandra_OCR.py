import os
import sys
import subprocess
import time
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

def parse_args():
    args = sys.argv[1:]
    input_dir       = Path(args[0]) if len(args) > 0 else Path("input")
    output_dir      = Path(args[1]) if len(args) > 1 else Path("outputs")
    model_cache_dir = Path(args[2]) if len(args) > 2 else Path("Chandra_models")
    return input_dir, output_dir, model_cache_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_all_pdfs(root: Path) -> list[Path]:
    """Return every .pdf file under root, sorted for deterministic order."""
    return sorted(root.rglob("*.pdf"))


def mirror_output_path(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Given:
        pdf_path   = input/courses/vol1/sec1/chap1/file.pdf
        input_root = input
        output_root= output
    Returns:
        output/courses/vol1/sec1/chap1/file.md
    """
    relative = pdf_path.relative_to(input_root)          # courses/vol1/.../file.pdf
    output_pdf = output_root / relative                   # output/courses/vol1/.../file.pdf
    return output_pdf.with_suffix(".md")                  # → file.md


def run_chandra(input_pdf: Path, output_md: Path, hf_home: Path, env: dict) -> tuple[bool, str]:
    """
    Run:  chandra <input_pdf> <output_md> --method hf
    Returns (success: bool, message: str).
    """
    cmd = ["chandra", str(input_pdf), str(output_md), "--method", "hf", "--max-workers", "4", "--no-headers-footers", "--no-html", "--no-images"]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7600,          # 10-minute max per file — adjust if needed
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "OK"
        else:
            detail = (result.stderr or result.stdout or "no output").strip()
            return False, f"exit code {result.returncode} — {detail}"
    except FileNotFoundError:
        return False, (
            "'chandra' not found on PATH. "
            "Make sure your virtual environment is activated."
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after 2 hours"
    except Exception as exc:
        return False, str(exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_dir, output_dir, model_cache_dir = parse_args()

    # ── Validate input directory ───────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir.resolve()}")
        sys.exit(1)

    # ── Build environment (inherit current env, override HF_HOME) ────────────
    env = os.environ.copy()
    env["HF_HOME"] = str(model_cache_dir.resolve())

    # ── Discover PDFs ─────────────────────────────────────────────────────────
    pdfs = find_all_pdfs(input_dir)
    if not pdfs:
        print(f"[WARN] No PDF files found under: {input_dir.resolve()}")
        sys.exit(0)

    total = len(pdfs)
    print(f"\n{'═'*60}")
    print(f"  Batch Chandra Processor")
    print(f"{'═'*60}")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  Cache  : {model_cache_dir.resolve()}")
    print(f"  PDFs   : {total} file(s) found")
    print(f"{'═'*60}\n")

    succeeded  = []
    failed     = []
    skipped    = []

    for idx, pdf in enumerate(pdfs, start=1):
        out_md = mirror_output_path(pdf, input_dir, output_dir)

        # Create output directory tree if it doesn't exist yet
        out_md.parent.mkdir(parents=True, exist_ok=True)

        label = f"[{idx:>{len(str(total))}}/{total}]"

        # Optional: skip already-converted files (comment out if you want re-runs)
        if out_md.exists():
            print(f"{label} SKIP  {pdf}  →  already converted")
            skipped.append(pdf)
            continue

        print(f"{label} START {pdf}")
        t0 = time.time()
        ok, msg = run_chandra(pdf, out_md, model_cache_dir, env)
        elapsed = time.time() - t0

        if ok:
            print(f"{label} OK    → {out_md}  ({elapsed:.1f}s)")
            succeeded.append(pdf)
        else:
            print(f"{label} FAIL  {pdf}")
            print(f"       Reason: {msg}")
            failed.append((pdf, msg))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Done — {total} PDF(s) processed")
    print(f"  ✓ Succeeded : {len(succeeded)}")
    print(f"  ↷ Skipped   : {len(skipped)}  (already converted)")
    print(f"  ✗ Failed    : {len(failed)}")
    print(f"{'═'*60}")

    if failed:
        print("\n  Failed files:")
        for pdf, reason in failed:
            print(f"    • {pdf}")
            print(f"      {reason}")

        # Write a failure log next to the script for easy review
        log_path = Path("batch_chandra_failures.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"Batch run failed on {len(failed)} file(s)\n\n")
            for pdf, reason in failed:
                fh.write(f"{pdf}\n  {reason}\n\n")
        print(f"\n  Failure log written to: {log_path.resolve()}")
        sys.exit(1)   # non-zero exit so CI/scripts know something went wrong

    sys.exit(0)


if __name__ == "__main__":
    main()