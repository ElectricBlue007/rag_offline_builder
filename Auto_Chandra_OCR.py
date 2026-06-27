import os
import sys
import subprocess
import time
import tempfile
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
    relative = pdf_path.relative_to(input_root)
    output_pdf = output_root / relative
    return output_pdf.with_suffix(".md")


def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Return the total number of pages in a PDF.
    Tries pypdf → pymupdf → pdfinfo (CLI) in that order.
    Raises RuntimeError if none are available.
    """
    # ── pypdf (pip install pypdf) ──────────────────────────────────────────
    try:
        import pypdf
        with open(pdf_path, "rb") as fh:
            return len(pypdf.PdfReader(fh).pages)
    except ImportError:
        pass

    # ── pymupdf / fitz (pip install pymupdf) ──────────────────────────────
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    except ImportError:
        pass

    # ── pdfinfo command-line tool (poppler-utils) ──────────────────────────
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":")[1].strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(
        f"Cannot determine page count for: {pdf_path}\n"
        "  Install one of:  pip install pypdf   |   pip install pymupdf\n"
        "  or make sure 'pdfinfo' (poppler-utils) is on PATH."
    )


def run_chandra_page(
    input_pdf: Path,
    output_md: Path,
    page_num: int,
    env: dict,
) -> tuple[bool, str]:
    """
    Run:  chandra <input_pdf> <output_md> --method hf --page-range <page_num>
    Returns (success: bool, message: str).
    """
    cmd = [
        "chandra",
        str(input_pdf),
        str(output_md),
        "--method", "hf",
        "--page-range", str(page_num),
        "--no-headers-footers",
        "--no-html",
        "--no-images",
    ]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,          # 5 minutes max per page — adjust if needed
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "OK"
        detail = (result.stderr or result.stdout or "no output").strip()
        return False, f"exit code {result.returncode} — {detail}"
    except FileNotFoundError:
        return False, (
            "'chandra' not found on PATH. "
            "Make sure your virtual environment is activated."
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after 5 minutes"
    except Exception as exc:
        return False, str(exc)


def process_pdf_page_by_page(
    pdf_path: Path,
    out_md: Path,
    env: dict,
) -> tuple[bool, str]:
    """
    Convert a PDF to Markdown one page at a time, then concatenate the
    individual page outputs **in page order** into a single .md file.

    Returns (success: bool, summary_message: str).
    """
    # ── 1. How many pages? ─────────────────────────────────────────────────
    try:
        total_pages = get_pdf_page_count(pdf_path)
    except RuntimeError as exc:
        return False, str(exc)

    print(f"       Pages  : {total_pages}")

    page_texts: list[tuple[int, str]] = []   # (page_number, markdown_text)
    failed_pages: list[tuple[int, str]] = []

    # ── 2. Process each page inside a temp directory ───────────────────────
    with tempfile.TemporaryDirectory(prefix="chandra_pages_") as tmp_dir:
        tmp = Path(tmp_dir)

        for page_num in range(1, total_pages + 1):
            page_md = tmp / f"page_{page_num:06d}.md"

            ok, msg = run_chandra_page(pdf_path, page_md, page_num, env)

            if ok and page_md.exists():
                text = page_md.read_text(encoding="utf-8")
                page_texts.append((page_num, text))
                # Overwrite same line for a compact progress display
                print(
                    f"       Page {page_num:>{len(str(total_pages))}}/{total_pages} ✓",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"\n       Page {page_num:>{len(str(total_pages))}}/{total_pages} "
                    f"✗  {msg}"
                )
                failed_pages.append((page_num, msg))
                # Insert a placeholder so the page slot is not silently dropped
                page_texts.append((
                    page_num,
                    f"\n<!-- [Chandra] Page {page_num} failed: {msg} -->\n",
                ))

        print()   # move past the \r progress line

    # ── 3. Sort by page number (guarantees correct order) ─────────────────
    page_texts.sort(key=lambda t: t[0])

    # ── 4. Concatenate and write final output ──────────────────────────────
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        for page_num, text in page_texts:
            fh.write(text)
            if text and not text.endswith("\n"):
                fh.write("\n")

    if failed_pages:
        pages_list = ", ".join(str(p) for p, _ in failed_pages)
        return False, f"{len(failed_pages)} page(s) failed (pages: {pages_list})"

    return True, f"OK — {total_pages} page(s) written in order"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_dir, output_dir, model_cache_dir = parse_args()

    # ── Validate input directory ───────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir.resolve()}")
        sys.exit(1)

    # ── Build environment ──────────────────────────────────────────────────
    env = os.environ.copy()
    env["HF_HOME"]        = str(model_cache_dir.resolve())
    env["HF_HUB_OFFLINE"] = "1"   # never try to reach huggingface.co

    # ── Discover PDFs ──────────────────────────────────────────────────────
    pdfs = find_all_pdfs(input_dir)
    if not pdfs:
        print(f"[WARN] No PDF files found under: {input_dir.resolve()}")
        sys.exit(0)

    total = len(pdfs)
    print(f"\n{'═'*60}")
    print(f"  Batch Chandra Processor  (page-by-page · offline)")
    print(f"{'═'*60}")
    print(f"  Input        : {input_dir.resolve()}")
    print(f"  Output       : {output_dir.resolve()}")
    print(f"  Cache        : {model_cache_dir.resolve()}")
    print(f"  PDFs found   : {total}")
    print(f"  HF_HUB_OFFLINE=1  (no network calls to HuggingFace)")
    print(f"{'═'*60}\n")

    succeeded: list[Path]              = []
    failed:    list[tuple[Path, str]]  = []
    skipped:   list[Path]              = []

    for idx, pdf in enumerate(pdfs, start=1):
        out_md = mirror_output_path(pdf, input_dir, output_dir)
        out_md.parent.mkdir(parents=True, exist_ok=True)

        label = f"[{idx:>{len(str(total))}}/{total}]"

        # Skip already-converted files (comment out to force re-runs)
        if out_md.exists():
            print(f"{label} SKIP  {pdf}  →  already converted")
            skipped.append(pdf)
            continue

        print(f"{label} START {pdf}")
        t0 = time.time()
        ok, msg = process_pdf_page_by_page(pdf, out_md, env)
        elapsed = time.time() - t0

        if ok:
            print(f"{label} OK    → {out_md}  ({elapsed:.1f}s)")
            succeeded.append(pdf)
        else:
            print(f"{label} FAIL  {pdf}")
            print(f"       Reason: {msg}")
            failed.append((pdf, msg))

    # ── Summary ────────────────────────────────────────────────────────────
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

        log_path = Path("batch_chandra_failures.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"Batch run failed on {len(failed)} file(s)\n\n")
            for pdf, reason in failed:
                fh.write(f"{pdf}\n  {reason}\n\n")
        print(f"\n  Failure log written to: {log_path.resolve()}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()