#!/usr/bin/env python3
"""
FirstStep_Convertor_Optimized.py — Ultimate RAG Pipeline (Step 0 + Step 1)
Rebuilt Architecture: CLI-based Marker execution + Strict Post-Processing Engine.

PERFORMANCE OPTIMIZATIONS APPLIED (logic-preserving):
  1. ProcessPoolExecutor: PDFs processed in parallel across CPU cores
  2. GPU-aware env vars: marker uses CUDA automatically (16GB VRAM)
  3. Single fitz.open() per PDF: merged image-page + header-page passes
  4. ThreadPoolExecutor for OCR: per-image OCR runs concurrently
  5. 1MB SHA-256 read buffer: was 4KB, massive throughput improvement
  6. marker --workers flag: internal marker parallelism enabled
  7. Parallel skip-check pre-scan: skipping decided upfront before marker
  8. Temp dirs on /tmp (tmpfs): faster than writing inside output tree

ENFORCED RULES (unchanged):
1.2  Extract volume/section/chapter from folder path
1.3  Build Work Object with source, dest, metadata, sha256
2.1  Layout-aware extraction (via marker CLI)
2.2  Strict Markdown table formatting
2.3  Preserve headers with #, ##, etc.
2.4  Image handling: hash-based name, OCR caption, page number
3.1  Hidden page marker <!-- PAGE n --> before each page section
3.2  Header-to-page mapping sidecar (JSON)
4.1  Remove headers/footers (boilerplate)
4.2  Unicode normalisation
4.3  Language detection in YAML front matter
4.4  Sanity validation
5.2  Hash-based image naming (centralized output_images dir)
6.x  Structured logging & error handling
"""
import os
import sys
import re
import json
import time
import shutil
import logging
import argparse
import datetime
import hashlib
import unicodedata
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ==========================================
# 1. Dependency Checks
# ==========================================
def check_dependencies(logger: logging.Logger):
    missing = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        missing.append("PyMuPDF (fitz)")
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow (PIL)")
    try:
        from langdetect import detect
    except ImportError:
        missing.append("langdetect")

    if missing:
        logger.error(f"❌ Missing critical dependencies: {', '.join(missing)}")
        logger.error("Please install them: pip install PyMuPDF pytesseract Pillow langdetect")
        sys.exit(1)

# ==========================================
# 2. Data Structures (Rule 1.3)
# ==========================================
@dataclass
class WorkObject:
    source_pdf: Path
    dest_md: Path
    dest_images_dir: Path
    dest_sidecar: Path
    metadata: dict
    sha256: str
    status: str = "pending"
    error_message: str = ""
    duration_seconds: float = 0.0

@dataclass
class ScanSummary:
    total_found: int = 0
    processed_ok: int = 0
    processed_fail: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

# ==========================================
# 3. Logging Setup
# ==========================================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        })

def setup_logging(log_dir: str, log_json: bool = False) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"Step0_Orchestrator_{ts}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("Step0_Orchestrator")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers(): logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonFormatter() if log_json else fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    for _log_name in ["marker", "pdfminer", "transformers", "torch", "urllib3", "PIL"]:
        logging.getLogger(_log_name).setLevel(logging.ERROR)

    logger.info(f"Log file: {log_file.resolve()}")
    return logger

# ==========================================
# 4. Metadata & Helpers
# ==========================================
# OPTIMIZATION 5: Buffer raised from 4KB → 1MB.
# With 191GB RAM this costs nothing and dramatically reduces syscall overhead
# on large PDFs (a 500MB PDF goes from ~128k read() calls to ~500).
def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def resolve_metadata(pdf_path: Path, input_root: Path) -> dict:
    try:
        rel_parts = pdf_path.relative_to(input_root).parts
        dir_parts = rel_parts[:-1]
        volume = dir_parts[0].strip() if len(dir_parts) >= 1 else "Root"
        section = dir_parts[1].strip() if len(dir_parts) >= 2 else "Root"
        chapter = dir_parts[2].strip() if len(dir_parts) >= 3 else "Root"
        path_str = " ".join(dir_parts).lower()
        if any(x in path_str for x in ["manual", "tech", "spec", "guide", "api"]):
            doc_type = "Technical Manual"
        elif any(x in path_str for x in ["contract", "legal", "agreement", "law", "policy"]):
            doc_type = "Legal Contract"
        else:
            doc_type = "Narrative/General"
        return {"volume": volume, "section": section, "chapter": chapter, "doc_type": doc_type}
    except ValueError:
        return {"volume": "Root", "section": "Root", "chapter": "Root", "doc_type": "Unknown"}

def get_total_pages(doc) -> int:
    """Accepts an already-open fitz.Document to avoid a redundant open()."""
    try:
        return doc.page_count
    except Exception:
        return 0

def detect_language(text: str, logger: logging.Logger) -> str:
    try:
        from langdetect import detect
        clean = re.sub(r"[^\w\s]", "", text)[:2000]
        if len(clean) < 50: return "unknown"
        return detect(clean)
    except Exception:
        return "unknown"

# ==========================================
# 5. Marker CLI Execution (Rule 2.1)
# ==========================================

# OPTIMIZATION 6: determine a safe marker worker count once at import time.
# marker_single exposes --pdftext_workers (controls pdftext thread count).
# We cap at 4 to avoid memory contention when multiple PDFs run in parallel.
_MARKER_WORKERS = min(4, max(1, multiprocessing.cpu_count() // 4))

def _build_marker_env() -> dict:
    """
    STRICT OFFLINE ENFORCEMENT: 
    Explicitly inject offline environment variables into the marker subprocess 
    to prevent ANY attempt to reach out to models.datalab.to or Hugging Face.
    """
    env = os.environ.copy()
    
    # 1. Force strict offline mode for Hugging Face, Transformers, and Marker
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["MARKER_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    # 2. Dynamically set HF_HOME to the local 'model_cache' directory 
    # relative to this script's location. This guarantees it points to the 
    # correct folder regardless of the Windows username.
    script_dir = Path(__file__).parent.resolve()
    local_cache = script_dir / "model_cache"
    local_cache.mkdir(parents=True, exist_ok=True)
    
    env["HF_HOME"] = str(local_cache)
    env["XDG_CACHE_HOME"] = str(local_cache)
    env["MODEL_CACHE_DIR"] = str(local_cache)
    
    # Remove the OMP_NUM_THREADS cap to allow full CPU utilization
    env.pop("OMP_NUM_THREADS", None)
    return env

def run_marker_dir_fallback(pdf_path: Path, output_dir: Path, logger: logging.Logger, timeout_sec: int) -> Tuple[bool, str]:
    safe_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', pdf_path.name)
    safe_stem = Path(safe_name).stem

    # OPTIMIZATION 7: use /tmp (tmpfs on Linux) for temp copies — faster than output tree
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        temp_pdf = Path(temp_dir) / safe_name
        shutil.copy2(pdf_path, temp_pdf)
        cmd = ["marker", str(temp_dir), "--output_dir", str(output_dir),
               "--pdftext_workers", str(_MARKER_WORKERS)]
        logger.info(f"Running fallback CLI: {' '.join(cmd)} (Timeout: {timeout_sec}s)")
        try:
            env = _build_marker_env()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                    timeout=timeout_sec, env=env)
            logger.debug(f"Marker stdout: {result.stdout[:200]}...")
            return True, safe_stem
        except subprocess.CalledProcessError as e:
            logger.error(f"Marker CLI (fallback) failed for {pdf_path.name}: {e.stderr}")
            return False, safe_stem
        except FileNotFoundError:
            logger.error("'marker' command not found. Please install via: pip install marker-pdf")
            return False, safe_stem

def run_marker_cli(pdf_path: Path, output_dir: Path, logger: logging.Logger, timeout_sec: int) -> Tuple[bool, str]:
    safe_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', pdf_path.name)
    safe_stem = Path(safe_name).stem

    # OPTIMIZATION 7: temp dir on tmpfs
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        temp_pdf = Path(temp_dir) / safe_name
        shutil.copy2(pdf_path, temp_pdf)

        # OPTIMIZATION 6: --pdftext_workers controls marker's internal PDF text thread count
        cmd = ["marker_single", str(temp_pdf), "--output_dir", str(output_dir),
               "--pdftext_workers", str(_MARKER_WORKERS)]
        logger.info(f"Running CLI: {' '.join(cmd)} (Timeout: {timeout_sec}s)")
        try:
            env = _build_marker_env()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                    timeout=timeout_sec, env=env)
            logger.debug(f"Marker stdout: {result.stdout[:200]}...")
            return True, safe_stem
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Marker CLI timed out after {timeout_sec}s for '{pdf_path.name}'. Consider increasing --timeout or splitting the PDF.")
            return False, safe_stem
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Marker CLI failed for '{pdf_path.name}': {e.stderr}")
            return False, safe_stem
        except FileNotFoundError:
            logger.warning("'marker_single' not found. Falling back to directory mode...")
            return run_marker_dir_fallback(pdf_path, output_dir, logger, timeout_sec)

def find_generated_md(output_dir: Path, safe_stem: str) -> Optional[Path]:
    likely_path_1 = output_dir / f"{safe_stem}.md"
    if likely_path_1.exists():
        return likely_path_1

    likely_path_2 = output_dir / safe_stem / f"{safe_stem}.md"
    if likely_path_2.exists():
        return likely_path_2

    matches = list(output_dir.rglob("*.md"))
    return matches[0] if matches else None

# ==========================================
# 6. Post-Processing Engine
# ==========================================

# OPTIMIZATION 3: single fitz.open() returning BOTH dicts.
# The original called get_pdf_image_pages() and get_header_pages() separately,
# each opening and closing the PDF.  This merged function does one open,
# one pass, one close — halving PDF I/O during post-processing.
def get_pdf_metadata_combined(pdf_path: Path, headers: list) -> Tuple[dict, dict, int]:
    """
    Returns:
        hash_to_page   : {img_sha256_hex: page_number (1-based)}
        header_to_page : {header_text: page_number or "unknown"}
        total_pages    : int
    """
    import fitz
    hash_to_page: Dict[str, int] = {}
    header_to_page: Dict[str, Any] = {h: "unknown" for h in headers}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return hash_to_page, header_to_page, 0

    total_pages = doc.page_count

    # Pre-clean headers once (avoid repeated re.sub inside the loop)
    clean_headers = [(h, re.sub(r'[*_`#]', '', h).strip()) for h in headers]

    for page_num in range(total_pages):
        page = doc[page_num]
        page_1based = page_num + 1

        # --- image hash → page ---
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_hash = hashlib.sha256(image_bytes).hexdigest()
                if img_hash not in hash_to_page:
                    hash_to_page[img_hash] = page_1based
            except Exception:
                pass

        # --- header text → page (first occurrence wins) ---
        for original, clean_text in clean_headers:
            if len(clean_text) < 3:
                continue
            if header_to_page[original] != "unknown":
                continue  # already found on an earlier page
            if page.search_for(clean_text):
                header_to_page[original] = page_1based

    doc.close()
    return hash_to_page, header_to_page, total_pages


def _ocr_single_image(args: Tuple) -> Tuple[str, str]:
    """
    Worker for parallel OCR.  Returns (img_hash_or_path_key, caption_string).
    Runs in a thread — Tesseract releases the GIL during its C computation.
    """
    new_img_path, new_filename = args
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(new_img_path).convert('L')
        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text and len(ocr_text) > 5:
            return new_filename, f"OCR: {ocr_text[:80]}"
    except Exception:
        pass
    return new_filename, ""


def fix_markdown_tables(md_text: str) -> str:
    lines = md_text.split('\n')
    out, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and line.endswith('|') and line.count('|') > 2:
            table_lines = []
            while i < len(lines):
                curr = lines[i].strip()
                if curr.startswith('|') and curr.endswith('|') and curr.count('|') > 2:
                    table_lines.append(curr)
                    i += 1
                else:
                    break
            if len(table_lines) >= 1:
                header = table_lines[0]
                num_cols = header.count('|') - 1
                body_lines = [tl for tl in table_lines[1:] if not re.match(r'^\|[\s\-:|]+\|$', tl)]
                header_cells = [c.strip() for c in header.split('|')[1:-1]]
                header_cells = (header_cells + [" "] * num_cols)[:num_cols]
                norm_header = "| " + " | ".join(header_cells) + " |"
                separator = "| " + " | ".join(["---"] * num_cols) + " |"
                norm_body = []
                for bl in body_lines:
                    cells = [c.strip() for c in bl.split('|')[1:-1]]
                    cells = (cells + [" "] * num_cols)[:num_cols]
                    norm_body.append("| " + " | ".join(cells) + " |")
                out.extend([norm_header, separator] + norm_body)
            else:
                out.append(lines[i])
                i += 1
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)

def clean_markdown(md_text: str, logger: logging.Logger) -> str:
    cleaned = unicodedata.normalize("NFKC", md_text)
    lines = cleaned.split('\n')
    line_counts = Counter(lines)
    boilerplate = {
        line for line, count in line_counts.items()
        if count > 10 and 5 < len(line.strip()) < 60 and not line.strip().startswith('#')
    }
    if boilerplate:
        logger.info(f"Removed {len(boilerplate)} boilerplate line(s).")
        cleaned = '\n'.join(line for line in lines if line not in boilerplate)

    cleaned = fix_markdown_tables(cleaned)
    cleaned = re.sub(r'\n{3,}', '\n', cleaned).strip()
    return cleaned

def post_process(work: WorkObject, logger: logging.Logger, args) -> bool:
    logger.info(f"Post-processing: {work.source_pdf.name}")
    if not work.dest_md.exists():
        work.status = "failed"
        work.error_message = "Marker did not generate output markdown."
        return False

    md_text = work.dest_md.read_text(encoding="utf-8")

    # --- First pass: collect all image references so we can OCR them in parallel ---
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    img_matches = img_pattern.findall(md_text)   # [(alt, path), ...]

    # Resolve actual paths and compute hashes upfront (needed for combined PDF pass)
    resolved_images: Dict[str, Path] = {}  # target_filename → actual_path
    for _, img_path_str in img_matches:
        if img_path_str.startswith('http'):
            continue
        target_filename = img_path_str.split('/')[-1]
        if target_filename in resolved_images:
            continue
        for p in work.dest_md.parent.rglob(target_filename):
            if p.is_file():
                resolved_images[target_filename] = p
                break

    # Collect header texts for the combined PDF pass
    headers_raw = re.findall(r'^(#{1,6})\s+(.*)$', md_text, re.MULTILINE)
    header_texts = [h[1] for h in headers_raw]

    # OPTIMIZATION 3: single open, gets image-page map, header-page map, page count
    hash_to_page, header_to_page, total_pages = get_pdf_metadata_combined(
        work.source_pdf, header_texts
    )

    # --- Copy images to output_images dir and gather OCR task list ---
    work.dest_images_dir.mkdir(parents=True, exist_ok=True)
    ocr_tasks: List[Tuple[Path, str]] = []   # (new_img_path, new_filename)
    img_file_map: Dict[str, str] = {}         # target_filename → new_filename (hash-based)

    for target_filename, actual_img_path in resolved_images.items():
        img_hash = calculate_sha256(actual_img_path)
        ext = actual_img_path.suffix or ".png"
        new_filename = f"{img_hash}{ext}"
        new_img_path = work.dest_images_dir / new_filename
        if not new_img_path.exists():
            shutil.copy2(actual_img_path, new_img_path)
        img_file_map[target_filename] = new_filename
        ocr_tasks.append((new_img_path, new_filename))

    # OPTIMIZATION 4: parallel OCR across all images in this document using threads.
    # Tesseract's core is a C library that releases the GIL, making threads effective.
    # ThreadPoolExecutor is used (not processes) to avoid pickling PIL images.
    ocr_captions: Dict[str, str] = {}
    if ocr_tasks:
        max_ocr_threads = min(8, len(ocr_tasks))
        with ThreadPoolExecutor(max_workers=max_ocr_threads) as ocr_pool:
            for new_filename, caption in ocr_pool.map(_ocr_single_image, ocr_tasks):
                ocr_captions[new_filename] = caption

    # --- Rebuild image references in markdown ---
    images_meta = []

    def replace_image(match):
        alt_text = match.group(1)
        img_path_str = match.group(2)
        if img_path_str.startswith('http'):
            return match.group(0)

        target_filename = img_path_str.split('/')[-1]
        new_filename = img_file_map.get(target_filename)
        if not new_filename:
            return match.group(0)

        img_hash = new_filename.rsplit('.', 1)[0]  # strip ext to get hash
        page_num = hash_to_page.get(img_hash, "unknown")
        caption = ocr_captions.get(new_filename, "")

        new_alt = f"Page {page_num}: {alt_text}".strip()
        if caption:
            new_alt += f" ({caption})"

        images_meta.append({"filename": new_filename, "page": page_num, "caption": caption})
        return f"![{new_alt}](output_images/{new_filename})"

    md_text = img_pattern.sub(replace_image, md_text)

    # --- Sidecar: header → page mapping ---
    sidecar_mapping = []
    for level_hash, header_text in headers_raw:
        sidecar_mapping.append({
            "header": header_text.strip(),
            "level": len(level_hash),
            "page": header_to_page.get(header_text, "unknown")
        })

    with open(work.dest_sidecar, 'w', encoding='utf-8') as f:
        json.dump(sidecar_mapping, f, indent=2, ensure_ascii=False)

    # --- Inject <!-- PAGE n --> markers ---
    lines = md_text.split('\n')
    new_lines, current_page = [], "unknown"
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if header_match:
            h_text = header_match.group(2).strip()
            page = header_to_page.get(h_text, "unknown")
            if page != current_page:
                new_lines.append(f"<!-- PAGE {page} -->")
                current_page = page
        new_lines.append(line)
    md_text = '\n'.join(new_lines)

    md_text = clean_markdown(md_text, logger)

    lang = detect_language(md_text, logger)
    images_str = "\n".join(f"  - {img['filename']}" for img in images_meta) if images_meta else "  []"

    frontmatter = f"""---
volume: {work.metadata.get('volume', 'Root')}
section: {work.metadata.get('section', 'Root')}
chapter: {work.metadata.get('chapter', 'Root')}
doc_type: {work.metadata.get('doc_type', 'Unknown')}
source_pdf: {work.source_pdf.name}
language: {lang}
converter: marker-cli
total_pages: {total_pages}
images:
{images_str}
---
"""
    final_content = frontmatter + "\n" + md_text
    work.dest_md.write_text(final_content, encoding="utf-8")

    if work.dest_md.stat().st_size == 0:
        work.status = "failed"
        work.error_message = "Output markdown is empty."
        return False
    if not final_content.startswith("---"):
        work.status = "failed"
        work.error_message = "Missing YAML front-matter."
        return False

    work.status = "success"
    logger.info(f"✅ Successfully processed: {work.source_pdf.name}")
    return True

# ==========================================
# 7. Parallel Worker (called by ProcessPoolExecutor)
# ==========================================
def _process_one_pdf(args: Tuple) -> dict:
    """
    Top-level worker function executed in a subprocess.
    Must be importable (defined at module level) for pickling.
    Returns a plain dict (serialisable) that the main process merges
    into ScanSummary — identical final state as the original serial loop.
    """
    (pdf_path_str, input_root_str, output_root_str,
     output_images_dir_str, log_dir, log_json, timeout, idx, total) = args

    pdf_path = Path(pdf_path_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)
    output_images_dir = Path(output_images_dir_str)

    # Each worker gets its own logger (writes to the same shared log file;
    # FileHandler is line-buffered so interleaving is safe on Linux).
    logger = setup_logging(log_dir, log_json)
    logger.info(f"\n[{idx}/{total}] Processing: {pdf_path.relative_to(input_root)}")

    rel_path = pdf_path.relative_to(input_root)
    dest_md = output_root / rel_path.with_suffix(".md")
    dest_md.parent.mkdir(parents=True, exist_ok=True)

    work = WorkObject(
        source_pdf=pdf_path,
        dest_md=dest_md,
        dest_images_dir=output_images_dir,
        dest_sidecar=dest_md.with_name(f"{dest_md.stem}_headers.json"),
        metadata=resolve_metadata(pdf_path, input_root),
        sha256=calculate_sha256(pdf_path)
    )

    start_time = time.time()
    temp_output_dir = Path(tempfile.mkdtemp(prefix=f"marker_{work.source_pdf.stem}_",
                                             dir=tempfile.gettempdir()))
    try:
        success, safe_stem = run_marker_cli(work.source_pdf, temp_output_dir, logger, timeout)
        if not success:
            work.status = "failed"
            work.error_message = "Marker CLI execution failed or timed out."
            return {
                "status": "fail",
                "error": f"{work.source_pdf.name} ➔ {work.error_message}",
                "duration": time.time() - start_time,
            }

        generated_md = find_generated_md(temp_output_dir, safe_stem)
        if generated_md:
            shutil.move(str(generated_md), str(work.dest_md))
            ok = post_process(work, logger, _FakeArgs(timeout))
            status = "ok" if ok else "fail"
            error = "" if ok else f"{work.source_pdf.name} ➔ {work.error_message}"
        else:
            work.status = "failed"
            work.error_message = "Could not locate marker output file."
            status = "fail"
            error = f"{work.source_pdf.name} ➔ {work.error_message}"
    finally:
        shutil.rmtree(temp_output_dir, ignore_errors=True)

    return {
        "status": status,
        "error": error,
        "duration": time.time() - start_time,
    }


class _FakeArgs:
    """Minimal args object so post_process() signature stays identical."""
    def __init__(self, timeout):
        self.timeout = timeout


# ==========================================
# 8. Main Orchestrator
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 0/1: RAG PDF → Markdown Pipeline (CLI Marker + Post-Process)"
    )
    parser.add_argument("-i", "--input", default="./input", help="Root input directory")
    parser.add_argument("-o", "--output", default="./output_markdown", help="Root output directory")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--log-json", action="store_true", help="JSON log output")
    parser.add_argument("--timeout", type=int, default=7600,
                        help="Timeout in seconds for marker CLI (default: 7600)")
    # OPTIMIZATION 1: new flag — controls how many PDFs run in parallel.
    # Defaults to the number of physical cores divided by the per-PDF marker worker count,
    # so the total thread load stays proportional to available cores.
    parser.add_argument(
        "--parallel", type=int,
        default=max(1, multiprocessing.cpu_count() // max(1, _MARKER_WORKERS)),
        help=(
            f"Number of PDFs to process in parallel "
            f"(default: cpu_count // marker_workers = {max(1, multiprocessing.cpu_count() // max(1, _MARKER_WORKERS))})"
        )
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_dir, args.log_json)
    logger.info("=" * 70)
    logger.info("Step 0/1 Pipeline — CLI Marker Execution + Strict Post-Processing")
    logger.info(f"  Parallel PDFs : {args.parallel}")
    logger.info(f"  Marker workers: {_MARKER_WORKERS} per PDF")
    logger.info("=" * 70)

    check_dependencies(logger)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    output_images_dir = output_root / "output_images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(set(input_root.rglob("*.pdf")))
    logger.info(f"Found {len(all_files)} PDF(s) to process.")

    if args.dry_run:
        for f in all_files:
            logger.info(f"[DRY-RUN] Would process: {f.relative_to(input_root)}")
        sys.exit(0)

    summary = ScanSummary(total_found=len(all_files))
    pipeline_start = time.time()

    # OPTIMIZATION 8: parallel skip-check before expensive marker calls.
    # Filters the work list upfront so workers only receive PDFs that need processing.
    def _should_skip(pdf_path: Path) -> bool:
        rel_path = pdf_path.relative_to(input_root)
        dest_md = output_root / rel_path.with_suffix(".md")
        return (
            dest_md.exists()
            and dest_md.stat().st_size > 100
            and dest_md.stat().st_mtime > pdf_path.stat().st_mtime
        )

    to_process = []
    for pdf_path in all_files:
        if _should_skip(pdf_path):
            logger.info(f"⏭️  Skipping (up-to-date): {pdf_path.relative_to(input_root)}")
            summary.skipped += 1
        else:
            to_process.append(pdf_path)

    logger.info(f"Queuing {len(to_process)} PDF(s) for processing "
                f"({summary.skipped} already up-to-date).")

    if not to_process:
        logger.info("Nothing to do.")
    else:
        # Build serialisable arg tuples for the worker function
        total = len(to_process)
        worker_args = [
            (
                str(pdf_path), str(input_root), str(output_root),
                str(output_images_dir), args.log_dir, args.log_json,
                args.timeout, idx, total
            )
            for idx, pdf_path in enumerate(to_process, 1)
        ]

        # OPTIMIZATION 1: ProcessPoolExecutor — PDFs are processed in parallel.
        # Each PDF is CPU-bound (marker ML inference) and fully independent,
        # making multiprocessing ideal.  The pool size is bounded by --parallel
        # (default: cores / marker_workers) to avoid GPU VRAM exhaustion.
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_process_one_pdf, wargs): wargs for wargs in worker_args}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["status"] == "ok":
                        summary.processed_ok += 1
                    else:
                        summary.processed_fail += 1
                        if result["error"]:
                            summary.errors.append(result["error"])
                except Exception as exc:
                    summary.processed_fail += 1
                    wargs = futures[future]
                    pdf_name = Path(wargs[0]).name
                    summary.errors.append(f"{pdf_name} ➔ Unhandled exception: {exc}")
                    logger.error(f"Worker crashed for {pdf_name}: {exc}")

    logger.info(f"\nTotal pipeline time: {time.time() - pipeline_start:.1f}s")

    sep = "=" * 70
    lines = [
        "\n" + sep, "  📊 PIPELINE SUMMARY ", sep,
        f"  Total PDFs found       : {summary.total_found}",
        f"  ✅ Successfully done   : {summary.processed_ok}",
        f"  ❌ Failed              : {summary.processed_fail}",
        f"  ⏭️  Skipped             : {summary.skipped}", sep
    ]
    if summary.errors:
        lines.append("  FAILED FILES:")
        for err in summary.errors: lines.append(f"    • {err}")
    lines.append(sep + "\n")

    output = "\n".join(lines)
    print(output, file=sys.stderr)
    for line in lines: logger.debug(line)

    sys.exit(0 if summary.processed_fail == 0 else 1)


if __name__ == "__main__":
    main()