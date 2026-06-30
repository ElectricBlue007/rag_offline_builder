"""
test_dry_run.py — Validates preprocessor + structure splitter against the
real sample markdown files WITHOUT loading the embedding model.

Run with:
    python test_dry_run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Chunker.config import PreprocessorConfig, StructureSplitterConfig
from Chunker.preprocessors.md_preprocessor import MarkdownPreprocessor
from Chunker.preprocessors.language_detector import detect_language
from Chunker.splitters.structure_splitter import StructureSplitter


SAMPLE_FILES = [
    Path(__file__).parent / "tests" / "fixtures" / "Binder2.md",
    Path(__file__).parent / "tests" / "fixtures" / "Testing.md",
]


def hr(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def test_file(md_path: Path):
    hr(f"FILE: {md_path.name}")
    raw = md_path.read_text(encoding="utf-8")
    print(f"  Raw length : {len(raw):,} chars")

    # ── Pre-process ─────────────────────────────────────────────────
    prep   = MarkdownPreprocessor(PreprocessorConfig())
    doc    = prep.process(raw, source_path=str(md_path))

    print(f"  Clean length: {len(doc.cleaned_text):,} chars")
    print(f"  Image alts extracted: {len(doc.image_alts)}")
    for i, alt in enumerate(doc.image_alts):
        preview = alt.alt_text[:80].replace('\n', ' ')
        print(f"    [{i}] pos={alt.position}  \"{preview}…\"")

    # ── Structure split ─────────────────────────────────────────────
    splitter  = StructureSplitter(StructureSplitterConfig())
    sections  = splitter.split(doc.cleaned_text, source_path=str(md_path))

    text_secs  = [s for s in sections if s.section_type == "text"]
    table_secs = [s for s in sections if s.section_type == "table"]
    img_secs   = [s for s in sections if s.section_type == "image_description"]

    print(f"\n  Structural sections : {len(sections)} total")
    print(f"    text             : {len(text_secs)}")
    print(f"    table            : {len(table_secs)}")
    print(f"    image_description: {len(img_secs)}")

    print("\n  Section breakdown:")
    for i, sec in enumerate(sections):
        lang    = detect_language(sec.content)
        preview = sec.content[:60].replace('\n', ' ')
        print(
            f"    [{i:02d}] type={sec.section_type:<20} "
            f"lang={lang}  "
            f"chars={len(sec.content):>5}  "
            f"breadcrumb=\"{sec.breadcrumb[:50]}\""
        )
        print(f"          preview: \"{preview}…\"")

    print(f"\n  ✅  {md_path.name} — OK")


def main():
    print("\n🔍  DRY-RUN TEST  (no embedding model loaded)\n")
    all_ok = True
    for path in SAMPLE_FILES:
        if not path.exists():
            print(f"  ⚠️  File not found: {path}")
            all_ok = False
            continue
        try:
            test_file(path)
        except Exception as exc:
            print(f"  ❌  FAILED on {path.name}: {exc}")
            import traceback; traceback.print_exc()
            all_ok = False

    hr()
    if all_ok:
        print("  ALL TESTS PASSED ✅")
    else:
        print("  SOME TESTS FAILED ❌")
    print()


if __name__ == "__main__":
    main()
