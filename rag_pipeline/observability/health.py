"""
Data-health rules: "where is my data broken, and what do I do about it?"

This module answers the question a newcomer actually has when they look at a
RAG pipeline's output: the run said SUCCESS, so why are the answers bad?

It reads the *.chunks.json files the pipeline writes and checks them against
rules that each map to a real, known retrieval failure mode. Every finding
carries three things:

  * WHAT was found (count, percentage, real example chunk_ids),
  * WHY it matters for search quality, in plain language,
  * the FIX -- which knob in settings.yaml or which file to look at.

It is read-only, stdlib-only, and can be run on its own without Ollama, a GPU
or any model at all:

    python -m rag_pipeline.observability.health --chunks-dir ./chunks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import i18n
from .i18n import Bi

RED, YELLOW, GREEN = "\U0001F534", "\U0001F7E1", "\U0001F7E2"

_PLACEHOLDER_RE = re.compile(r"\[IMAGE_DESC_PLACEHOLDER_\d+\]")
_HTML_TAG_RE = re.compile(r"</?(table|td|tr|th|thead|tbody|div|span|p)\b[^>]*>", re.I)


@dataclass
class Finding:
    id: str
    icon: str
    title: Bi
    why: Bi
    fix: Bi
    count: int = 0
    total: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def share(self) -> str:
        return i18n.pct(self.count, self.total)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "severity": self.icon, "count": self.count,
            "total": self.total, "share": self.share,
            "title": self.title[0], "examples": self.examples[:5],
        }


# ───────────────────────────────────────────────────────────────────────────────
# Small accessors -- the pipeline's chunk dicts, tolerant of missing keys.
# ───────────────────────────────────────────────────────────────────────────────

def _meta(c: dict) -> dict:
    return c.get("metadata") or {}


def _text(c: dict) -> str:
    """The human-facing text of a chunk (what a citation would show)."""
    return (c.get("normalized_text") or c.get("content") or "").strip()


def _cid(c: dict) -> str:
    return str(c.get("chunk_id") or "(no chunk_id)")


def _chunk_type(c: dict) -> str:
    m = _meta(c)
    return str(m.get("chunk_type") or m.get("type") or "unknown")


def discover_chunk_files(chunks_dir: Path) -> List[Path]:
    return sorted(Path(chunks_dir).rglob("*.chunks.json"))


def load_chunks(chunks_dir: Path) -> List[dict]:
    """Load every chunk from every *.chunks.json under chunks_dir."""
    out: List[dict] = []
    for p in discover_chunk_files(chunks_dir):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for c in data.get("chunks", []):
            c.setdefault("_file", str(p))
            out.append(c)
    return out


# ───────────────────────────────────────────────────────────────────────────────
# The rules
# ───────────────────────────────────────────────────────────────────────────────

def _severity(count: int, total: int, red_at: float, always_red: bool = False) -> str:
    """Escalate by how much of the corpus is affected, not by raw count."""
    if count == 0:
        return GREEN
    if always_red:
        return RED
    return RED if total and (count / total) >= red_at else YELLOW


def analyze_chunks(
    chunks: Sequence[dict],
    *,
    min_chunk_chars: int = 80,
    max_chunk_chars: int = 2000,
    expect_embed_text: bool = True,
) -> List[Finding]:
    """
    Run every rule and return only the rules that actually fired, worst first.

    `min_chunk_chars` / `max_chunk_chars` should be passed from settings.yaml
    (semantic_chunker.*) so the report judges your data by YOUR thresholds,
    not by hardcoded ones.
    """
    found: List[Finding] = []
    total = len(chunks)
    if total == 0:
        return [Finding(
            id="no_chunks", icon=RED,
            title=("No chunks found at all", "هیچ قطعه‌ای پیدا نشد"),
            why=("There is nothing to search. Either the pipeline never ran, or it found no .md files in paths.input_dir.",
                 "چیزی برای جستجو وجود ندارد. یا پایپ‌لاین اجرا نشده یا هیچ فایل md در input_dir نبوده."),
            fix=("Put your OCR .md files in the folder named by paths.input_dir in settings.yaml, then run: python rag_pipeline/run_pipeline.py",
                 "فایل‌های md حاصل OCR را در مسیر input_dir در settings.yaml قرار دهید و run_pipeline.py را اجرا کنید."),
            count=0, total=0,
        )]

    def add(fid: str, hits: List[dict], title: Bi, why: Bi, fix: Bi,
            red_at: float = 0.25, always_red: bool = False) -> None:
        if not hits:
            return
        found.append(Finding(
            id=fid, icon=_severity(len(hits), total, red_at, always_red),
            title=title, why=why, fix=fix,
            count=len(hits), total=total,
            examples=[_cid(c) for c in hits[:5]],
        ))

    # 1. Empty text -- an unfindable, uncitable chunk.
    add("empty_text", [c for c in chunks if not _text(c)],
        ("Chunks with no readable text", "قطعات بدون متن قابل خواندن"),
        ("These chunks can be returned by search but show nothing to the reader, and they waste a slot in the LLM's context.",
         "این قطعات ممکن است در نتایج بیایند ولی چیزی نشان نمی‌دهند و جای خالی در زمینه مدل می‌گیرند."),
        ("Check the source document for blank sections; consider raising structure_splitter.min_section_chars.",
         "سند منبع را برای بخش‌های خالی ببینید؛ min_section_chars را بالاتر ببرید."),
        always_red=True)

    # 2. Too short to carry meaning.
    short = [c for c in chunks if 0 < len(_text(c)) < min_chunk_chars]
    add("too_short", short,
        (f"Chunks shorter than {min_chunk_chars} characters", f"قطعات کوتاه‌تر از {min_chunk_chars} کاراکتر"),
        ("Very short chunks embed poorly: the vector is dominated by noise, so they either never match or match everything.",
         "قطعات خیلی کوتاه تعبیه بدی دارند؛ یا هرگز مطابقت نمی‌کنند یا با همه چیز مطابقت می‌کنند."),
        ("Raise semantic_chunker.min_chunk_chars / min_chunk_tokens in settings.yaml so small fragments get merged instead of stored.",
         "در settings.yaml مقدار min_chunk_chars یا min_chunk_tokens را بالاتر ببرید تا تکه‌های کوچک ادغام شوند."),
        red_at=0.30)

    # 3. Too long -- will be truncated at answer time.
    add("too_long", [c for c in chunks if len(_text(c)) > max_chunk_chars],
        (f"Chunks longer than {max_chunk_chars} characters", f"قطعات بلندتر از {max_chunk_chars} کاراکتر"),
        ("Long chunks get cut when the context is built (context_builder.max_chars_per_chunk), so the part the answer needed may be silently dropped.",
         "قطعات بلند هنگام ساخت زمینه بریده می‌شوند و ممکن است بخش مورد نیاز پاسخ حذف شود."),
        ("Lower semantic_chunker.max_chunk_chars, or raise context_builder.max_chars_per_chunk so less is cut.",
         "max_chunk_chars را کمتر کنید یا max_chars_per_chunk را بالاتر ببرید."),
        red_at=0.30)

    # 4. Unknown language -- directly breaks the dense-side language filter.
    unknown_lang = [c for c in chunks
                    if str(_meta(c).get("language", "unknown")).lower() in ("", "unknown", "none")]
    add("unknown_language", unknown_lang,
        ("Chunks with language = unknown", "قطعات با زبان نامشخص"),
        ("retrieval.hybrid.language_filter filters vector search by this exact field. A chunk marked unknown can be excluded from every dense search result, so only keyword search can ever find it.",
         "فیلتر زبان در جستجوی برداری روی همین فیلد اعمال می‌شود؛ قطعه نامشخص ممکن است از همه نتایج برداری حذف شود."),
        ("Either set retrieval.hybrid.language_filter: false, or accept it: the detector in preprocessors/language_detector.py only distinguishes English vs Spanish and returns unknown for short text.",
         "یا language_filter را false کنید، یا بپذیریدؚ تشخیص‌دهنده فقط انگلیسی/اسپانیایی را تشخیص می‌دهد."),
        red_at=0.20)

    # 5. Missing breadcrumb -- weakens BM25 and makes citations vague.
    add("missing_breadcrumb", [c for c in chunks if not str(_meta(c).get("breadcrumb", "")).strip()],
        ("Chunks with no heading trail (breadcrumb)", "قطعات بدون مسیر عنوان (بریدکرامب)"),
        ("The breadcrumb is part of the text BM25 indexes and part of what the reader sees in a citation. Without it, keyword search is weaker and sources look anonymous.",
         "بریدکرامب هم در ایندکس BM25 و هم در ارجاع نشان داده می‌شود؛ بدون آن جستجوی کلیدواژه‌ای ضعیف‌تر است."),
        ("Usually means the OCR document has no #/##/### headings. Fix the OCR output, or accept weaker keyword search for those files.",
         "معمولاً یعنی سند OCR عنوان ندارد. خروجی OCR را اصلاح کنید."),
        red_at=0.50)

    # 6. Duplicate chunk_ids -- silent data loss at ingestion time.
    id_counts = Counter(_cid(c) for c in chunks)
    dup_ids = {k for k, v in id_counts.items() if v > 1}
    if dup_ids:
        found.append(Finding(
            id="duplicate_chunk_id", icon=RED,
            title=("Duplicate chunk_id values", "شناسه‌های تکراری chunk_id"),
            why=("Ingestion upserts by chunk_id, so duplicates overwrite each other: you will silently store fewer chunks than you produced.",
                 "بارگذاری بر اساس chunk_id انجام می‌شود؛ شناسه تکراری روی قبلی می‌نویسد و بی‌صدا داده کم می‌شود."),
            fix=("Look at enrichers/metadata_enricher.py: chunk_id must be unique per chunk. Two identical sections in one document can collide.",
                 "در metadata_enricher.py بررسی کنید؛ chunk_id باید برای هر قطعه یکتا باشد."),
            count=sum(id_counts[k] for k in dup_ids), total=total,
            examples=sorted(dup_ids)[:5],
        ))

    # 7. Exactly duplicated text -- wastes context slots on the same content.
    seen: Dict[str, str] = {}
    dup_text = []
    for c in chunks:
        t = _text(c)
        if not t:
            continue
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if h in seen:
            dup_text.append(c)
        else:
            seen[h] = _cid(c)
    add("duplicate_text", dup_text,
        ("Chunks whose text is byte-identical to another chunk", "قطعات با متن کاملاً تکراری"),
        ("Near-duplicates crowd each other out of the top results, so the LLM sees the same fact three times instead of three different facts.",
         "قطعات تکراری جای یکدیگر را در نتایج برتر پر می‌کنند و مدل یک واقعیت را چند بار می‌بیند."),
        ("Expected if the same boilerplate (headers/footers) repeats across pages. If it is your real content, de-duplicate before ingestion.",
         "اگر سربرگ/پانویس تکراری است طبیعی است؛ وگرنه پیش از بارگذاری حذف تکرار کنید."),
        red_at=0.20)

    # 8. Unresolved image placeholders -- a broken handoff between two stages.
    add("unresolved_image_placeholder",
        [c for c in chunks if _PLACEHOLDER_RE.search(_text(c))],
        ("Chunks still containing [IMAGE_DESC_PLACEHOLDER_n]", "قطعاتی که هنوز نشانه تصویر دارند"),
        ("The image description was never substituted back, so a placeholder token is being embedded and shown to users instead of the real description.",
         "توضیح تصویر جایگزین نشده و خود نشانه تعبیه و نمایش داده می‌شود."),
        ("Check that the sidecar <name>_metadata.json exists next to the .md file and that its image numbering matches. See stages/resolve_images.py.",
         "مطمئن شوید فایل <name>_metadata.json کنار فایل md وجود دارد و شماره‌گذاری تصاویر مطابقت دارد."),
        always_red=True)

    # 9. Raw HTML leaking into embedded text.
    add("raw_html", [c for c in chunks if _HTML_TAG_RE.search(_text(c))],
        ("Chunks that still contain raw HTML tags", "قطعاتی که هنوز تگ HTML خام دارند"),
        ("Table normalization fell back to raw HTML for these. Embedding models score <td> markup poorly, so these chunks are much harder to retrieve.",
         "نرمال‌سازی جدول برای این‌ها به HTML خام بازگشته است؛ بازیابی این قطعات سخت‌تر می‌شود."),
        ("Install the parser deps (beautifulsoup4, lxml) and re-run; see normalizers/table_normalizer.py and its fallback_to_raw_on_error setting.",
         "پکیج‌های beautifulsoup4 و lxml را نصب کنید و دوباره اجرا کنید."),
        red_at=0.15)

    # 10. Missing embed_text -- the vector will be built from nothing.
    if expect_embed_text:
        add("missing_embed_text", [c for c in chunks if not str(c.get("embed_text") or "").strip()],
            ("Chunks with no embed_text", "قطعات بدون embed_text"),
            ("embed_text is the exact string sent to the embedding model. Empty means the vector is meaningless, and meaning-based search will never find this chunk.",
             "embed_text همان متنی است که به مدل تعبیه می‌رود؛ خالی بودن یعنی وکتور بی‌معنا."),
            ("Check embed_text.enabled in settings.yaml and enrichers/embed_text_builder.py.",
             "تنطیم embed_text.enabled و فایل embed_text_builder.py را بررسی کنید."),
            always_red=True)

    # 11. Missing page numbers -- citations cannot point anywhere verifiable.
    add("missing_page_numbers",
        [c for c in chunks if not _meta(c).get("page_numbers")],
        ("Chunks with no page numbers", "قطعات بدون شماره صفحه"),
        ("Citations can still name the file, but nobody can verify the claim against a page of the original PDF.",
         "ارجاع فقط نام فایل را می‌دهد و امکان راستی‌آزمایی روی صفحه اصل PDF نیست."),
        ("Page numbers come from inline OCR page markers. If your OCR tool does not emit them, this stays empty -- see metadata_enricher.py's marker registry.",
         "شماره صفحه از نشانه‌های درون‌متنی OCR می‌آید؛ اگر ابزار OCR شما آن را نمی‌نویسد خالی می‌ماند."),
        red_at=0.90)

    # 12. Everything came from one document -- usually a glob/path mistake.
    files = Counter(str(_meta(c).get("source_file") or c.get("_file", "?")) for c in chunks)
    if len(files) == 1 and total > 20:
        found.append(Finding(
            id="single_source_file", icon=YELLOW,
            title=("All chunks come from a single document", "همه قطعات از یک سند هستند"),
            why=("If you expected several documents, the others were never found or never chunked.",
                 "اگر انتطار چند سند داشتید، بقیه پیدا یا پردازش نشده‌اند."),
            fix=("Check chunking.md_glob and paths.input_dir in settings.yaml, then re-run and read step 1 of the report.",
                 "تنطیمات md_glob و input_dir را ببینید و دوباره اجرا کنید."),
            count=total, total=total, examples=list(files)[:1],
        ))

    order = {RED: 0, YELLOW: 1, GREEN: 2}
    found.sort(key=lambda f: (order.get(f.icon, 3), -f.count))
    return found


def corpus_stats(chunks: Sequence[dict]) -> List[List[Any]]:
    """Plain descriptive numbers -- context for judging the findings."""
    if not chunks:
        return [["chunks", 0]]
    lengths = sorted(len(_text(c)) for c in chunks)
    n = len(lengths)
    types = Counter(_chunk_type(c) for c in chunks)
    langs = Counter(str(_meta(c).get("language", "unknown")) for c in chunks)
    files = Counter(str(_meta(c).get("source_file") or c.get("_file", "?")) for c in chunks)
    return [
        ["Chunks", n],
        ["Source documents", len(files)],
        ["Shortest chunk (chars)", lengths[0]],
        ["Median chunk (chars)", lengths[n // 2]],
        ["Longest chunk (chars)", lengths[-1]],
        ["Average chunk (chars)", round(sum(lengths) / n, 1)],
        ["Types", ", ".join(f"{k}={v}" for k, v in types.most_common())],
        ["Languages", ", ".join(f"{k}={v}" for k, v in langs.most_common())],
    ]


def findings_md(findings: Sequence[Finding], lang: str = i18n.BOTH,
                stats: Optional[List[List[Any]]] = None) -> str:
    """Render findings as the DATA_HEALTH.md document."""
    out = ["# Data health — \u0633\u0644\u0627\u0645\u062a \u062f\u0627\u062f\u0647\n"]
    out.append(i18n.para((
        "Each item below is a real, measurable property of YOUR data, not a generic warning. "
        "\U0001F534 = fix before trusting answers. \U0001F7E1 = worth knowing. "
        "Nothing here changes your data; this report only looks at it.",
        "هر مورد زیر یک ویژگی واقعی و قابل اندازه‌گیری داده‌های شماست، نه هشدار کلی. "
        "\U0001F534 یعنی پیش از اعتماد به پاسخ‌ها اصلاح کنید. \U0001F7E1 یعنی دانستنش مفید است. این گزارش داده را تغییر نمی‌دهد.",
    ), lang))

    if stats:
        out.append(i18n.heading(("Corpus at a glance", "نمای کلی داده‌ها"), 2, lang))
        out.append(i18n.table(["Measure", "Value"], stats))

    if not findings:
        out.append("\n" + i18n.heading(("Result: no problems found", "نتیجه: مشکلی پیدا نشد"), 2, lang))
        out.append(i18n.para(("Every rule passed on this data.", "تمام قواعد روی این داده‌ها قبول شدند."), lang))
        return "\n".join(out)

    out.append("\n" + i18n.heading(("Summary", "خلاصه"), 2, lang))
    out.append(i18n.table(
        ["", "Finding", "Affected", "Share", "What it is"],
        [[f.icon, f.id, f.count, f.share, f.title[0]] for f in findings],
    ))

    for f in findings:
        out.append("\n" + i18n.heading((f"{f.icon} {f.title[0]}", f.title[1]), 2, lang))
        out.append(i18n.table(
            ["Affected chunks", "Share of corpus", "Rule id"],
            [[f"{f.count} of {f.total}", f.share, f.id]],
        ))
        out.append("\n**Why this matters / \u0686\u0631\u0627 \u0645\u0647\u0645 \u0627\u0633\u062a**\n")
        out.append(i18n.para(f.why, lang))
        out.append("**What to do / \u0686\u0647 \u06a9\u0627\u0631 \u06a9\u0646\u06cc\u062f**\n")
        out.append(i18n.para(f.fix, lang))
        if f.examples:
            out.append("**Examples / \u0646\u0645\u0648\u0646\u0647\u200c\u0647\u0627**\n")
            out.append(i18n.fence("\n".join(f.examples), "text"))
    return "\n".join(out)


def console_summary(findings: Sequence[Finding]) -> str:
    if not findings:
        return f"  {GREEN} data health: no problems found"
    lines = ["  data health:"]
    for f in findings:
        lines.append(f"    {f.icon} {f.id:<30} {f.count:>6} of {f.total} ({f.share})  {f.title[0]}")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────────────
# Standalone CLI -- needs no models, no network, no GPU.
# ───────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the chunks on disk for data problems that hurt retrieval quality.")
    ap.add_argument("--chunks-dir", default=None, help="Default: paths.chunks_dir from settings.yaml")
    ap.add_argument("--out", default=None, help="Write DATA_HEALTH.md to this folder as well as printing.")
    ap.add_argument("--lang", default=i18n.BOTH, choices=[i18n.EN, i18n.FA, i18n.BOTH])
    args = ap.parse_args(argv)

    chunks_dir = args.chunks_dir
    min_chars, max_chars, expect_embed = 80, 2000, True
    try:
        from ..settings import load_settings
        s = load_settings()
        chunks_dir = chunks_dir or s.paths.chunks_dir
        # Judge the data by the user's OWN thresholds, not hardcoded ones.
        min_chars = getattr(s.pipeline.semantic, "min_chunk_chars", min_chars)
        max_chars = getattr(s.pipeline.semantic, "max_chunk_chars", max_chars)
        expect_embed = getattr(s.pipeline.embed_text, "enabled", True)
    except Exception as exc:
        print(f"  (settings.yaml not used: {exc})")
        chunks_dir = chunks_dir or "./chunks"

    chunks = load_chunks(Path(chunks_dir))
    findings = analyze_chunks(chunks, min_chunk_chars=min_chars,
                             max_chunk_chars=max_chars, expect_embed_text=expect_embed)
    stats = corpus_stats(chunks)

    print(f"\nChecked {len(chunks)} chunk(s) from {chunks_dir}")
    for row in stats:
        print(f"  {row[0]:<24} {row[1]}")
    print(console_summary(findings))

    if args.out:
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "DATA_HEALTH.md"
        path.write_text(findings_md(findings, lang=args.lang, stats=stats), encoding="utf-8")
        print(f"\n  \U0001F4C4 wrote {path}")

    return 1 if any(f.icon == RED for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
