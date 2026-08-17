"""
RunReport -- the "what just happened, and where is my data broken" recorder.

Design goals (in priority order)
--------------------------------
1. A person who has never seen this project should be able to read the
   generated Markdown and understand, step by step, what the code did to
   their documents -- including WHICH SOURCE FILE implements each step, so
   the report doubles as a guided tour of the codebase.
2. It must be impossible for reporting to break a pipeline run. Every hook
   is wrapped so a reporting bug degrades to "no report", never to a failed
   ingestion.
3. Zero new dependencies (stdlib only) and zero behaviour change: this class
   observes, it never transforms.

Usage
-----
    report = RunReport(out_dir="reports", title="Chunking pipeline")
    with report.step("split_structure", items_in=1) as s:
        sections = splitter.split(text)
        s.done(len(sections))
        s.note(("2 tables kept whole", "۲ جدول دست‌نخورده ماند"))
        s.sample(("First section", "بخش اول"), sections[0].text)
    path = report.write()
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import i18n
from .i18n import Bi

# ───────────────────────────────────────────────────────────────────────────────
# The teaching layer: what each pipeline step is FOR, in plain language.
#
# `source` points at the file a newcomer should open to see that step's real
# code. `watch` names the data problem that step is most likely to cause --
# which is exactly what the reader is hunting for when something looks wrong.
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepDoc:
    title: Bi
    what: Bi
    watch: Bi
    source: str
    unit_in: Bi = ("items in", "ورودی")
    unit_out: Bi = ("items out", "خروجی")


STEP_DOCS: Dict[str, StepDoc] = {
    "load_markdown": StepDoc(
        title=("Load Markdown", "خواندن فایل مارک‌داون"),
        what=("Reads one OCR .md file from disk, exactly as it is, plus its sidecar *_metadata.json if present. Nothing is changed yet.",
              "یک فایل md حاصل OCR را بی‌هیچ تغییری از دیسک می‌خواند، همراه فایل متادیتای کناری اگر وجود داشته باشد."),
        watch=("An empty or tiny file here means the OCR step upstream failed, not this pipeline.",
               "اگر فایل خالی یا بسیار کوتاه باشد، مشکل از مرحله OCR است، نه این پایپ‌لاین."),
        source="stages/load_markdown.py",
        unit_in=("files", "فایل"), unit_out=("characters", "کاراکتر"),
    ),
    "preprocess_markdown": StepDoc(
        title=("Clean the Markdown", "پاکسازی مارک‌داون"),
        what=("Normalises OCR noise and pulls image descriptions out of the body text, leaving a numbered placeholder such as [IMAGE_DESC_PLACEHOLDER_3] where each picture was.",
              "نویز OCR را یکدست می‌کند و توضیح تصاویر را از متن جدا می‌کند و جای هر تصویر یک نشانه مانند [IMAGE_DESC_PLACEHOLDER_3] می‌گذارد."),
        watch=("If placeholders survive into the final chunks, this step and resolve_images disagree -- see DATA_HEALTH.md.",
               "اگر این نشانه‌ها در خروجی نهایی بمانند، این مرحله با resolve_images هماهنگ نبوده است."),
        source="preprocessors/md_preprocessor.py",
        unit_in=("characters", "کاراکتر"), unit_out=("characters", "کاراکتر"),
    ),
    "split_structure": StepDoc(
        title=("Split by headings", "تقسیم بر اساس عنوان‌ها"),
        what=("Cuts the document at its #, ## and ### headings into sections, and keeps every table as one atomic section so a table is never cut in half.",
              "سند را از محل عنوان‌های #، ## و ### به بخش‌ها تقسیم می‌کند و هر جدول را یکجا نگه می‌دارد تا جدول نیمه نشود."),
        watch=("1 section out of a long document means the OCR produced no headings -- retrieval quality will suffer.",
               "اگر از یک سند بلند فقط یک بخش دربیاید، یعنی OCR عنوانی تولید نکرده و کیفیت بازیابی افت می‌کند."),
        source="splitters/structure_splitter.py",
        unit_in=("characters", "کاراکتر"), unit_out=("sections", "بخش"),
    ),
    "resolve_images": StepDoc(
        title=("Put image descriptions back", "بازگرداندن توضیح تصاویر"),
        what=("Replaces each [IMAGE_DESC_PLACEHOLDER_n] with the real screenshot description, so a picture becomes searchable text.",
              "هر [IMAGE_DESC_PLACEHOLDER_n] را با توضیح واقعی همان تصویر جایگزین می‌کند تا تصویر قابل جستجو شود."),
        watch=("Unresolved placeholders mean the sidecar *_metadata.json is missing or its numbering does not match the document.",
               "ماندن نشانه‌ها یعنی فایل متادیتا نیست یا شماره‌گذاری‌اش با سند نمی‌خواند."),
        source="stages/resolve_images.py",
        unit_in=("sections", "بخش"), unit_out=("sections", "بخش"),
    ),
    "split_semantic": StepDoc(
        title=("Split by meaning", "تقسیم معنایی"),
        what=("Splits any section that is still too long at the points where the topic actually changes, measured with the embedding model. This is the step that needs Ollama.",
              "بخش‌های هنوز بلند را دقیقاً در نقاطی که موضوع عوض می‌شود تقسیم می‌کند. این مرحله به Ollama نیاز دارد."),
        watch=("Output equal to input means every section was already small enough (normal), or the model was unreachable and it fell back (check warnings).",
               "برابری ورودی و خروجی عادی است، مگر اینکه مدل در دسترس نبوده باشد (هشدارها را ببینید)."),
        source="splitters/semantic_chunker.py",
        unit_in=("sections", "بخش"), unit_out=("chunks", "قطعه"),
    ),
    "refine_chunks": StepDoc(
        title=("LLM tidy-up (optional)", "بازنگری با LLM (اختیاری)"),
        what=("Asks a local LLM to merge or re-cut awkward chunk boundaries. Disabled by default (llm_chunk_refiner.enabled: false), in which case it is a true no-op.",
              "از یک LLM محلی می‌خواهد مرزهای نامناسب قطعات را اصلاح کند. پیش‌فرض خاموش است."),
        watch=("Enabling this costs one LLM call per file and can change chunk_ids, which forces a re-ingest.",
               "روشن کردنش به ازای هر فایل یک فراخوانی LLM هزینه دارد و chunk_id را عوض می‌کند."),
        source="splitters/llm_chunk_refiner.py",
        unit_in=("chunks", "قطعه"), unit_out=("chunks", "قطعه"),
    ),
    "normalize_tables": StepDoc(
        title=("Turn tables into sentences", "تبدیل جدول به جمله"),
        what=("Rewrites HTML tables as readable prose, because embedding models score plain sentences far better than <td> soup.",
              "جداول HTML را به متن روان تبدیل می‌کند، چون مدل‌های تعبیه با جمله بهتر کار می‌کنند تا با تگ‌های HTML."),
        watch=("If < or <td> still appears in a final chunk, this step silently fell back to raw HTML for that table.",
               "اگر در قطعه نهایی هنوز تگ HTML هست، این مرحله برای آن جدول به حالت خام بازگشته است."),
        source="normalizers/table_normalizer.py",
        unit_in=("chunks", "قطعه"), unit_out=("chunks", "قطعه"),
    ),
    "enrich_context": StepDoc(
        title=("Add surrounding context", "افزودن زمینه"),
        what=("Prefixes each chunk with where it came from (document name and heading trail) so an isolated paragraph still makes sense on its own.",
              "به ابتدای هر قطعه می‌نویسد از کجا آمده (نام سند و مسیر عنوان‌ها) تا قطعه جداافتاده هم معنا داشته باشد."),
        watch=("This text is what keyword (BM25) search indexes, so an empty breadcrumb weakens keyword search.",
               "همین متن در جستجوی کلیدواژه‌ای (BM25) ایندکس می‌شود؛ بریدکرامب خالی آن را ضعیف می‌کند."),
        source="enrichers/contextual_enricher.py",
        unit_in=("chunks", "قطعه"), unit_out=("chunks", "قطعه"),
    ),
    "build_embed_text": StepDoc(
        title=("Build the text to embed", "ساخت متن تعبیه"),
        what=("Assembles the exact string that will be turned into a vector, including the model's instruction prefix. Stored separately from the text shown to humans.",
              "دقیقاً رشته‌ای را می‌سازد که به وکتور تبدیل می‌شود، همراه پیشوند دستوری مدل. جدا از متنی که به کاربر نشان داده می‌شود."),
        watch=("An empty embed_text produces a meaningless vector that can still be returned by search.",
               "embed_text خالی وکتور بی‌معنا می‌سازد که باز هم ممکن است در نتایج بیاید."),
        source="enrichers/embed_text_builder.py",
        unit_in=("chunks", "قطعه"), unit_out=("chunks", "قطعه"),
    ),
    "enrich_metadata": StepDoc(
        title=("Attach metadata", "افزودن متادیتا"),
        what=("Gives every chunk its chunk_id, source file, heading trail, detected language, type (text/table/image) and page numbers. This is what citations are built from.",
              "به هر قطعه chunk_id، فایل منبع، مسیر عنوان‌ها، زبان، نوع و شماره صفحه می‌دهد. ارجاعات از اینجا ساخته می‌شوند."),
        watch=("language: unknown here becomes a retrieval bug later, because dense search filters on this field.",
               "مقدار unknown برای زبان، بعداً به ایراد بازیابی تبدیل می‌شود، چون جستجوی برداری روی این فیلد فیلتر می‌کند."),
        source="enrichers/metadata_enricher.py",
        unit_in=("chunks", "قطعه"), unit_out=("chunks", "قطعه"),
    ),
    "write_chunks": StepDoc(
        title=("Write chunks to disk", "نوشتن قطعات روی دیسک"),
        what=("Saves one <name>.chunks.json per input document. These files are the handover point to ingestion; nothing before this is persisted.",
              "برای هر سند یک فایل <name>.chunks.json می‌نویسد. این فایل‌ها نقطه تحویل به مرحله بارگذاری هستند."),
        watch=("If a file is missing here, ingestion cannot see that document at all.",
               "اگر فایلی اینجا نباشد، مرحله بارگذاری آن سند را اصلاً نمی‌بیند."),
        source="stages/write_chunks.py",
        unit_in=("chunks", "قطعه"), unit_out=("files", "فایل"),
    ),
    # Steps outside the chunking pipeline (ingestion / retrieval CLIs).
    "embed_chunks": StepDoc(
        title=("Generate vectors", "تولید وکتور"),
        what=("Sends every chunk's embed_text to Ollama and gets back one vector per chunk.",
              "embed_text هر قطعه را به Ollama می‌فرستد و برای هر قطعه یک وکتور می‌گیرد."),
        watch=("vectors out < chunks in means some chunks will never be findable by meaning-based search.",
               "اگر تعداد وکتورها کمتر از قطعات باشد، بعضی قطعات هرگز در جستجوی معنایی پیدا نمی‌شوند."),
        source="embedding/embed_generator.py",
        unit_in=("chunks", "قطعه"), unit_out=("vectors", "وکتور"),
    ),
    "ingest_chroma": StepDoc(
        title=("Store in the vector database", "ذخیره در پایگاه وکتور"),
        what=("Upserts chunks and vectors into Chroma. Upsert means existing ids are overwritten and ids no longer present are LEFT BEHIND.",
              "قطعات و وکتورها را در Chroma درج/به‌روز می‌کند. شناسه‌های حذف‌شده پاک نمی‌شوند."),
        watch=("Collection count larger than the number of chunks on disk means stale ghost chunks are still being served.",
               "اگر تعداد رکوردهای Chroma بیشتر از قطعات روی دیسک باشد، قطعات مرده هنوز سرو می‌شوند."),
        source="storage/chroma_ingestion.py",
        unit_in=("chunks", "قطعه"), unit_out=("records", "رکورد"),
    ),
    "build_bm25": StepDoc(
        title=("Build the keyword index", "ساخت ایندکس کلیدواژه"),
        what=("Builds the BM25 keyword index from scratch, from the chunks on disk. Unlike Chroma, this is a full rebuild every time.",
              "ایندکس کلیدواژه‌ای BM25 را از صفر و از قطعات روی دیسک می‌سازد — برخلاف Chroma، هر بار کامل بازسازی می‌شود."),
        watch=("Because this rebuilds and Chroma does not, the two indexes drift apart after you delete a document.",
               "چون این کامل بازسازی می‌شود و Chroma نه، پس از حذف یک سند این دو ایندکس ناهماهنگ می‌شوند."),
        source="storage/bm25_index.py",
        unit_in=("chunks", "قطعه"), unit_out=("indexed docs", "سند ایندکس‌شده"),
    ),
}

_UNKNOWN = StepDoc(
    title=("Step", "مرحله"),
    what=("No description registered for this step name.", "توضیحی برای این مرحله ثبت نشده است."),
    watch=("—", "—"),
    source="(unknown)",
)


def doc_for(step_name: str) -> StepDoc:
    return STEP_DOCS.get(step_name, _UNKNOWN)


# ───────────────────────────────────────────────────────────────────────────────
# Recording
# ───────────────────────────────────────────────────────────────────────────────

OK, FAILED, SKIPPED = "ok", "failed", "skipped"


@dataclass
class Step:
    order: int
    name: str
    items_in: Optional[float] = None
    items_out: Optional[float] = None
    seconds: float = 0.0
    status: str = OK
    error: Optional[str] = None
    scope: str = ""                       # e.g. the file being processed
    notes: List[Bi] = field(default_factory=list)
    warnings: List[Bi] = field(default_factory=list)
    samples: List[Tuple[Bi, str]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> Optional[float]:
        if self.items_in is None or self.items_out is None:
            return None
        return self.items_out - self.items_in

    def to_dict(self) -> dict:
        return {
            "order": self.order, "name": self.name, "scope": self.scope,
            "items_in": self.items_in, "items_out": self.items_out,
            "seconds": round(self.seconds, 4), "status": self.status,
            "error": self.error,
            "notes": [n[0] for n in self.notes],
            "warnings": [w[0] for w in self.warnings],
            "details": self.details,
        }


class StepHandle:
    """Handed to the caller inside `with report.step(...) as s:`."""

    def __init__(self, step: Step, sample_chars: int):
        self._step = step
        self._sample_chars = sample_chars

    def done(self, items_out: Optional[float]) -> None:
        self._step.items_out = items_out

    def input(self, items_in: Optional[float]) -> None:
        self._step.items_in = items_in

    def note(self, bi: Bi) -> None:
        self._step.notes.append(_as_bi(bi))

    def warn(self, bi: Bi) -> None:
        self._step.warnings.append(_as_bi(bi))

    def detail(self, key: str, value: Any) -> None:
        self._step.details[key] = value

    def sample(self, label: Bi, text: object) -> None:
        """Attach a real excerpt of the data as it looked at this point."""
        body = "" if text is None else str(text)
        self._step.samples.append((_as_bi(label), body[: self._sample_chars]))


def _as_bi(value) -> Bi:
    if isinstance(value, tuple) and len(value) == 2:
        return (str(value[0]), str(value[1]))
    return (str(value), str(value))


class RunReport:
    """
    Collects one record per step, then writes a folder of Markdown files.

    Safety contract: `step()` never suppresses an exception from the wrapped
    code (the pipeline's own error handling stays in charge), it only records
    that the step failed. `write()` swallows its own errors, because failing
    to write a report must never fail a run that otherwise succeeded.
    """

    def __init__(
        self,
        out_dir: os.PathLike | str = "reports",
        title: str = "Pipeline run",
        lang: str = i18n.BOTH,
        enabled: bool = True,
        sample_chars: int = 700,
        settings_path: Optional[str] = None,
        command: Optional[str] = None,
    ):
        self.out_dir = Path(out_dir)
        self.title = title
        self.lang = lang
        self.enabled = enabled
        self.sample_chars = sample_chars
        self.settings_path = settings_path
        self.command = command or " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
        self.started_at = datetime.now()
        self.steps: List[Step] = []
        self.findings: List[Any] = []          # health.Finding objects
        self.context: List[Bi] = []            # free-form run facts
        self._t0 = time.time()

    # ── recording ────────────────────────────────────────────────────────────

    @contextmanager
    def step(self, name: str, items_in: Optional[float] = None, scope: str = "") -> Iterator[StepHandle]:
        step = Step(order=len(self.steps) + 1, name=name, items_in=items_in, scope=scope)
        handle = StepHandle(step, self.sample_chars)
        if not self.enabled:
            yield handle
            return
        self.steps.append(step)
        t0 = time.time()
        try:
            yield handle
        except Exception as exc:
            step.status = FAILED
            step.error = f"{type(exc).__name__}: {exc}"
            raise                      # never swallow the caller's failure
        finally:
            step.seconds = time.time() - t0

    def record(self, name: str, items_in=None, items_out=None, seconds: float = 0.0,
               status: str = OK, scope: str = "", notes: Optional[List[Bi]] = None,
               warnings: Optional[List[Bi]] = None, **details) -> Step:
        """Record an already-finished step (for code that cannot use `with`)."""
        step = Step(
            order=len(self.steps) + 1, name=name, items_in=items_in, items_out=items_out,
            seconds=seconds, status=status, scope=scope,
            notes=[_as_bi(n) for n in (notes or [])],
            warnings=[_as_bi(w) for w in (warnings or [])],
            details=details,
        )
        if self.enabled:
            self.steps.append(step)
        return step

    def add_context(self, bi: Bi) -> None:
        self.context.append(_as_bi(bi))

    def add_findings(self, findings) -> None:
        self.findings.extend(findings or [])

    # ── output ──────────────────────────────────────────────────────────────

    @property
    def total_seconds(self) -> float:
        return time.time() - self._t0

    def console_summary(self) -> str:
        """Compact table for stdout -- the 10-second version of the report."""
        rows = []
        for s in self.steps:
            icon = {OK: "✅", FAILED: "❌", SKIPPED: "⏭"}.get(s.status, "•")
            arrow = "→"
            a = "-" if s.items_in is None else f"{s.items_in:g}"
            b = "-" if s.items_out is None else f"{s.items_out:g}"
            flag = "  ⚠" if s.warnings else ""
            rows.append(f"  {icon} {s.order:>2}. {s.name:<22} {a:>7} {arrow} {b:<7} "
                        f"{i18n.human_seconds(s.seconds):>8}{flag}")
        worst = _worst_severity(self.findings)
        head = f"\n── {self.title} ── {len(self.steps)} step(s) in {i18n.human_seconds(self.total_seconds)}"
        tail = ""
        if self.findings:
            tail = f"\n  data health: {worst} — {len(self.findings)} finding(s), see DATA_HEALTH.md"
        return head + "\n" + "\n".join(rows) + tail

    def write(self) -> Optional[Path]:
        """Write the report folder. Returns its path, or None if disabled/failed."""
        if not self.enabled:
            return None
        try:
            stamp = self.started_at.strftime("%Y%m%d_%H%M%S")
            folder = self.out_dir / f"run_{stamp}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "00_SUMMARY.md").write_text(self._summary_md(folder), encoding="utf-8")
            for s in self.steps:
                name = f"{s.order:02d}_{s.name}.md"
                (folder / name).write_text(self._step_md(s), encoding="utf-8")
            if self.findings:
                from .health import findings_md          # local import: avoids a cycle
                (folder / "DATA_HEALTH.md").write_text(
                    findings_md(self.findings, lang=self.lang), encoding="utf-8")
            (folder / "RAW_TRACE.json").write_text(
                json.dumps({
                    "title": self.title,
                    "started_at": self.started_at.isoformat(timespec="seconds"),
                    "total_seconds": round(self.total_seconds, 3),
                    "command": self.command,
                    "python": platform.python_version(),
                    "settings_path": self.settings_path,
                    "steps": [s.to_dict() for s in self.steps],
                    "findings": [getattr(f, "to_dict", lambda: {})() for f in self.findings],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return folder
        except Exception as exc:                      # never break a good run
            print(f"  [observability] WARNING: could not write report: {exc}")
            return None

    # ── Markdown builders ───────────────────────────────────────────────────

    def _summary_md(self, folder: Path) -> str:
        L = self.lang
        out: List[str] = [f"# {self.title} — run summary\n"]
        out.append(i18n.para((
            "This folder explains, step by step, what this project just did to your documents. "
            "Read this file top to bottom; each step also has its own file with real excerpts of your data.",
            "این پوشه مرحله‌به‌مرحله توضیح می‌دهد که این پروژه با اسناد شما چه کرد. این فایل را از بالا به پایین بخوانید؛ هر مرحله فایل جداگانه‌ای با نمونه واقعی داده‌های شما دارد.",
        ), L))

        out.append(i18n.heading(("Run facts", "اطلاعات اجرا"), 2, L))
        facts = [
            ("Started", self.started_at.strftime("%Y-%m-%d %H:%M:%S")),
            ("Duration", i18n.human_seconds(self.total_seconds)),
            ("Command", self.command),
            ("Settings file", self.settings_path or "(default settings.yaml)"),
            ("Python", platform.python_version()),
            ("Steps recorded", len(self.steps)),
            ("Steps failed", sum(1 for s in self.steps if s.status == FAILED)),
        ]
        out.append(i18n.table(["Field", "Value"], facts))

        if self.context:
            out.append("\n" + i18n.heading(("Configuration that shaped this run", "تنظیمات موثر در این اجرا"), 2, L))
            for c in self.context:
                out.append(i18n.bullet(c, L))

        out.append("\n" + i18n.heading(("The funnel: how many items survived each step",
                                        "قیف: در هر مرحله چند مورد باقی ماند"), 2, L))
        out.append(i18n.para((
            "`In` and `Out` count whatever that step works on -- read the unit column. "
            "A negative change is not automatically a bug, but every change should have a reason you can name.",
            "ستون‌های ورودی و خروجی واحد همان مرحله را می‌شمارند. کاهش حتماً ایراد نیست، اما باید بتوانید دلیلش را نام ببرید.",
        ), L))
        rows = []
        for s in self.steps:
            d = doc_for(s.name)
            icon = {OK: "✅", FAILED: "❌", SKIPPED: "⏭"}.get(s.status, "•")
            delta = s.delta
            delta_txt = "—" if delta is None else (f"+{delta:g}" if delta > 0 else f"{delta:g}")
            rows.append([
                s.order, f"[{s.name}]({s.order:02d}_{s.name}.md)", d.title[0],
                "—" if s.items_in is None else f"{s.items_in:g}",
                "—" if s.items_out is None else f"{s.items_out:g}",
                delta_txt, f"{d.unit_in[0]} → {d.unit_out[0]}",
                i18n.human_seconds(s.seconds),
                icon + (" ⚠" if s.warnings else ""),
            ])
        out.append(i18n.table(["#", "Step", "What it does", "In", "Out", "Change", "Unit", "Time", "OK?"], rows))

        losses = [s for s in self.steps if (s.delta or 0) < 0]
        gains = [s for s in self.steps if (s.delta or 0) > 0]
        out.append("\n" + i18n.heading(("Where the numbers changed", "کجا اعداد تغییر کردند"), 2, L))
        if not losses and not gains:
            out.append(i18n.para(("No step changed its item count.", "هیچ مرحله‌ای تعداد موارد را تغییر نداد."), L))
        for s in gains:
            out.append(i18n.bullet((
                f"**{s.name}** produced {s.delta:g} more item(s) than it received — splitting is expected here. See {doc_for(s.name).source}.",
                f"مرحله **{s.name}** {s.delta:g} مورد بیشتر از ورودی تولید کرد — تقسیم شدن طبیعی است.",
            ), L))
        for s in losses:
            out.append(i18n.bullet((
                f"**{s.name}** returned {abs(s.delta):g} fewer item(s) than it received — confirm this was filtering, not data loss. See {doc_for(s.name).source}.",
                f"مرحله **{s.name}** {abs(s.delta):g} مورد کمتر برگرداند — مطمئن شوید فیلتر بوده، نه از دست رفتن داده.",
            ), L))

        failures = [s for s in self.steps if s.status == FAILED]
        if failures:
            out.append("\n" + i18n.heading(("Failures", "خطاها"), 2, L))
            out.append(i18n.table(["#", "Step", "Scope", "Error"],
                                  [[s.order, s.name, s.scope or "—", s.error or "—"] for s in failures]))

        all_warnings = [(s, w) for s in self.steps for w in s.warnings]
        if all_warnings:
            out.append("\n" + i18n.heading(("Warnings", "هشدارها"), 2, L))
            for s, w in all_warnings:
                out.append(i18n.bullet((f"`{s.name}`: {w[0]}", f"`{s.name}`: {w[1]}"), L))

        if self.findings:
            out.append("\n" + i18n.heading(("Data health", "سلامت داده"), 2, L))
            out.append(i18n.para((
                f"{len(self.findings)} finding(s); worst severity {_worst_severity(self.findings)}. Full detail in DATA_HEALTH.md.",
                f"{len(self.findings)} یافته؛ بدترین سطح {_worst_severity(self.findings)}. جزئیات کامل در DATA_HEALTH.md.",
            ), L))
            rows = [[getattr(f, "icon", "•"), getattr(f, "id", "?"), getattr(f, "count", ""),
                     getattr(f, "title", ("", ""))[0]] for f in self.findings]
            out.append(i18n.table(["", "Finding", "Count", "Summary"], rows))

        out.append("\n" + i18n.heading(("How to read this folder", "راهنمای خواندن این پوشه"), 2, L))
        out.append(i18n.side_by_side([
            ("00_SUMMARY.md — this file: the whole run at a glance.",
             "00_SUMMARY.md — همین فایل: کل اجرا در یک نگاه."),
            ("NN_<step>.md — one file per step: what it does, what it received, what it produced, and real excerpts.",
             "NN_<step>.md — یک فایل برای هر مرحله: کارش چیست، چه گرفت، چه داد، به همراه نمونه واقعی."),
            ("DATA_HEALTH.md — the problems found in your data, with the fix for each.",
             "DATA_HEALTH.md — مشکلات داده‌های شما همراه راه‌حل."),
            ("RAW_TRACE.json — the same numbers as machine-readable JSON, for diffing two runs.",
             "RAW_TRACE.json — همان اعداد به شکل JSON برای مقایسه دو اجرا."),
        ], lang=L))
        return "\n".join(out)

    def _step_md(self, s: Step) -> str:
        L = self.lang
        d = doc_for(s.name)
        out = [f"# Step {s.order}: {d.title[0]} — {d.title[1]}\n",
               f"`{s.name}`\n"]

        out.append(i18n.heading(("What this step does", "این مرحله چه کار می‌کند"), 2, L))
        out.append(i18n.para(d.what, L))

        out.append(i18n.heading(("Where this lives in the code", "کد این مرحله کجاست"), 2, L))
        out.append(f"`rag_pipeline/{d.source}`\n")

        out.append(i18n.heading(("What happened this run", "در این اجرا چه شد"), 2, L))
        delta = s.delta
        out.append(i18n.table(
            ["Measure", "Value"],
            [
                [f"Received ({d.unit_in[0]})", "—" if s.items_in is None else f"{s.items_in:g}"],
                [f"Produced ({d.unit_out[0]})", "—" if s.items_out is None else f"{s.items_out:g}"],
                ["Change", "—" if delta is None else f"{delta:+g}"],
                ["Time", i18n.human_seconds(s.seconds)],
                ["Status", s.status],
                ["Scope", s.scope or "—"],
            ],
        ))
        if s.error:
            out.append("\n**Error:**\n")
            out.append(i18n.fence(s.error, "text"))

        if s.details:
            out.append("\n" + i18n.heading(("Settings and measurements", "تنطیمات و اندازه‌گیری‌ها"), 3, L))
            out.append(i18n.table(["Key", "Value"], [[k, v] for k, v in s.details.items()]))

        if s.notes:
            out.append("\n" + i18n.heading(("Notes", "یادداشت‌ها"), 3, L))
            for n in s.notes:
                out.append(i18n.bullet(n, L))

        if s.warnings:
            out.append("\n" + i18n.heading(("⚠️ Warnings", "⚠️ هشدارها"), 3, L))
            for w in s.warnings:
                out.append(i18n.bullet(w, L))

        out.append("\n" + i18n.heading(("What usually goes wrong here", "معمولاً اینجا چه خراب می‌شود"), 2, L))
        out.append(i18n.para(d.watch, L))

        if s.samples:
            out.append(i18n.heading(("Your actual data at this point", "داده واقعی شما در این نقطه"), 2, L))
            out.append(i18n.para((
                "Excerpts, not the whole content. If these look wrong, the problem is at or before this step.",
                "فقط بریده، نه کل محتوا. اگر این‌ها غلط به نظر می‌رسند، مشکل در این مرحله یا قبل از آن است.",
            ), L))
            for label, text in s.samples:
                out.append(f"**{label[0]}** — {i18n.rtl(label[1])}\n")
                out.append(i18n.fence(text, "text", max_chars=self.sample_chars))
        return "\n".join(out)


def _worst_severity(findings) -> str:
    order = {"🔴": 3, "🟡": 2, "🟢": 1}
    worst, icon = 0, "🟢"
    for f in findings or []:
        i = getattr(f, "icon", "🟢")
        if order.get(i, 0) > worst:
            worst, icon = order.get(i, 0), i
    return icon
