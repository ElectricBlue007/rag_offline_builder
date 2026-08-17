"""
pipeline.py — Main pipeline orchestrator.

Chains:
  MarkdownLoader
    → MarkdownPreprocessor
      → StructureSplitter
        → SemanticChunker
          → LlmChunkRefiner        (optional: LLM-judged merge of split-too-early chunks)
            → TableChunkNormalizer      (HTML tables → embeddable prose)
              → ContextualEnricher      (situates each chunk in its document)
                → EmbedTextBuilder      (applies the embedding model's prefix)
                  → MetadataEnricher    (adds chunk_id + descriptive metadata)
                    → JSON output (one file per .md input)

Each stage is independent and can be swapped or extended without touching
the others.
"""

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from rag_pipeline.config import PipelineConfig, RuntimeConfig
from rag_pipeline import model_manager
from rag_pipeline.observability import RunReport
from rag_pipeline.preprocessors.md_preprocessor import MarkdownPreprocessor
from rag_pipeline.splitters.structure_splitter import StructureSplitter
from rag_pipeline.splitters.semantic_chunker import SemanticChunker
from rag_pipeline.splitters.llm_chunk_refiner import LlmChunkRefiner
from rag_pipeline.normalizers.chunk_normalizer import TableChunkNormalizer
from rag_pipeline.enrichers.contextual_enricher import ContextualEnricher
from rag_pipeline.enrichers.embed_text_builder import EmbedTextBuilder
from rag_pipeline.enrichers.metadata_enricher import MetadataEnricher, EnrichedChunk
from rag_pipeline.stages.load_markdown import MarkdownLoaderStage
from rag_pipeline.stages.preprocess_markdown import MarkdownPreprocessStage
from rag_pipeline.stages.split_structure import StructureSplitStage
from rag_pipeline.stages.resolve_images import ImagePlaceholderResolutionStage
from rag_pipeline.stages.split_semantic import SemanticSplitStage
from rag_pipeline.stages.refine_chunks import ChunkRefinementStage
from rag_pipeline.stages.normalize_tables import TableNormalizationStage
from rag_pipeline.stages.enrich_context import ContextEnrichmentStage
from rag_pipeline.stages.build_embed_text import EmbedTextBuildStage
from rag_pipeline.stages.enrich_metadata import MetadataEnrichmentStage
from rag_pipeline.stages.write_chunks import ChunkJsonWriterStage


# ── Markdown file loader ───────────────────────────────────────────────────────

def _discover_md_files(input_dir: Path, glob: str) -> list[Path]:
    """Recursively find all markdown files matching the glob pattern."""
    files = sorted(input_dir.glob(glob))
    # Exclude _metadata.json siblings (they're JSON, not markdown — but just in case)
    return [f for f in files if f.suffix.lower() == ".md"]


def _load_md(path: Path) -> str:
    """Read a markdown file as UTF-8 text."""
    return path.read_text(encoding="utf-8", errors="replace")


# -- small read-only helpers used only for the human-readable run report ------
# None of these touch the data; they just describe it.

def _count_types(items) -> dict:
    """How many text / table / image_description items are in this list?"""
    out: dict = {}
    for it in items:
        key = getattr(it, "section_type", "") or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


def _embeddable(chunk) -> str:
    """The text that will actually be embedded, whichever field holds it."""
    for attr in ("embed_text", "normalized_text", "embed_content", "content"):
        value = getattr(chunk, attr, "")
        if value:
            return value
    return ""


def _lang(chunk) -> str:
    """Detected language of a chunk, tolerating dict or dataclass metadata."""
    meta = getattr(chunk, "metadata", None)
    if isinstance(meta, dict):
        return meta.get("language") or "unknown"
    return getattr(meta, "language", "") or "unknown"


def _preview(items, n: int = 3) -> str:
    """A few real items, each with the heading path it came from."""
    parts = []
    for it in list(items)[:n]:
        crumb = getattr(it, "breadcrumb", "") or "(no heading)"
        kind = getattr(it, "section_type", "?")
        body = (getattr(it, "content", "") or "")[:220].replace("\n", " ")
        parts.append(f"[{kind}] {crumb}\n    {body}")
    return "\n\n".join(parts)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MarkdownChunkingPipeline:
    """
    Orchestrates the full chunking pipeline for a directory of .md files.

    Usage:
        cfg = PipelineConfig(input_dir=Path("./outputs"))
        pipeline = MarkdownChunkingPipeline(cfg)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig, runtime: Optional[RuntimeConfig] = None,
                 report: Optional[RunReport] = None):
        self.cfg = config
        self.runtime = runtime or RuntimeConfig()

        # -- Observability, opt-in and side-effect free --------------------
        # `report` is optional. When nothing is passed we build a DISABLED
        # RunReport: its .step() still yields a handle, but records nothing
        # and writes nothing. That means this class behaves exactly as it
        # always did for every existing caller and test, while
        # `run_pipeline.py` (report on by default) gets a full step-by-step
        # account of what happened to the data.
        self.report = report if report is not None else RunReport(enabled=False)

        # Instantiate each stage with its sub-config. `runtime` is
        # threaded through to every stage that calls a model, so they
        # all share the same "only one model on the GPU at a time"
        # policy from settings.yaml's runtime: block.
        self.preprocessor     = MarkdownPreprocessor(config.preprocessor)
        self.struct_split     = StructureSplitter(config.structure)
        self.sem_chunker      = SemanticChunker(config.semantic, runtime=self.runtime)
        # semantic_config is passed through so the refiner's post-merge
        # resplit safety net uses the SAME chunk-size ceiling as the
        # initial semantic split, instead of a hardcoded value of its own.
        self.llm_refiner      = LlmChunkRefiner(
            config.llm_chunk_refiner, semantic_config=config.semantic, runtime=self.runtime
        )
        self.table_normalizer = TableChunkNormalizer(config.table_normalizer)
        self.contextual       = ContextualEnricher(config.contextual, runtime=self.runtime)
        self.embed_text_bldr  = EmbedTextBuilder(config.embed_text)
        self.enricher         = MetadataEnricher(config.metadata)

        # pipeline.py only wires stages. Each transformation lives in one
        # documented module under rag_pipeline/stages/.
        self.stage_names = (
            "load_markdown", "preprocess_markdown", "split_structure",
            "resolve_images", "split_semantic", "refine_chunks",
            "normalize_tables", "enrich_context", "build_embed_text",
            "enrich_metadata", "write_chunks",
        )
        self._writer_stage = ChunkJsonWriterStage()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Process every .md file found under input_dir and write chunk JSONs
        to output_dir.

        Deliberately staged by MODEL rather than by file: every file's
        embedding-model work (steps 1-4) runs before ANY file's LLM
        refiner work (step 4b) starts, instead of interleaving embed →
        refine → embed → refine per file. On a GPU that can't hold the
        embedding model and the refiner LLM in VRAM at once, per-file
        interleaving forces Ollama to evict and reload whichever model
        isn't currently needed on EVERY file — for N files that's up to
        2N model loads. Staging by model means the embedding model loads
        once for the whole run and the refiner model loads once for the
        whole run, regardless of N. When the refiner is disabled (the
        default), stage 2 is a no-op and this behaves exactly as before.

        Returns a summary dict:
            {
              "files_processed": int,
              "total_chunks": int,
              "failed": [str, ...],
            }
        """
        input_dir  = Path(self.cfg.input_dir)
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_files = _discover_md_files(input_dir, self.cfg.md_glob)

        self.report.add_context((
            f"Input folder: {input_dir}  |  Output folder: {output_dir}  |  "
            f"File pattern: {self.cfg.md_glob}",
            f"پوشه ورودی: {input_dir}  |  پوشه خروجی: {output_dir}  |  "
            f"الگوی فایل: {self.cfg.md_glob}",
        ))
        self.report.record(
            "discover_files",
            items_out=len(md_files),
            status="ok" if md_files else "failed",
            scope=str(input_dir),
            notes=[(
                f"Found {len(md_files)} markdown file(s) under {input_dir} "
                f"matching the pattern {self.cfg.md_glob!r}.",
                f"{len(md_files)} فایل مارک‌داون در {input_dir} پیدا شد.",
            )],
            warnings=[] if md_files else [(
                "Nothing matched, so every later step has nothing to work with. Check "
                "paths.input_dir in settings.yaml, and remember it is relative to the folder "
                "you run the command from, not to the project folder.",
                "هیچ فایلی پیدا نشد؛ پس مراحل بعدی هیچ داده‌ای ندارند. "
                "مقدار paths.input_dir در settings.yaml را بررسی کنید؛ این مسیر نسبت به پوشه اجراست.",
            )],
            files=[str(p) for p in md_files[:50]],
        )

        if not md_files:
            self._log(f"⚠️  No .md files found under: {input_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": []}

        self._log(f"\n📂 Input  : {input_dir}")
        self._log(f"📂 Output : {output_dir}")
        self._log(f"🔍 Found  : {len(md_files)} markdown file(s)\n")

        failed: list[str] = []
        total_chunks = 0

        # ── Stage 1 — embedding-model work for every file ───────────────
        # (load, preprocess, structure split, semantic chunk). Collects
        # each file's intermediate state instead of finishing it, so the
        # embedding model stays the only model in play until every file
        # has been through it.
        staged: list[dict] = []
        for i, md_path in enumerate(md_files, start=1):
            self._log(f"[{i}/{len(md_files)}] Chunking  : {md_path.relative_to(input_dir)}")
            t0 = time.time()
            try:
                sem_chunks, preprocessed = self._embed_stage(md_path)
                staged.append({
                    "md_path": md_path,
                    "sem_chunks": sem_chunks,
                    "preprocessed": preprocessed,
                    "elapsed": time.time() - t0,
                })
            except Exception as exc:
                self._log(f"  ❌  FAILED (chunking): {exc}")
                failed.append(str(md_path))

        # ── Stage 2 — LLM refiner work for every file that survived stage 1
        # A genuine no-op (refine() returns its input unchanged) when
        # llm_chunk_refiner.enabled is False, so this costs nothing and
        # triggers no model load at all in the default configuration.
        if self.cfg.llm_chunk_refiner.enabled and staged:
            self._log(f"\n[{len(staged)} file(s)] Refining chunks with LLM...")
            # Stage 1 is fully done for every file at this point — release
            # the embedding model NOW rather than waiting for the refiner
            # model's own ensure_active() call to evict it implicitly.
            # Same "load it, use it, explicitly stop it, then load the
            # next one" flow the pipeline module docstring promises,
            # made visible at the orchestration level too.
            model_manager.release(
                self.cfg.semantic.embedding_host, self.cfg.semantic.embedding_model, self.runtime
            )
        for entry in staged:
            t0 = time.time()
            try:
                before = len(entry["sem_chunks"])
                enabled = self.cfg.llm_chunk_refiner.enabled
                with self.report.step("refine_chunks", items_in=before,
                                      scope=entry["md_path"].name) as st:
                    entry["sem_chunks"] = self.llm_refiner.refine(entry["sem_chunks"])
                    after = len(entry["sem_chunks"])
                    st.done(after)
                    st.detail("enabled", enabled)
                    st.detail("chunks_merged", max(0, before - after))
                    st.note((
                        (f"An LLM re-read neighbouring chunks and merged {before - after} pair(s) that had "
                         "been cut before the thought was finished.")
                        if enabled else
                        ("This step is switched off (llm_chunk_refiner.enabled: false), so the chunks "
                         "passed through untouched and no model was loaded. Turning it on costs roughly "
                         "double the chunking time and usually improves chunk boundaries."),
                        (f"یک مدل زبانی قطعات مجاور را خواند و {before - after} جفت را ادغام کرد.")
                        if enabled else
                        ("این مرحله خاموش است؛ قطعات بدون تغییر رد شدند و مدلی لود نشد."),
                    ))
                entry["elapsed"] += time.time() - t0
            except Exception as exc:
                rel = entry["md_path"].relative_to(input_dir)
                self._log(f"  ❌  FAILED (refine) {rel}: {exc}")
                failed.append(str(entry["md_path"]))
                entry["skip"] = True

        # ── Stage 3 — everything else (no model, or whatever model the
        # contextual enricher uses if --contextual-mode llm is set) —
        # run per file and save. No embedding/refiner calls happen here.
        if self.cfg.llm_chunk_refiner.enabled and staged:
            # Refiner stage is done for every file — release its model
            # before stage 3 potentially loads yet another one (the
            # contextual enricher's LLM, if --contextual-mode llm).
            model_manager.release(
                self.cfg.llm_chunk_refiner.llm_host, self.cfg.llm_chunk_refiner.llm_model, self.runtime
            )
        for entry in staged:
            if entry.get("skip"):
                continue
            md_path = entry["md_path"]
            t0 = time.time()
            try:
                enriched = self._finish_stage(
                    entry["sem_chunks"], md_path, entry["preprocessed"]
                )
                self._save_chunks(enriched, md_path, output_dir, input_dir)
                elapsed = entry["elapsed"] + (time.time() - t0)
                self._log(
                    f"  ✅  {len(enriched)} chunks  "
                    f"({elapsed:.1f}s)  "
                    f"→ {self._output_path(md_path, output_dir, input_dir).name}"
                )
                total_chunks += len(enriched)
            except Exception as exc:
                self._log(f"  ❌  FAILED: {exc}")
                failed.append(str(md_path))

        self._log(f"\n{'='*55}")
        self._log(f"  DONE — {len(md_files) - len(failed)} files, {total_chunks} chunks")
        if failed:
            self._log(f"  Failed: {failed}")
        self._log(f"{'='*55}\n")

        # Final safety net: whatever model was active last (embedding,
        # refiner, or the contextual enricher's LLM if mode="llm") is
        # explicitly released so nothing is left resident in VRAM once
        # the whole chunking run is done.
        if self.cfg.contextual.mode == "llm":
            model_manager.release(self.cfg.contextual.llm_host, self.cfg.contextual.llm_model, self.runtime)
        else:
            model_manager.release(
                self.cfg.semantic.embedding_host, self.cfg.semantic.embedding_model, self.runtime
            )

        return {
            "files_processed": len(md_files) - len(failed),
            "total_chunks": total_chunks,
            "failed": failed,
        }

    def process_file(self, md_path: Path) -> list[EnrichedChunk]:
        """
        Run the full pipeline for a SINGLE markdown file, start to
        finish, in one call. Used directly by callers/tests that want
        one file's result without going through run()'s multi-file
        staging — behaves identically to how this method always has,
        just now implemented as the two stages run() also uses.
        """
        sem_chunks, preprocessed = self._embed_stage(md_path)
        sem_chunks = self.llm_refiner.refine(sem_chunks)
        return self._finish_stage(sem_chunks, md_path, preprocessed)

    # ── Stage helpers (also used directly by run()'s batching) ─────────────

    def _embed_stage(self, md_path: Path):
        """
        Run stages 01–05 for one file; component seams stay replaceable.

        Each stage is wrapped in `self.report.step(...)`, which records how
        many items went in, how many came out, how long it took, a plain
        explanation, and a real excerpt of YOUR data at that point. If no
        report was passed to the constructor those calls do nothing.
        """
        scope = md_path.name

        with self.report.step("load_markdown", items_in=1, scope=scope) as st:
            raw_text = MarkdownLoaderStage().run(md_path)
            st.done(1)
            st.detail("characters_read", len(raw_text))
            st.detail("lines_read", raw_text.count("\n") + 1)
            st.sample(("The file exactly as it is on disk",
                       "فایل دقیقاً همان‌طور که روی دیسک است"), raw_text)

        with self.report.step("preprocess_markdown", items_in=1, scope=scope) as st:
            preprocessed = MarkdownPreprocessStage(self.preprocessor).run(raw_text, md_path)
            st.done(1)
            removed = len(raw_text) - len(preprocessed.cleaned_text)
            st.detail("characters_before", len(raw_text))
            st.detail("characters_after", len(preprocessed.cleaned_text))
            st.detail("characters_removed", removed)
            st.detail("image_captions_extracted", len(preprocessed.image_alts))
            st.note((
                f"Cleaning removed {removed} character(s) of page furniture (page numbers, repeated "
                f"headers, OCR noise) and lifted {len(preprocessed.image_alts)} image caption(s) out "
                "of the text so they can become their own searchable chunks.",
                f"پاک‌سازی {removed} کاراکتر اضافه (شماره صفحه، سرصفحه تکراری، نوفه OCR) را حذف کرد و "
                f"{len(preprocessed.image_alts)} توضیح تصویر را جدا کرد.",
            ))
            st.sample(("The same document after cleaning",
                       "همان سند پس از پاک‌سازی"), preprocessed.cleaned_text)

        with self.report.step("split_structure", items_in=1, scope=scope) as st:
            sections = StructureSplitStage(self.struct_split).run(preprocessed.cleaned_text, md_path)
            st.done(len(sections))
            st.detail("section_types", _count_types(sections))
            st.note((
                f"The document was cut at its headings into {len(sections)} section(s). Tables are "
                "kept whole on purpose, so a table is never cut in half.",
                f"سند بر اساس عنوان‌ها به {len(sections)} بخش تقسیم شد. جدول‌ها عمداً یکپارچه می‌مانند.",
            ))
            st.sample(("First sections, with the heading path each one came from",
                       "بخش‌های اول همراه مسیر عنوان"), _preview(sections))

        with self.report.step("resolve_images", items_in=len(sections), scope=scope) as st:
            sections = ImagePlaceholderResolutionStage().run(sections, preprocessed.image_alts)
            st.done(len(sections))
            leftover = sum(1 for s in sections if "[IMAGE:" in (getattr(s, "content", "") or ""))
            st.detail("captions_available", len(preprocessed.image_alts))
            st.detail("unresolved_placeholders_left", leftover)
            st.note((
                "Image captions taken out during cleaning are put back where the picture was, so a "
                "screenshot becomes searchable text instead of an empty gap.",
                "توضیح تصاویر دوباره سر جای خود قرار می‌گیرد تا تصویر قابل جستجو شود.",
            ))
            if leftover:
                st.warn((
                    f"{leftover} section(s) still contain an [IMAGE: ...] placeholder, which means the "
                    "caption for that picture was never found. Those chunks will be embedded with a "
                    "placeholder instead of a description, and will be almost impossible to retrieve.",
                    f"{leftover} بخش هنوز جای‌نگهدار [IMAGE: ...] دارند؛ یعنی توضیح آن تصویر پیدا نشد.",
                ))

        with self.report.step("split_semantic", items_in=len(sections), scope=scope) as st:
            sem_chunks = SemanticSplitStage(self.sem_chunker).run(sections)
            st.done(len(sem_chunks))
            sizes = [len(getattr(c, "content", "") or "") for c in sem_chunks]
            if sizes:
                st.detail("smallest_chunk_chars", min(sizes))
                st.detail("largest_chunk_chars", max(sizes))
                st.detail("average_chunk_chars", round(sum(sizes) / len(sizes)))
            st.detail("embedding_model", self.cfg.semantic.embedding_model)
            st.detail("breakpoint_rule",
                      f"{self.cfg.semantic.breakpoint_threshold_type} "
                      f"@ {self.cfg.semantic.breakpoint_threshold_amount}")
            st.note((
                f"Long sections were split again wherever the meaning shifts, judged by the embedding "
                f"model {self.cfg.semantic.embedding_model!r}. {len(sections)} section(s) became "
                f"{len(sem_chunks)} chunk(s). Chunks much smaller than "
                f"{self.cfg.semantic.min_chunk_chars} characters are usually noise; chunks near "
                f"{self.cfg.semantic.max_chunk_chars} may be answering several questions at once.",
                f"بخش‌های بلند در نقطه تغییر معنا دوباره تقسیم شدند. "
                f"{len(sections)} بخش به {len(sem_chunks)} قطعه تبدیل شد.",
            ))
            st.sample(("First chunks after semantic splitting",
                       "قطعات اول پس از تقسیم معنایی"), _preview(sem_chunks))

        return sem_chunks, preprocessed

    def _finish_stage(self, sem_chunks, md_path: Path, preprocessed) -> list[EnrichedChunk]:
        """Run stages 07–10 after optional stage 06 refinement (instrumented)."""
        scope = md_path.name
        for i, chunk in enumerate(sem_chunks): chunk.chunk_index = i

        with self.report.step("normalize_tables", items_in=len(sem_chunks), scope=scope) as st:
            table_chunks = sum(1 for c in sem_chunks if getattr(c, "section_type", "") == "table")
            sem_chunks = TableNormalizationStage(self.table_normalizer).run(sem_chunks)
            st.done(len(sem_chunks))
            st.detail("table_chunks", table_chunks)
            st.detail("enabled", self.cfg.table_normalizer.enabled)
            st.note((
                f"{table_chunks} table chunk(s) were rewritten as sentences. A raw HTML table embeds "
                "badly because the words a person would search for are buried in tags; the sentence "
                "version is what goes into the search index, while the original HTML is kept for display.",
                f"{table_chunks} قطعه جدولی به جمله تبدیل شد. جدول HTML خام برای جستجو مناسب نیست؛ "
                "نسخه جمله‌ای به ایندکس می‌رود و HTML اصلی برای نمایش می‌ماند.",
            ))
            if table_chunks:
                st.sample(("A normalized table, as the search index will see it",
                           "یک جدول نرمال‌شده همان‌گونه که ایندکس می‌بیند"),
                          next((_embeddable(c) for c in sem_chunks
                                if getattr(c, "section_type", "") == "table"), ""))

        with self.report.step("enrich_context", items_in=len(sem_chunks), scope=scope) as st:
            sem_chunks = ContextEnrichmentStage(self.contextual).run(
                sem_chunks, document_text=preprocessed.cleaned_text, source_filename=md_path.name
            )
            st.done(len(sem_chunks))
            st.detail("mode", self.cfg.contextual.mode)
            st.note((
                f"Every chunk was given a short line saying which document and which heading it came "
                f"from (mode: {self.cfg.contextual.mode}). This is what lets a chunk still make sense "
                "when search returns it alone, with no surrounding pages.",
                f"به هر قطعه یک خط کوتاه افزوده شد که می‌گوید از کدام سند و کدام عنوان آمده است "
                f"(حالت: {self.cfg.contextual.mode}).",
            ))
            if sem_chunks:
                st.sample(("A chunk after its context line was added",
                           "یک قطعه پس از افزودن خط زمینه"),
                          getattr(sem_chunks[0], "contextualized_text", "") or _embeddable(sem_chunks[0]))

        with self.report.step("build_embed_text", items_in=len(sem_chunks), scope=scope) as st:
            sem_chunks = EmbedTextBuildStage(self.embed_text_bldr).run(sem_chunks)
            st.done(len(sem_chunks))
            st.detail("enabled", self.cfg.embed_text.enabled)
            st.note((
                "This is the exact text that becomes a vector. If retrieval is disappointing, read "
                "this field first: the search model never sees anything else.",
                "این دقیقاً متنی است که به وکتور تبدیل می‌شود. اگر بازیابی ضعیف بود، اول همین را بخوانید.",
            ))
            if sem_chunks:
                st.sample(("The exact text sent to the embedding model",
                           "متن دقیقی که به مدل تعبیه فرستاده می‌شود"),
                          _embeddable(sem_chunks[0]))

        with self.report.step("enrich_metadata", items_in=len(sem_chunks), scope=scope) as st:
            enriched = MetadataEnrichmentStage(self.enricher).run(sem_chunks, md_path)
            st.done(len(enriched))
            langs: dict = {}
            for c in enriched:
                key = _lang(c)
                langs[key] = langs.get(key, 0) + 1
            st.detail("languages_detected", langs)
            st.note((
                "Each chunk now has a stable id, the heading path it came from, a detected language "
                "and its page numbers. Search filters by language, so a chunk labelled with the wrong "
                "language becomes invisible to queries in its real language.",
                "هر قطعه اکنون شناسه پایدا، مسیر عنوان، زبان تشخیص‌داده‌شده و شماره صفحه دارد. "
                "جستجو بر اساس زبان فیلتر می‌کند؛ زبان اشتباه یعنی قطعه دیده نمی‌شود.",
            ))

        return enriched

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _output_path(self, md_path: Path, output_dir: Path, input_dir: Path) -> Path:
        """Mirror the input directory structure in the output directory."""
        try:
            rel = md_path.relative_to(input_dir)
        except ValueError:
            rel = Path(md_path.name)
        return output_dir / rel.with_suffix(".chunks.json")

    def _save_chunks(
        self, chunks: list[EnrichedChunk], md_path: Path, output_dir: Path, input_dir: Path
    ) -> None:
        """Run the final write stage for one document."""
        out_path = self._output_path(md_path, output_dir, input_dir)
        with self.report.step("write_chunks", items_in=len(chunks), scope=md_path.name) as st:
            self._writer_stage.run(chunks, md_path, out_path)
            st.done(len(chunks))
            st.detail("output_file", str(out_path))
            st.detail("bytes_written", out_path.stat().st_size if out_path.exists() else 0)
            st.note((
                f"Saved as {out_path.name}: one JSON file per input document. This file is the "
                "contract between chunking and everything downstream — open it to see exactly "
                "what will be searched.",
                f"ذخیره شد در {out_path.name}: برای هر سند یک فایل JSON. "
                "همین فایل مرز بین مرحله قطعه‌بندی و بقیه سیستم است.",
            ))

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)
