## [1.7.0] - 2026-08-17

### Fixed — Wrong page numbers in references

Reported symptom: chapters and sections were always detected correctly, but
the page numbers in citations were wrong. Full explanation in `PAGES.md`.

**Root cause — two different things were being compared.** The old fallback
decided a chunk's page by walking the OCR sidecar's per-page `num_chunks`
and stopping when the running total passed the pipeline's `chunk_index`.
But the sidecar counts the **OCR tool's** chunks (49 for `Testing.pdf`)
while the pipeline builds its **own** semantic chunks from the heading
structure (11 for the same file). Measured on the real fixture:

```
BEFORE  chunks 0-3 -> page 1, chunks 4-10 -> page 2
        pages 3, 4, 5, 6 were UNREACHABLE
AFTER   chunks spread across pages 1, 2, 3, 4, 5, 6
```

Headings were unaffected because they come from the Markdown heading tree
and never passed through this code — exactly the asymmetry reported.

**Second defect — page markers are boundaries, not labels.** Where inline
markers do exist, a chunk with no marker inside it got no page at all, text
before the first marker was attributed to the newly opened page instead of
the previous one (an off-by-one at every page break), and a trailing marker
was applied to the chunk containing it rather than the next one.

- **New `rag_pipeline/enrichers/page_mapper.py`** — all page arithmetic,
  standard library only, no regexes or I/O.
  - `build_page_index()` reads the sidecar and degrades gracefully:
    per-page `token_count` → element counts → a bare total page count.
    Handles 0- and 1-indexed page numbers, several key spellings
    (`pages` / `page_stats` / `elements`), and string-typed junk values.
  - `assign_pages_by_length()` places chunks by **cumulative character
    position**, not chunk index, so it survives the pipeline chunking text
    differently from the OCR tool.
  - `pages_for_span()` returns **every** page a chunk overlaps, so a chunk
    straddling a break honestly reports `pages 4-5` instead of guessing.
  - `resolve_marker_pages()` carries a running "current page" cursor across
    chunks, fixing all three marker defects above.
  - Empty pages get a small weight floor so they remain citable.
- **`metadata_enricher.py`** — `enrich()` is now two-pass, because page
  attribution is a document-level question that cannot be answered while
  looking at one chunk in isolation. Picks the best method available per
  document and records it as `page_source`: `markers` (exact) /
  `estimated` (inferred) / `none`.
- **`_estimate_page_numbers()` is kept but no longer called**, so its
  original tests still document the old behaviour for comparison.

### Added — Estimated pages are visibly marked as estimates

A page the reader cannot verify must not look identical to one they can; a
confidently wrong page silently discredits every other citation.

- `document_reference.page_is_estimated()`, and `~` in rendered references
  (`Testing.pdf, ~page 4 - Chapter: ...`, short form `~p.4`).
- `page_source` and `pages_estimated` added to `structured_reference()`,
  `reference_fields()`, and each answer's `sources[]` entry.
- Query Inspector shows `~p.N` per chunk and carries `page_source` in the
  trace, so the recovery method is always visible.
- Absent `page_source` still renders as exact, so previously stored chats
  and existing callers are unchanged.

### Tests

- **New `tests/test_page_mapper.py` (44 tests)**, registered in
  `run_tests.py` as suite 32, including the regression test asserting that
  **all six pages of the real fixture become reachable**.
- Two `test_metadata_enricher.py` page tests rewrote their expectations:
  they encoded the old chunk-counting behaviour, which was the bug itself.
  Enricher suite grew 40 → 49 tests covering marker inheritance,
  page-break straddling, marker stripping, and estimate labelling.
- Suite total: **1043 tests, 0 failures** (9 errors remain the optional
  `ragas` package being absent from the environment).

### Note

Page numbers are written into chunk metadata at ingestion time, so the
ingestion pipeline must be re-run for already-indexed chunks to pick up
corrected pages. Your current Markdown contains **no** page markers at all
(verified with `grep`), so pages are `estimated`; `PAGES.md` explains the
two ways to obtain exact pages.

---

## [1.6.0] - 2026-08-16

### Added — Broad/general questions ("what is prompt engineering?")

General questions retrieved five chunks that were often all from one
narrow section, so answers read as if the document only covered that
corner of the topic. See `GENERAL_QUESTIONS.md`.

- **New `rag_pipeline/retrieval/document_map.py`**
  - `is_broad_question()` detects general questions in English **and
    Persian** (چیست، توضیح بده، انواع, ...), with specific-question
    patterns taking precedence.
  - `diversify_by_section()` caps how many chunks one section may occupy,
    then backfills, so breadth never costs a retrieval slot.
  - `build_document_map_block()` prepends a compact outline of the source
    document for orientation. It explicitly forbids citing itself, so it
    cannot become a source of unsupported claims.
- **Config**: `max_chunks_per_section` (default 2), `document_map`
  (`auto` | `always` | `off`), `document_map_max_chapters`,
  `document_map_max_sections`, wired through all three explicit
  `ContextBuilderConfig` construction sites.
- **`ChromaIngestor.list_metadatas()`** to build the outline from the whole
  corpus, cached per answerer instance.
- Query Inspector reports breadth detection and which sections were used.
- **New `tests/test_document_map.py` (60 tests)**; suite total 990.

---

## [1.5.0] - 2026-08-16

### Added — Document references (page / chapter / section)

Answers can now say **where** in the document each claim came from, instead
of only a bracketed number. See `REFERENCES.md`.

- **New `rag_pipeline/document_reference.py`** — single source of truth for
  building a document location out of chunk metadata. Standard library only.
  - Heading levels are labelled by **position in the heading chain, not by
    `#` depth**, so a document whose top heading is `##` still has chapters.
  - Levels deeper than three reuse the `Subsection` label.
  - Persian labels available (فصل / بخش / زیربخش).
  - Page runs collapse for readability: `4-6`, and `4, 9` when there is a gap.
  - **Backward compatible:** when `chapter`/`section` fields are absent it
    parses the old `breadcrumb` string, so references work on a `chroma_db`
    built before this version — no re-ingestion required.
  - Tolerates ChromaDB's list flattening (`page_numbers` arriving as `"1,2"`).
- **`enrichers/metadata_enricher.py`** — each chunk now stores `chapter`,
  `section`, `subsection`, `heading_path`, `heading_depth`, `page_display`
  and a ready-to-display `reference`. All flat scalars, since ChromaDB
  rejects list-valued metadata.
- **`generation/context_builder.py`** — context entries are labelled
  `Chapter: X > Section: Y` rather than a bare `X > Y`, which is what lets
  the model name the location in its answer.
- **`generation/rag_answerer.py`** — the system prompt asks the model to
  repeat the document location after each citation and to copy it exactly
  rather than guess. `sources` entries gained `citation_number`, `source`,
  `pages`, `chapter`, `section`, `subsection`, `heading_path`, `levels`,
  `reference` and `reference_short`; all previously existing keys are
  unchanged.
- **Query Inspector** — every candidate row carries its document location,
  including candidates that were dropped, so you can see whether the right
  part of the document was retrieved but discarded.
- **Web app, `run_answer.py`, `run_query.py`** — all display the full
  location.
- **`settings.yaml`** → `generation.context_builder`:
  `label_heading_levels` and `cite_document_location` (both `true`).
  Forwarded explicitly at all three call sites that rebuild the config.
- **Tests** — new `tests/test_document_reference.py` (**46 tests**),
  registered in `tests/run_tests.py`. Suite total: **930 tests, 0 failures**
  (9 pre-existing environmental errors from the optional `ragas` /
  `langchain_community` packages).

### Fixed

- `clean_heading()` stripped Markdown hashes with an anchored `^#+` pattern,
  which never matched breadcrumb segments because splitting on `" > "`
  leaves a leading space (`" ## Few-shot"`). Whitespace is now trimmed
  first. Caught by a test before shipping.

---

## [1.4.0] - 2026-08-16

### Added
- Query Inspector in the web app: every answer now carries a "Why this answer?" panel showing which chunks were retrieved, which of them actually reached the prompt, and the literal prompt the model received. Four tabs: Flow (findings + retrieval funnel + per-stage timings), Chunks (per-candidate cards with dense/BM25 rank, RRF score, rerank score and movement, sent-vs-dropped, cited-vs-ignored, chars sent vs total, truncation flag, and query-term highlighting), Exact prompt (system / context block / full user message, each copyable, plus context-window usage), Setup (models, thresholds, language filter, and the three different forms of your query). Labels are English + Persian.
- rag_pipeline/observability/query_trace.py: QueryTracer / trace_answer(answerer, question). It wraps the live components, calls the ordinary RagAnswerer.answer() once, and restores every method in a finally block, so the answer is byte-for-byte identical with tracing on or off and the displayed prompt is the real one, not a reconstruction. If building the trace fails, the trace is dropped and the answer still returns.
- Automatic findings per question (dropped chunks, truncation, uncited context, invented citations, single-source answers, empty dense/sparse results, language filter hiding everything, context-window overflow risk).
- webapp: GET /api/trace/demo returns a realistic example trace so the whole visualization can be seen without Ollama/Chroma/BM25 ("See example inspector" button in the sidebar); GET /api/chats/<id>/messages/<index>/trace re-opens the inspector for older messages; GET /api/health reports pipeline status without forcing a build.
- tests/test_query_trace.py (26 tests, no Ollama needed) using the real HybridRetriever, real RRF and real RagAnswerer with fakes only at the edges.
- VISUALIZATION.md.

### Changed
- webapp/chat_store.py add_message() accepts an optional trace and stores it on the message, so the inspector survives a page reload. Messages saved by older versions have no trace and are treated as "inspector not available".
- POST /api/chats/<id>/messages accepts {"trace": true} or ?trace=1. Without it, behaviour is exactly as before.

## [1.3.0] - 2026-08-16

### Fixed
- settings.yaml is now actually honoured by run_query.py, run_answer.py, run_evaluation.py and webapp/app.py (they built config from dataclass defaults, so reranker backend/model, hybrid retriever counts and candidate_pool silently ignored your yaml). Precedence is now CLI flag > settings.yaml > default.
- rag_answerer.py reported every retrieved chunk as a source even though only context_builder.max_chunks chunks reached the prompt; sources/context_used now match the prompt exactly.
- requirements.txt had two conflicting langchain-community pins and an unused langchain pin; split into requirements.txt (core), requirements-rerank.txt, requirements-eval.txt.
- Removed dead _resolve_image_placeholders() from pipeline.py.
- doctor.py read context_builder.max_chunks from the wrong path (it is nested under generation:), so the retrieved-vs-used check never fired.

### Added
- rag_pipeline/run_doctor.py + doctor.py: pre-flight check of Python, settings, dependencies, Ollama reachability, configured models, data counts, index consistency and config contradictions, with a verdict and the next command to run.
- rag_pipeline/observability/: run_report.py (per-step Markdown report with in/out counts, timing, warnings and real data excerpts), health.py (13 data-quality rules over produced chunks), i18n.py (English/Persian output).
- run_pipeline.py flags --no-report, --report-dir, --report-lang; writes reports/run_<timestamp>/{00_SUMMARY.md, NN_<step>.md, DATA_HEALTH.md, RAW_TRACE.json}.
- tests/test_observability.py; FIXES.md; OBSERVABILITY.md.

### Notes
- No new dependencies (standard library only). Observability is inert for existing library callers and never alters chunk output.

# Changelog

## 1.0.1 — 2026-08-16

Continuation release after restoring and revalidating the source project.

### Fixes

- Made Ollama embedding requests omit empty `options` payloads for older client compatibility.
- Bundled a dependency-free BM25 implementation with portable persisted indexes.
- Unified the embedding model tag as `qwen3-embedding:8b`.
- Corrected the Hugging Face tokenizer identifier and aligned the default retrieval context to five chunks.
- Added Flask explicitly to the runtime requirements for the web UI.

### Verification

- Core regression tests: **151 passed**.
- Fixture pipeline: **20 chunks, 20 unique chunk IDs**.
- Chunk integrity: **passed**; embedding text contains no HTML table markup.
- BM25 build and persistence smoke test: **passed**.
- Live Ollama/Chroma/RAGAS execution remains target-machine work because those services and optional packages are unavailable in this sandbox.

## 1.2.0 — Modular pipeline architecture

- Added one documented adapter module per ingestion step under `rag_pipeline/stages/`.
- Kept `pipeline.py` as the sole orchestrator.
- Added `stages/README.md` and `PIPELINE_MODULES.md` with contracts and debugging steps.
- Preserved replaceable component seams used by the existing tests.
