# Observability / مشاهده‌پذیری

**English:** This file explains how to see what the pipeline is doing, step by step, and how to find out where your data has problems — even if you have never seen this project before.

<div dir="rtl">

**فارسی:** این فایل توضیح می‌دهد چگونه مرحله به مرحله ببینید پروژه چه کاری انجام می‌دهد و مشکل داده‌های شما کجاست — حتی اگر این پروژه را نمی‌شناسید.

</div>

---

## 1. Two commands / دو دستور

```bash
# A. Is my machine ready?  —  آیا سیستم من آماده است؟
python rag_pipeline/run_doctor.py
python rag_pipeline/run_doctor.py --out reports      # also writes reports/DOCTOR.md

# B. Run the chunking and explain every step  —  اجرا و توضیح هر مرحله
python rag_pipeline/run_pipeline.py
```

**Always run the doctor first.** It never changes your data. It only looks and reports.

<div dir="rtl">

**اول دکتر را اجرا کنید.** هیچ داده‌ای را تغییر نمی‌دهد؛ فقط نگاه می‌کند و گزارش می‌دهد.

</div>

---

## 2. `run_doctor.py` — the pre-flight check

It answers the questions you must answer **before** blaming the code:

| Group | What it checks |
|---|---|
| Python | Version is new enough |
| Settings | `settings.yaml` exists and parses |
| Dependencies | Which required packages are installed / missing, and the exact `pip install` to fix it |
| Models (Ollama) | Is the server reachable at `ollama.host`; is every model named in `settings.yaml` actually pulled |
| Your data | How many `.md` inputs, how many chunks, how many vectors in Chroma, how big the BM25 index |
| Index consistency | Chroma count vs BM25 count — if they differ, one index is stale |
| Config sanity | Contradictions inside `settings.yaml` (e.g. `generation.num_ctx` bigger than `ollama.num_ctx`, dead settings, retrieved-vs-used mismatch) |

Each line is one of:

- `✅` fine
- `⚠️` works, but you should know
- `❌` **blocking** — the pipeline cannot work until this is fixed
- `⏭` cannot be checked right now (for example: model list cannot be verified while Ollama is off)

At the end it prints a verdict (`READY` / `NOT READY`) and **the single next command** to run.
Hugging Face model ids (like `Qwen/Qwen3-Embedding-8B`) are intentionally skipped, because those are downloaded by the Python ML libraries, not by Ollama.

---

## 3. `run_pipeline.py` — the run report

Every run writes a folder:

```
reports/run_20260816_012530/
├─ 00_SUMMARY.md      ← START HERE
├─ 01_load_markdown.md
├─ 02_preprocess_markdown.md
├─ 03_split_structure.md
├─ 04_resolve_images.md
├─ 05_split_semantic.md
├─ ... one file per step ...
├─ DATA_HEALTH.md     ← problems in YOUR data, with fixes
└─ RAW_TRACE.json     ← same data for scripts/CI
```

### `00_SUMMARY.md`
One table with every step: how many items went **in**, how many came **out**, the change, how long it took, and status. This is where you see the suspicious number — for example one document becoming 800 chunks, or 3.

### `NN_<step>.md`
For each step, in plain language:

1. **What it does** and **why it exists**
2. **Which source file implements it** — so you can go read the real code
3. **Measurements** (characters read, sections found, smallest/largest chunk, model used, …)
4. **A real excerpt of your own data** at that moment — this is the part that makes the project understandable: you literally see your text change shape from stage to stage
5. **Warnings** raised by that step

### `DATA_HEALTH.md`
Rules applied to the chunks you just produced, each with severity, how many chunks are affected, why it matters for answer quality, and how to fix it:

`no_chunks`, `empty_text`, `too_short`, `too_long`, `unknown_language`,
`missing_breadcrumb`, `duplicate_chunk_id`, `duplicate_text`,
`unresolved_image_placeholder`, `raw_html`, `missing_embed_text`,
`missing_page_numbers`, `single_source_file`

Severity: 🔴 fix before trusting answers · 🟡 look at it · 🟢 fine.
Thresholds are **not invented** — they are read from your `settings.yaml`
(`semantic_chunker.min_chunk_chars`, `max_chunk_chars`, `embed_text.enabled`).

You can also check data health on its own, without re-running the pipeline:

```bash
python -m rag_pipeline.observability.health --chunks-dir ./chunks
```

---

## 4. Flags / پرچم‌ها

| Flag | Meaning |
|---|---|
| `--no-report` | Turn observability completely off (original behavior, zero overhead) |
| `--report-dir DIR` | Where to write reports (default `./reports`) |
| `--report-lang en\|fa\|both` | Report language (default `both`) |

---

## 5. Design rules I followed / قواعد طراحی

1. **Observation only.** No step's input or output was changed by this layer. Chunk files are byte-for-byte what they were.
2. **Off by default for library callers.** `MarkdownChunkingPipeline(cfg)` with no `report=` gets a disabled report that records nothing — so all pre-existing code and tests are unaffected.
3. **Never break a good run.** Report writing and the health check are wrapped so an error there prints a warning instead of failing your pipeline. A real step failure is still recorded **and re-raised**, never hidden.
4. **No new dependencies.** Standard library only.
5. **Local only.** Nothing is uploaded anywhere.
6. **Says where to look.** Every step file names the source file that implements it.
