# FIXES.md — what was broken and what I changed

Version 1.2.0 → 1.3.0. Every item below was found by reading the code and
proven by running it, not guessed.

<div dir="rtl">

این فایل می‌گوید چه چیزی خراب بود، چرا مهم بود، و دقیقاً چه تغییری دادم.

</div>

---

## Fixed in this release

### B1 · `settings.yaml` was silently ignored by the CLIs — **critical**

**Files:** `run_query.py`, `run_answer.py`, `run_evaluation.py`, `webapp/app.py`

**What happened:** each CLI built its config objects from **dataclass defaults**
and then applied only the command-line arguments. Values you set in
`settings.yaml` never reached the retriever or the reranker.

The practical damage: `settings.yaml` says
`retrieval.reranker.backend: ollama` with
`reranker_ollama: B-A-M-N/qwen3-reranker-0.6b-fp16:latest`, but the dataclass
default is `backend='huggingface'`, `ollama_model='qwen3:4b'`. So with
`--rerank` you were silently running a **different backend and a different
model** than your configuration file specified. Same class of bug for
`HybridRetrieverConfig` (defaults `30/30/60/True`) and
`RerankRuntime.candidate_pool` (default `10` vs your yaml `20`).

**Fix:** each entry point now starts from `load_settings()` and applies CLI
flags as overrides on top, using `dataclasses.replace(...)`. Precedence is now
the documented one: **CLI flag > settings.yaml > dataclass default.**

---

### B2 · Reranker/retriever config objects rebuilt from scratch — **high**

Same four entry points constructed fresh config objects instead of copying the
loaded ones, so any field the CLI did not know about was reset to its default.
Now every field is preserved with `replace()` and only the explicitly passed
flags change.

---

### B3 · The answer used more chunks than the prompt contained — **high**

**File:** `generation/rag_answerer.py`

**What happened:** the prompt was built from the first `max_chunks` chunks
(`context_builder.max_chunks: 5`), but the returned `sources` list and
`context_used` reported **all** retrieved chunks. So the UI cited sources the
model never actually read — the worst kind of RAG bug, because it makes a
wrong answer look well-sourced.

**Fix:** one `used_chunks = chunks[: self.context_cfg.max_chunks]` is now the
single source of truth for the prompt, the sources list, and `context_used`.

---

### B5 · `requirements.txt` had a conflicting duplicate pin — **high**

`langchain-community` appeared twice with two different constraints, and
`langchain>=0.2.0` was listed but never imported anywhere in the code. A fresh
`pip install -r requirements.txt` could resolve to an unpredictable version.

**Fix:** split into three honest files, so a normal user does not download
gigabytes of `torch` for features that are off by default:

| File | Contents |
|---|---|
| `requirements.txt` | core only — what every run needs |
| `requirements-rerank.txt` | `sentence-transformers`, `transformers` — only for the HuggingFace reranker backend |
| `requirements-eval.txt` | `ragas`, `langchain-openai`, `langchain-community` — only for evaluation |

---

### B10 · Dead method in `pipeline.py` — **low**

`_resolve_image_placeholders` was left behind after the logic moved into
`stages/resolve_images.py`. Removed, so there is only one place to read.

---

### B11 · The doctor's own config check was reading the wrong key — **found by running it**

`context_builder` is nested under `generation:` in `settings.yaml`, so the
retrieved-vs-used comparison read `None` and silently never fired. Fixed to
read `generation.context_builder.max_chunks` (with a fallback to the top-level
key), so the check now really compares `generation.n_results = 5` against
`context_builder.max_chunks = 5`.

---

## Added in this release (the answer to "where are my data issues?")

| New file | What it gives you |
|---|---|
| `rag_pipeline/run_doctor.py` + `doctor.py` | Pre-flight check: Python, settings, dependencies, Ollama reachability, every configured model, your data counts, index consistency, config contradictions — then a verdict and the single next command |
| `rag_pipeline/observability/run_report.py` | Records every pipeline step (in → out, delta, seconds, status, warnings, a real excerpt of your data) and writes it as Markdown |
| `rag_pipeline/observability/health.py` | 13 rules over the chunks you produced, each with severity, share of corpus, why it matters, and how to fix it |
| `rag_pipeline/observability/i18n.py` | Bilingual English/Persian output |
| `rag_pipeline/tests/test_observability.py` | Tests proving the layer records correctly, re-raises real failures, and is completely inert when disabled |
| `OBSERVABILITY.md` | How to run it and how to read every report file |

See `OBSERVABILITY.md`. Report layout:
`reports/run_<timestamp>/{00_SUMMARY.md, NN_<step>.md, DATA_HEALTH.md, RAW_TRACE.json}`.

**Guarantees:** no new dependencies (standard library only), no change to your
chunk output, disabled by default for library callers, `--no-report` to switch
it off, and a broken report can never fail an otherwise good run.

---

## Known issues I did NOT fix (deliberately — tell me and I will)

| # | Issue | Why I left it |
|---|---|---|
| B4 | Re-running the pipeline does not delete stale chunk files; `run_ingestion.py` only upserts. Chunks from a deleted/renamed document stay in Chroma and BM25 forever and can still be retrieved. | Needs a decision from you: wipe-and-rebuild, or track deletions. Both change behavior. |
| B6 | `config.json` + `config_loader.py` are a second, dead configuration system next to `settings.yaml`. | Deleting files is beyond "observe and fix confirmed bugs". |
| B7 | `bm25_index.pkl` uses `pickle` — unsafe to load if the file is ever replaced by someone else. | Changing the on-disk format invalidates existing indexes. |
| B8 | `hybrid_retriever` language filter (`where={"language": …}`) has no fallback: if language detection on the query is wrong, dense retrieval returns nothing. | Needs a product decision (retry without filter? warn?). |
| B9 | `webapp/app.py` latches its startup error forever, has no `/health` endpoint, and runs the Flask dev server. | Deployment change, not a bug fix. |
| — | 9 tests in `tests/test_ragas_evaluator.py` error with `ModuleNotFoundError: ragas` / `langchain_community` when those optional packages are absent. Pre-existing, unrelated to my changes. | Correct fix is `skipUnless(...)` guards so they SKIP instead of ERROR. Say the word. |

---

## Verification / تأیید

- Full suite before my changes: **825 tests, 0 failures, 9 errors** (the 9 are the missing-optional-package errors above).
- Full suite after: same 825 pass **plus** the new observability suite; the 9 environmental errors are unchanged.
- `python rag_pipeline/test_dry_run.py` → `ALL TESTS PASSED ✅`.
- `python rag_pipeline/run_doctor.py` runs end to end and writes `DOCTOR.md`.

**What I could not verify here:** my sandbox has no Ollama and no internet, so
`pip install` and any real model call (embedding, generation, reranking,
Chroma ingestion) must be verified on your machine. The doctor is built exactly
for that moment — run it first and it will tell you what is missing.
