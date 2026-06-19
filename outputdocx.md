Implemented the full RAG Document Intelligence Compiler scaffold and validated it.

**What Changed**
- Added strict normalization, structure analysis, and chunk candidate preparation in `core/markdown_normalizer.py:1`, `core/structure_analyzer.py:1`, `core/chunk_preparer.py:1`
- Added shared models/errors and orchestration in `rag_doc_compiler/models.py:1`, `rag_doc_compiler/errors.py:1`, `rag_doc_compiler/pipeline.py:1`
- Added adapters and output exporters in `adapters/file_loader.py:1`, `adapters/ocr_input_adapter.py:1`, `output/markdown_writer.py:1`, `output/metadata_exporter.py:1`
- Added deterministic CLI commands in `cli/main.py:1`
- Added pytest behavior contracts in `tests/test_normalizer.py:1`, `tests/test_structure_analyzer.py:1`, `tests/test_chunk_preparer.py:1`, `tests/test_edge_cases.py:1`
- Added docs/config/dev setup in `README.md:1`, `config/settings.py:1`, `pyproject.toml:1`, `requirements-dev.txt:1`

**Validation**
- Installed `pytest` from `requirements-dev.txt`
- Ran `python -m pytest -q`
- Result: `13 passed in 0.20s`

**Usage**
- Run pipeline: `python -m cli.main run input.md output_dir/`
- Validate only: `python -m cli.main validate input.md`
- Debug structure/chunks/labels: `python -m cli.main debug input.md`

Note: `git status` could not run because this directory’s `.git` entry is not recognized by Git as a repository.

Everything is clear.

The next practical step is to harden this from scaffold to production behavior. I’d proceed in this order:

- Add JSON Schema validation for `index.json`
- Add richer Markdown block parsing contracts for OCR edge cases
- Add snapshot/regression fixtures with real Chandra OCR samples
- Add observability reports for chunk density, hierarchy gaps, and normalization flags
- Add pluggable interfaces for future chunking/evaluation strategies

Send the next requirement or sample OCR Markdown, and I’ll implement it.

Fixed.

**Root Cause**
- `Testing.md:5` had an OCR-glued heading: `...appearance.## Acknowledgements`
- The analyzer missed that `##` section, then saw `### Reviewers and Contributors` at `Testing.md:7`
- That looked like an invalid jump from `#` to `###`, so fail-fast correctly stopped.

**Change**
- Added structural-only normalization for embedded ATX headings in `core/markdown_normalizer.py:1`
- It splits glued headings onto a new line and records `embedded_atx_heading_split`
- Added regression coverage in `tests/test_normalizer.py:1`

**Validation**
- `python -m pytest -q` → `14 passed`
- `python -m cli.main run ./Testing.md ./output_Testing` → succeeded
- Outputs written:
  - `output_Testing/clean.md`
  - `output_Testing/index.json`
  I checked the outputs. They are now structurally valid and usable.

**Checks Passed**
- `output_Testing/clean.md:1` exists and preserves content with traceable structural splits.
- `output_Testing/index.json:1` is valid JSON.
- All mandatory chunk fields are present.
- Chunk adjacency links are consistent.
- Semantic labels are within the allowed sets.
- Max token estimate is now `224`, so no oversized candidate remains.
- HTML tables are now detected as `reference` chunks instead of being swallowed into paragraphs.

**Fix Applied**
- Added OCR-glued HTML table splitting in `core/markdown_normalizer.py:1`
- Added HTML table block detection in `core/structure_analyzer.py:1`
- Added regression tests in `tests/test_normalizer.py:1` and `tests/test_structure_analyzer.py:1`

**Validation**
- `python -m pytest -q` → `16 passed`
- Regenerated:
  - `output_Testing/clean.md`
  - `output_Testing/index.json`

**One Caveat**
- The hierarchy currently treats `## Introduction` as under `# Table of contents`, because that is what the heading levels in the OCR Markdown imply. This is structurally valid, but semantically suspicious. The next hardening step should add an OCR heading-level audit/report rather than silently changing hierarchy.
I checked the actual Chandra output and the regenerated compiler output.

**Status**
- Command succeeded:
  - `python -m cli.main run ./output_Testing_pdf2markdown/Testing.md ./output_Testing`
- Outputs exist:
  - `output_Testing/clean.md:1`
  - `output_Testing/index.json:1`
- JSON index is valid and schema-complete.
- Chunk adjacency is valid.
- No oversized chunks remain.
- HTML tables are now correctly treated as `reference` chunks.

**Structural Repairs Detected**
The compiler made 3 traceable structural-only repairs from the OCR Markdown:

- `output_Testing/clean.md:6` — split glued heading `## Acknowledgements`
- `output_Testing/clean.md:39` — split glued heading `# Table of contents`
- `output_Testing/clean.md:48` — split glued HTML table from paragraph text

These are expected for this Chandra OCR output.

**Important Caveat**
The current hierarchy is technically valid but semantically questionable:

- `# Table of contents`
  - `## Introduction`
  - `## One-shot & few-shot`

This happens because the OCR Markdown itself uses `# Table of contents` followed by `## Introduction`. The compiler preserves the given heading levels instead of silently rewriting them.

**Conclusion**
The output is clear and usable for RAG/debugging now. The next production hardening step should be a heading-level audit layer that reports suspicious OCR hierarchy, especially TOC sections accidentally becoming parents of real document sections.
No.

Your OCR does **text/layout extraction**. The pipeline does **post-OCR document intelligence compilation**.

What Chandra OCR already gives you:
- Markdown text from PDF
- Images extracted to files
- Basic headings/tables if detected
- Some metadata like pages, token counts, image counts
- Raw OCR/vision interpretation

What this pipeline adds:
- Fail-fast validation of broken Markdown structure
- Traceable structural normalization, e.g. glued headings/tables
- Hierarchical document tree
- Flat reference map
- Deterministic chunk candidates
- Chunk IDs, adjacency, heading paths, line ranges
- Semantic labels like `intent` and `content_type`
- Debug metadata and evaluation metadata
- JSON index for RAG orchestration, visualization, and evaluation

In your file, OCR produced useful Markdown, but it also produced issues like:

- `...appearance.## Acknowledgements`
- `...page.# Table of contents`
- `...output.<table border="1">`

The pipeline detected and repaired those structurally, then built `index.json`.

So the correct model is:

`PDF → Chandra OCR → raw Markdown/images/metadata → this compiler → clean Markdown + RAG intelligence index`

OCR is the extractor. This pipeline is the compiler layer after extraction.

No, not unless you change the input Markdown or the pipeline code again.

You already ran it successfully:

`python -m cli.main run ./output_Testing_pdf2markdown/Testing.md ./output_Testing`

Current results are here:

- Clean Markdown: `output_Testing/clean.md:1`
- JSON index: `output_Testing/index.json:1`

Run it again only if:
- `output_Testing_pdf2markdown/Testing.md` changes
- You modify the compiler code
- You want to regenerate `output_Testing/` from scratch

To inspect the current result, use:

`python -m cli.main debug ./output_Testing_pdf2markdown/Testing.md`

Checked `output_Testing`. The rerun output is valid.

**Result**
- `output_Testing/clean.md:1` exists and is structurally cleaned.
- `output_Testing/index.json:1` exists and parses correctly.
- Observability:
  - `25` blocks
  - `25` chunk candidates
  - `9` sections
- Required chunk schema fields: complete.
- Chunk adjacency: valid.
- Semantic labels: valid.
- Max token estimate: `224`.
- Oversized chunks: none.

**Structural Repairs Present**
- `output_Testing/clean.md:6` split glued `## Acknowledgements`
- `output_Testing/clean.md:39` split glued `# Table of contents`
- `output_Testing/clean.md:48` split glued `<table ...>` from paragraph text

**Remaining Issue**
- The hierarchy is still OCR-driven:
  - `# Table of contents`
    - `## Introduction`
    - `## One-shot & few-shot`

This is not a pipeline failure. It reflects the heading levels produced by the OCR Markdown. For strict preservation, the pipeline should not silently rewrite that. The correct next step is to add a `structure_audit` section to `index.json` that flags suspicious hierarchy like “content section nested under Table of contents.”


It means the Markdown structure starts too deep.

In `output_Testing_pdf2markdown/Binder1/Binder1.md:22`, the first heading is:

`### Informes del sistema`

But before that there is no:

- `# ...`
- `## ...`

So the analyzer sees this:

- current hierarchy level: `0` document root
- next heading level: `3`
- jump: `0 → 3`

Because the pipeline is fail-fast, it stops instead of guessing the missing parent sections.

**Why It Happens**
Your OCR output appears to be a document fragment or page/table extract. It starts with a table, then uses `###` headings without their parent `#` / `##` headings.

**This Is Not a Runtime Bug**
It is a structural validation error. The Markdown is valid enough for display, but incomplete for strict hierarchy-aware RAG indexing.

**Options**
- Best production option: add an explicit `structure_audit` mode that reports orphan headings before failing.
- Practical ingestion option: add a configurable policy like `allow_orphan_headings=True`, which creates a synthetic parent section such as `Inferred Section`.
- Strict option: fix the OCR Markdown manually by adding parent headings before line `22`.

For example:

```markdown
# Binder1

## System Configuration

### Informes del sistema
```

Right now the pipeline correctly refuses to infer that silently.


I checked `output_Testing/Binder1`. Initially it was **not fully clear** beyond your two test headings.

**Issue Found**
- Some OCR lines had `</table>` glued to trailing image/text/heading content.
- That caused mixed oversized chunks:
  - one chunk was `903` estimated tokens
  - table/image/next-section content was being grouped incorrectly

**Fixed**
- Added structural split for trailing content after `</table>` in `core/markdown_normalizer.py:1`
- Tightened debug flag attribution in `core/chunk_preparer.py:1`
- Added regression test in `tests/test_normalizer.py:1`
- Regenerated `output_Testing/Binder1`

**Current Binder1 Output**
- `output_Testing/Binder1/clean.md:1` valid
- `output_Testing/Binder1/index.json:1` valid
- `18` chunks
- `9` sections
- Missing schema fields: none
- Bad adjacency: none
- Oversized chunks: none
- Max token estimate is now below threshold
- Tests: `17 passed`

**Remaining Caveat**
Your added headings:

```markdown
# idk
## idk 2
```

make the hierarchy technically valid, but they are artificial. Everything else is now structurally clean for this test output.

Implemented the post-clean Markdown chunking step.

**What Changed**
- Added finalized chunking strategy in `core/chunking_strategy.py:1`
- Added `FinalChunk` and `ChunkingResult` models in `rag_doc_compiler/models.py:1`
- Added `chunks.json` exporter in `output/chunk_exporter.py:1`
- Updated pipeline to run:
  - normalize/clean Markdown
  - analyze structure
  - prepare debug candidates
  - create finalized chunks
  - write `clean.md`, `chunks.json`, `index.json`
- Updated CLI output in `cli/main.py:1`
- Updated docs in `README.md:1`
- Added tests in `tests/test_chunking.py:1`

**New Output**
Running:

`python -m cli.main run input.md output_dir/`

now writes:

- `clean.md`
- `chunks.json`
- `index.json`

**Chunking Behavior**
- Section-level chunks always exist.
- Semantic chunks exist for parsed content blocks.
- Semantic chunks do not overlap.
- Each final chunk includes:
  - `original_text`
  - `embedding_text`
  - `heading_path`
  - `keywords`
  - language metadata
  - provenance
  - adjacency

**Validation**
- `python -m pytest -q` → `21 passed`
- Regenerated:
  - `output_Testing/Testing/chunks.json`
  - `output_Testing/Binder1/chunks.json`
- Binder1 has no oversized final chunks.

I’m rerunning tests and the same query to confirm retrieval is less noisy.

Completed the embedding + hybrid retrieval step.

**Added**
- Embedding profiles: `core/embedding_profiles.py:1`
- Local deterministic embedding engine: `core/embedding_engine.py:1`
- Vector store interface: `core/vector_store.py:1`
- FAISS-compatible local store: `storage/faiss_store.py:1`
- Qdrant-compatible local adapter: `storage/qdrant_store.py:1`
- Graph expansion: `core/graph_connector.py:1`
- Hybrid retrieval engine: `core/retrieval_engine.py:1`
- Reranker: `core/reranker.py:1`
- Agent layer:
  - `agents/query_planner.py:1`
  - `agents/retrieval_agent.py:1`
  - `agents/evidence_selector.py:1`
  - `agents/answer_generator.py:1`
- RAG service API: `api/rag_service.py:1`
- Chunk/graph adapters:
  - `adapters/chunk_interface.py:1`
  - `adapters/graph_input_adapter.py:1`

**CLI**
- Added query command in `cli/main.py:1`:

`python -m cli.main query output_Testing/Testing/chunks.json "What is prompt engineering?"`

**Validation**
- Full suite passes: `28 passed`
- Query output now returns cited evidence, e.g.:
  - `[Source: local_document > Prompt Engineering > sec_1::prompt-engineering::section::0001]`
  - `[Source: local_document > Table of contents > Introduction > sec_8::introduction::definition::0026]`

**Important Note**
- This is local-first and dependency-free. The current embedding engine is deterministic for offline testing/dev.
- Real BGE, sentence-transformers, OpenAI, Jina, or Qdrant client integrations can now be plugged in behind the same interfaces without changing the pipeline.

You configure them as two separate concerns:

1. **Embedding model** — used to convert `chunks.json` into vectors.
2. **LLM model** — used later for answer generation over retrieved evidence.

Right now the project has a local deterministic embedding fallback, but no real downloaded model wired yet.

**Recommended Local Setup**

Use:

- Embedding model: `BAAI/bge-m3`
- LLM model: any local Ollama model, for example `llama3.1:8b` or `qwen2.5:7b`

Install models:

```bash
pip install sentence-transformers
ollama pull llama3.1:8b
```

Then add config like this to `config/settings.py`:

```python
@dataclass(frozen=True)
class ModelSettings:
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
```

**Suggested `.env`**

```bash
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

**How It Should Work**

Embedding path:

```text
chunks.json
→ embedding_text
→ BGE/sentence-transformers
→ vector store
```

LLM path:

```text
query
→ retrieval
→ evidence chunks
→ local LLM
→ cited answer
```

**My Recommendation**
For this project, use:

```bash
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=qwen2.5:7b
```

Reason:
- `bge-m3` is strong for multilingual retrieval.
- `qwen2.5:7b` is good for multilingual reasoning and usually lighter than larger models.

If you want, I can wire this config into the code next: real `sentence-transformers` embedding adapter + Ollama answer generator adapter.


The main adapters already exist in the workspace. I’ll validate and finish docs/CLI behavior for model setup.


Implemented the one-command full pipeline.

**What You Can Run**
```bash
python -m cli.main full-run input workspace --query "What does this document say?"
```

This now does:

- Runs Chandra OCR on every PDF under `input/`
- Saves OCR output to `workspace/output_markdowns/`
- Cleans Markdown structurally
- Builds `clean.md`, `chunks.json`, `index.json`
- Aggregates all documents into `workspace/corpus/chunks.json`
- Embeds with `BAAI/bge-m3` using `sentence-transformers`
- Stores vectors locally in the in-memory FAISS-compatible store
- Retrieves evidence
- Sends evidence to Ollama using your `gemma4` model
- Returns cited answer

**If OCR Already Exists**
```bash
python -m cli.main full-run input workspace --skip-ocr --query "What does this document say?"
```

**Model Config**
Defaults are now:

- Embedding: `BAAI/bge-m3`
- Embedding cache: `models/embeddings`
- Chandra HF cache: `models/chandra`
- LLM provider: `ollama`
- LLM model: `gemma4`

Set overrides if needed:
```bash
export LLM_MODEL=your-gemma-model-name
export EMBEDDING_MODEL=BAAI/bge-m3
```

**Files Added**
- `adapters/chandra_runner.py:1`
- `api/batch_pipeline.py:1`
- `tests/test_batch_pipeline.py:1`
- Updated `cli/main.py:1`
- Updated `config/settings.py:1`
- Rewrote `README.md:1`
- Updated `.env.example:1`

**Validation**
- `python -m pytest -q` → `31 passed`
- `python -m compileall -q ...` → passed
- Verified `full-run --skip-ocr --query ... --no-llm` end-to-end successfully.

No, that output does not mean the pipeline is broken. It was valid JSON, but it was too verbose for normal human use.

I changed the CLI so `full-run` and `query` now print readable output by default.

**Now Default Output Looks Like**
```text
PIPELINE COMPLETE

Raw OCR Markdown: workspace/output_markdowns
Corpus chunks: workspace/corpus/chunks.json

Compiled documents:
- Input: ...
  Clean Markdown: ...
  Chunks: ...
  Index: ...

ANSWER
...

SOURCES
1. Introduction lines 48-85
   chunk: ...
```

**If You Still Want JSON**
Use `--json`:

```bash
python -m cli.main full-run input workspace --skip-ocr --query "what does introduction said?" --json
```

or:

```bash
python -m cli.main query workspace/corpus/chunks.json "what does introduction said?" --json
```

**About Your Answer Quality**
The answer is understandable, but one issue remains from OCR structure:

```text
Table of contents > Introduction
```

That means Chandra OCR made `# Table of contents` a parent heading, then `## Introduction` became nested under it. The pipeline preserves this instead of silently changing document meaning. It is not a crash, but it explains why sources look strange.

**Validation**
- Tests pass: `31 passed`
- Verified `full-run --skip-ocr --query ... --no-llm` prints readable output correctly.

Files changed:
- `cli/main.py:1`
- `README.md:1`

