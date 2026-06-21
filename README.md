# RAG Document Intelligence Compiler

This project turns OCR or PDF extracted Markdown into a structured, validated, and reviewable RAG input pipeline.

The workflow is:

1. Read raw Markdown.
2. Normalize and structurally analyze it.
3. Build chunk candidates.
4. Plan semantic chunking.
5. Review each chunk for embedding readiness.
6. Optionally run LLM-based self-healing.

The pipeline is local and deterministic by default. LLM healing is optional.

## What You Get

- `clean.md`: normalized Markdown after Phase 1.
- `index.json`: structured chunk candidate index.
- `chunk_review_report.json`: automated review of each chunk before embedding.
- `result.json`: qualification report from the validation engine.

## Requirements

- Python `>=3.10`
- `pytest` for tests
- Optional: Ollama for LLM healing

On this workspace, use a Python 3.10+ executable explicitly if needed:

```powershell
& 'C:\Program Files\Python312\python.exe' -m pytest -q
```

## End To End Flow

### Step 1. Start from the raw Markdown

Example input:

```text
output_Testing_pdf2markdown\Testing\Testing.md
```

This file is the source Markdown produced by OCR or PDF-to-Markdown tooling.

### Step 2. Build the compiled output

Run the full compiler:

```powershell
python -m cli.main full output_Testing_pdf2markdown\Testing\Testing.md output_Testing_pdf2markdown\Testing\compiled --pretty
```

This creates:

```text
compiled\
  clean.md
  index.json
  chunk_review_report.json
```

`clean.md` is the normalized Markdown.

`index.json` contains:

- `flat_chunk_view`
- `hierarchical_structure_view`
- `semantic_chunk_plan`
- `chunk_review`
- `observability`

`chunk_review_report.json` is the same `chunk_review` section written as a separate file for easier inspection.

### Step 3. Validate the document

Run the qualification engine:

```powershell
python -m rag_doc_compiler.cli process output_Testing_pdf2markdown\Testing\Testing.md --output result.json --verbose
```

This reports:

- `is_chunkable`
- `is_embeddable`
- `is_retrievable`
- `overall_score`
- `structural_score`
- `semantic_score`
- `retrieval_score`
- `embedding_readiness_score`
- `issues`
- `warnings`
- `recommendations`
- `chunking_strategy`

### Step 4. Review chunks before embedding

The compiler automatically reviews every chunk and scores it for embedding readiness.

To inspect the report:

```powershell
Get-Content output_Testing_pdf2markdown\Testing\compiled\chunk_review_report.json
```

Or look at the embedded review inside `index.json`:

```powershell
Get-Content output_Testing_pdf2markdown\Testing\compiled\index.json
```

The review uses operator-style decisions:

- `APPROVE`
- `REPAIR`
- `RECHUNK`
- `REJECT`

### Step 5. Optional self-healing

Before using an LLM, you can do a no-edit dry run:

```powershell
python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --mode semantic-enhance --dry-run
```

If you want actual LLM repair, run:

```powershell
python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --output fixed.md --mode semantic-enhance --scope line
```

Use `--scope context` only when the issue needs multiple lines, such as tables or fences:

```powershell
python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --output fixed.md --mode all --scope context
```

Then run the full compiler again on the fixed file:

```powershell
python -m cli.main full fixed.md final_compiled --pretty
```

## Command Reference

### Full pipeline

```powershell
python -m cli.main full input.md output_dir --pretty
```

### Phase 1 only

```powershell
python -m cli.main run input.md output_dir --pretty
python -m cli.main validate input.md --pretty
python -m cli.main debug input.md --pretty
```

### Phase 2 qualification

```powershell
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
python -m rag_doc_compiler.cli stats input.md
```

### Optional healing

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --dry-run
python -m rag_doc_compiler.cli heal input.md result.json --mode semantic-enhance --dry-run
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --scope line
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --scope context --model qwen2.5:7b
```

## Output Files

### `clean.md`

The normalized Markdown produced by Phase 1. It keeps meaning intact and applies only traceable repairs.

### `index.json`

The chunk index. Important fields:

- `flat_chunk_view`: chunk candidates after sparse merging
- `heading_path`: hierarchical context for each chunk
- `line_range`: source line coverage
- `token_estimate`: approximate size
- `semantic_tags`: extraction tags
- `debug_metadata`: offsets, merge history, normalization flags
- `evaluation_metadata`: coherence and density signals
- `semantic_chunk_plan`: Phase 3 planning output
- `chunk_review`: automated review output

### `chunk_review_report.json`

The chunk review summary written as a standalone file. It contains:

- per-chunk scores
- rule-level issues
- final decision
- recommended action
- overall summary

## Readiness Checklist

A chunk is ready for embedding when it is not rejected by the review layer and has:

- a valid heading path
- parent context
- reasonable token size
- semantic cohesion
- retrievable information
- no major noise artifacts
- stable embedding metadata

A compiled document is ready when the review summary does not show widespread `REJECT` or `RECHUNK` decisions.

## Architecture

```text
adapters/
  file_loader.py               Load input Markdown

core/
  markdown_normalizer.py       Normalize and repair OCR artifacts
  structure_analyzer.py        Build hierarchical document structure
  chunk_preparer.py            Prepare chunk candidates
  semantic_chunk_planner.py    Plan semantic regions before chunking

agents/
  structure_agent.py           Structural validation rules
  semantic_agent.py            Semantic validation rules
  retrieval_agent.py           Retrieval utility rules
  chunk_planning_agent.py      Chunk planning and embedding readiness
  chunk_review_agent.py        Automated chunk review for embedding
  self_healing_agent.py        Optional LLM repair

output/
  markdown_writer.py           Write clean.md
  metadata_exporter.py         Write index.json

cli/
  main.py                      Full pipeline CLI

rag_doc_compiler/
  cli.py                       Validation and healing CLI
```

## Test Coverage

Run everything:

```powershell
python -m pytest -q
```

Run focused suites:

```powershell
python -m pytest tests\test_normalizer.py -q
python -m pytest tests\test_chunk_preparer.py -q
python -m pytest tests\test_semantic_chunk_planner.py -q
python -m pytest tests\test_chunk_review_agent.py -q
python -m pytest tests\test_phase3_pipeline_integration.py -q
python -m pytest tests\test_self_healing_agent.py -q
```

Current verified status:

```text
171 passed
```

## Practical Example

```powershell
$env:PYTHONIOENCODING='utf-8'

python -m cli.main full output_Testing_pdf2markdown\Testing\Testing.md output_Testing_pdf2markdown\Testing\compiled --pretty

python -m rag_doc_compiler.cli process output_Testing_pdf2markdown\Testing\Testing.md --output result.json --verbose

python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --mode semantic-enhance --dry-run

python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --output fixed.md --mode semantic-enhance --scope line

python -m cli.main full fixed.md output_Testing_pdf2markdown\Testing\compiled_fixed --pretty
```

## Notes

- Use `--dry-run` before any LLM healing.
- Prefer `--scope line` first.
- Use `chunk_review_report.json` to decide whether a chunk set is actually safe for embedding.
- The pipeline is designed to keep deterministic processing separate from optional agentic repair.
