# RAG Document Intelligence Compiler

A local pipeline for turning OCR/PDF-generated Markdown into clean, structured, RAG-ready documentation.

The project is designed around two guarantees:

- Preserve the original document meaning. The deterministic pipeline does not summarize or rewrite content.
- Make chunking and embedding safer by validating structure, semantic density, retrieval utility, and chunk readiness before you send content to an embedding model.

## What It Does

The pipeline has three layers of usage:

1. Deterministic compilation
   - Normalizes Markdown safely.
   - Repairs traceable structural problems such as broken ATX heading spacing and common OCR mojibake.
   - Builds a heading-aware document tree.
   - Produces chunk candidates with metadata, line ranges, heading paths, token estimates, and adjacency.
   - Merges sparse adjacent chunks so tiny OCR fragments do not become weak embedding vectors.

2. Qualification and rule validation
   - Runs agentic validators across structure, semantic quality, retrieval utility, chunk planning, and embedding readiness.
   - Produces issues, warnings, scores, recommendations, and a chunking strategy.

3. Optional self-healing
   - Lists exact issue lines without editing by default when used with `--dry-run` or `--no-llm`.
   - Can call a local Ollama model only when the user chooses to run LLM healing.
   - Supports safer line-only fixes or wider context-window fixes.

## Requirements

- Python `>=3.10`
- `pytest` for tests
- Optional: Ollama for LLM-based healing

On this Windows workspace, prefer a real Python 3.10+ executable. Example:

```powershell
& 'C:\Program Files\Python312\python.exe' -m pytest -q
```

If your `python` command already points to Python 3.10+, you can use `python` in all commands below.

## Quick Start

Run the full deterministic compiler and write clean Markdown plus chunk index:

```powershell
python -m cli.main full output_Testing_pdf2markdown\Testing\Testing.md output_Testing_pdf2markdown\Testing\compiled --pretty
```

Outputs:

- `compiled\clean.md`: normalized Markdown
- `compiled\index.json`: chunk candidate index

Run qualification and save the validation report:

```powershell
python -m rag_doc_compiler.cli process output_Testing_pdf2markdown\Testing\Testing.md --output result.json --verbose
```

Check exact fixable issue lines without calling an LLM:

```powershell
python -m rag_doc_compiler.cli heal output_Testing_pdf2markdown\Testing\Testing.md result.json --mode semantic-enhance --dry-run
```

## Recommended Workflow

### 1. Compile the document

Use this when you want the final Markdown and chunk candidate index:

```powershell
python -m cli.main full input.md output_dir --pretty
```

This creates:

```text
output_dir/
  clean.md
  index.json
```

Use `index.json -> flat_chunk_view` as the main input for downstream chunk review, embedding, or vector DB ingestion.

### 2. Validate the document

Use this when you want a quality report:

```powershell
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
```

The report includes:

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

### 3. Review exact issue lines

Before using any LLM, run dry-run healing:

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --mode semantic-enhance --dry-run
```

This prints exact source lines that can be fixed. No model is called and no file is written.

Equivalent no-LLM mode:

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --mode semantic-enhance --no-llm
```

### 4. Optional LLM healing

Install and run Ollama only if you want automatic issue repair:

```powershell
ollama pull qwen2.5:7b
```

Safer line-only healing:

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --mode semantic-enhance --scope line
```

Use wider context only for issues that genuinely need multiple lines, such as broken tables or code fences:

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --mode all --scope context
```

### 5. Re-run validation

Always validate the fixed file:

```powershell
python -m rag_doc_compiler.cli process fixed.md --output final_result.json --verbose
python -m cli.main full fixed.md final_compiled --pretty
```

## Command Reference

### Compile

```powershell
python -m cli.main run input.md output_dir --pretty
python -m cli.main full input.md output_dir --pretty
python -m cli.main validate input.md --pretty
python -m cli.main debug input.md --pretty
```

Use `full` for normal end-to-end compilation.

### Qualify

```powershell
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
python -m rag_doc_compiler.cli process docs_dir --output results_dir --verbose
python -m rag_doc_compiler.cli stats input.md
```

### Optional Healing

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --dry-run
python -m rag_doc_compiler.cli heal input.md result.json --mode semantic-enhance --dry-run
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --scope line
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --scope context --model qwen2.5:7b
```

Healing options:

| Option | Meaning |
| --- | --- |
| `--mode issues` | Use only hard issues. Default. |
| `--mode warnings` | Use warnings only. |
| `--mode all` | Use issues and warnings. |
| `--mode semantic-enhance` | Use issues and warnings for quality improvement. |
| `--dry-run` | Print exact fixable lines; do not call LLM; do not write a file. |
| `--no-llm` | Same safety behavior as dry-run listing. |
| `--scope line` | Replace only the resolved source line. Recommended first. |
| `--scope context` | Replace the extracted context window. Use for complex multi-line repairs. |
| `--model` | Ollama model name. Default: `qwen2.5:7b`. |

## Output Files

### `clean.md`

The normalized Markdown. It preserves original meaning and applies only deterministic, traceable repairs.

### `index.json`

The chunk candidate index. Important fields:

- `flat_chunk_view`: list of chunk candidates for review and embedding preparation
- `hierarchical_structure_view`: tree representation for debugging and visualization
- `heading_path`: full section path for context injection
- `line_range`: original source line range
- `token_estimate`: estimated chunk size
- `content_type`: narrative, technical, procedural, or reference
- `intent`: explanation, definition, list, warning, or example
- `debug_metadata`: source offsets, normalization flags, merge metadata
- `evaluation_metadata`: density and coherence signals

### `result.json`

The qualification report from `rag_doc_compiler.cli process`.

Use:

```powershell
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
```

## Readiness Checklist

A document is ready for chunking/embedding when:

- `is_chunkable` is `true`
- `is_embeddable` is `true`
- `is_retrievable` is `true`
- `issues` is empty or intentionally accepted
- Remaining warnings are understood
- `index.json -> flat_chunk_view` has no very small standalone chunks
- Chunks preserve heading paths and line ranges
- Tables and code blocks are kept with useful context

For the sample file:

```powershell
python -m cli.main full output_Testing_pdf2markdown\Testing\Testing.md output_Testing_pdf2markdown\Testing\compiled_final --pretty
```

Expected healthy chunk audit:

- Chunk count: `11`
- Chunks under 50 estimated tokens: `0`
- Chunks over 350 estimated tokens: `0`

## Architecture

```text
adapters/
  file_loader.py                 Load Markdown input

core/
  markdown_normalizer.py         Safe normalization and OCR repair
  structure_analyzer.py          Heading tree and typed block detection
  chunk_preparer.py              Chunk candidate preparation and sparse merge

agents/
  structure_agent.py             Rules 101-110
  semantic_agent.py              Rules 111-120
  retrieval_agent.py             Rules 121-130
  chunk_planning_agent.py        Rules 131-150
  orchestrator.py                Qualification engine
  self_healing_agent.py          Optional LLM-assisted issue repair

output/
  markdown_writer.py             Writes clean.md
  metadata_exporter.py           Writes index.json

cli/
  main.py                        Deterministic compiler CLI

rag_doc_compiler/
  cli.py                         Qualification and optional healing CLI
```

## Rule Coverage

Phase 1, deterministic document preparation:

- Rules 1-30: preservation, headings, code, tables, lists, links, images, noise handling, whitespace, stable IDs, UTF-8 normalization

Phase 2, qualification:

- Rules 101-110: structural validation
- Rules 111-120: semantic validation
- Rules 121-130: retrieval utility
- Rules 131-140: chunk planning
- Rules 141-150: embedding readiness

## Testing

Run all tests:

```powershell
python -m pytest -q
```

Run focused suites:

```powershell
python -m pytest tests\test_normalizer.py -q
python -m pytest tests\test_chunk_preparer.py -q
python -m pytest tests\test_phase2_qualification.py -q
python -m pytest tests\test_self_healing_agent.py -q
```

Current verified status:

```text
117 passed
```

## Troubleshooting

### Python version errors

The project requires Python 3.10 or newer. If `python --version` shows Python 3.8, use a newer executable:

```powershell
& 'C:\Program Files\Python312\python.exe' -m pytest -q
```

### Windows console Unicode errors

Set UTF-8 output before running verbose commands:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
```

### Dry-run shows many warnings

Warnings do not always mean the emitted chunk candidates are bad. Some warnings are raw-block warnings, while `chunk_preparer.py` may merge those sparse blocks into healthy chunk candidates.

Always inspect:

```text
compiled/index.json -> flat_chunk_view
```

### Ollama errors

Use dry-run first:

```powershell
python -m rag_doc_compiler.cli heal input.md result.json --dry-run
```

If you choose LLM healing, verify Ollama:

```powershell
ollama list
ollama run qwen2.5:7b "hello"
```

## Best Practices

- Run deterministic compilation first.
- Run qualification second.
- Use `heal --dry-run` before any LLM edit.
- Prefer `--scope line` before `--scope context`.
- Re-run validation after every fix.
- Use `index.json -> flat_chunk_view`, not raw Markdown alone, to judge embedding readiness.
- Keep `result.json` and `index.json` as audit artifacts for every processed document.

## Minimal End-To-End Example

```powershell
$env:PYTHONIOENCODING='utf-8'

python -m cli.main full input.md compiled --pretty

python -m rag_doc_compiler.cli process input.md --output result.json --verbose

python -m rag_doc_compiler.cli heal input.md result.json --mode semantic-enhance --dry-run

python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --mode semantic-enhance --scope line

python -m cli.main full fixed.md compiled_fixed --pretty

python -m rag_doc_compiler.cli process fixed.md --output final_result.json --verbose
```
