# Project Architecture Map

This document is a debugging map for the RAG Document Intelligence Compiler. Use it when a bug appears and you need to find the exact module responsible.

## Big Picture

The project turns OCR/PDF-extracted Markdown into cleaned Markdown, structured blocks, chunk candidates, semantic chunk plans, chunk review reports, and optional LLM-repaired chunks.

Main command:

```powershell
python -m cli.main full input.md output_dir --pretty
```

Optional chunk LLM repair:

```powershell
python -m cli.main full input.md output_dir --repair-chunks --repair-model llama3.2:3b --repair-max-attempts 2 --pretty
```

Core outputs:

```text
output_dir/
  clean.md                  normalized markdown
  index.json                chunks and full structure
  chunk_review_report.json  embedding-readiness review
  chunk_repair_report.json  optional LLM chunk repair audit
```

## End-To-End Pipeline

```text
input.md
  |
  v
adapters.file_loader.FileLoader
  |
  v
core.markdown_normalizer.MarkdownNormalizer
  |
  v
core.structure_analyzer.StructureAnalyzer
  |
  +--> core.semantic_chunk_planner.SemanticChunkPlanner
  |
  v
core.chunk_preparer.ChunkPreparer
  |
  v
output.metadata_exporter.MetadataExporter
  |
  v
agents.chunk_review_agent.ChunkReviewAgent
  |
  +--> optional agents.chunk_repair_agent.ChunkRepairAgent
  |       |
  |       +--> agents.rule_repair_prompt_agent.RuleRepairPromptAgent
  |       +--> local Ollama model or injected llm_client
  |       +--> rerun ChunkReviewAgent
  |
  v
output.markdown_writer.MarkdownWriter
output.metadata_exporter.MetadataExporter.write
```

The orchestrator for this flow is:

```text
rag_doc_compiler.pipeline.DocumentCompiler
```

## Main Orchestration

### `rag_doc_compiler/pipeline.py`

Class: `DocumentCompiler`

This is the main compiler object used by `cli/main.py`.

Responsibilities:

- load Markdown from disk
- normalize it
- analyze structure
- create semantic chunk plan
- prepare chunk candidates
- build `index.json`
- run chunk review
- optionally run LLM chunk repair
- write `clean.md`, `index.json`, `chunk_review_report.json`, and optionally `chunk_repair_report.json`

Debug here when:

- output files are missing
- `index.json` is missing expected sections
- chunk repair is not running
- CLI output paths are wrong
- `chunk_review_report.json` does not match `index.json["chunk_review"]`

## CLI Entry Points

### `cli/main.py`

Primary CLI:

```powershell
python -m cli.main full input.md output_dir --pretty
python -m cli.main run input.md output_dir --pretty
python -m cli.main debug input.md --pretty
python -m cli.main qualify input.md --pretty
```

Important functions:

- `build_parser`: defines CLI commands and flags
- `build_compiler_from_args`: passes `--repair-chunks`, model, and retry options into `DocumentCompiler`
- `run_phase1`: runs compiler and returns paths
- `run_phase2`: runs the qualification engine
- `main`: command dispatch

Debug here when:

- command-line arguments are not recognized
- `--repair-chunks` is ignored
- JSON printed to terminal is wrong
- command exits with the wrong status code

### `rag_doc_compiler/cli.py`

Legacy/secondary CLI for document qualification and Markdown healing.

Important commands:

```powershell
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md --scope context --model llama3.2:3b
python -m rag_doc_compiler.cli stats input.md
```

Debug here when:

- `heal` command does not map issues to lines
- `process` output JSON is malformed
- document-level LLM repair does not write `fixed.md`

## Data Models

### `rag_doc_compiler/models.py`

Defines Phase 1 and chunk candidate data structures.

Important models:

- `NormalizationResult`: normalized Markdown plus normalization flags
- `DocumentNode`: tree node used by structure analysis
- `DocumentBlock`: structured block with type-specific metadata
- `StructureAnalysis`: complete tree, flat map, ordered blocks
- `ChunkCandidate`: prepared chunk before embedding

Debug here when:

- serialized model fields are missing
- `index.json` schema needs to change
- chunk fields like `heading_path`, `line_range`, or `token_estimate` are wrong

### `agents/models.py`

Defines Phase 2 qualification models.

Important models:

- `ValidationIssue`
- `IssueSeverity`
- `IssueType`
- `SectionAnalysis`
- `ChunkingStrategy`
- `DocumentQualificationResult`

Debug here when:

- Phase 2 issue/warning format is wrong
- qualification result cannot serialize
- rule severity/type data is inconsistent

## Input Layer

### `adapters/file_loader.py`

Class: `FileLoader`

Loads UTF-8 Markdown from disk.

Debug here when:

- input path is rejected
- file cannot be read
- wrong encoding is suspected

### `adapters/ocr_input_adapter.py`

Class: `OCRInputAdapter`

Extracts Markdown from OCR payload shapes:

- raw string
- `{"markdown": "..."}`
- `{"text": "..."}`
- `{"content": "..."}`

Debug here when:

- OCR provider output is a dict and Markdown is not found
- OCR pipeline changes its payload keys

## Core Phase 1 Modules

### `core/markdown_normalizer.py`

Class: `MarkdownNormalizer`

First real cleanup step. It preserves meaning and performs structural repairs.

Responsibilities:

- normalize Unicode and OCR mojibake
- split glued headings and glued HTML tables
- remove presentation noise, badges, table-of-contents artifacts, continuation markers
- preserve code blocks
- preserve front matter
- normalize whitespace
- produce `NormalizationResult.flags_by_line`

Debug here when:

- `clean.md` still contains OCR corruption like mojibake
- headings are glued to paragraphs
- table of contents or decorative layout text appears in chunks
- code fences break during cleanup
- too much original content disappears before chunking

Tests:

```powershell
python -m pytest tests\test_normalizer.py -q
python -m pytest tests\test_rag_rules_comprehensive.py -q
```

### `core/structure_analyzer.py`

Class: `StructureAnalyzer`

Builds the document tree and block list from normalized Markdown.

Responsibilities:

- parse headings
- enforce heading hierarchy
- consume paragraphs, lists, quotes, code blocks, Markdown tables, HTML tables
- create `DocumentNode` tree
- create typed `DocumentBlock` entries

Debug here when:

- section tree is wrong
- heading paths are wrong
- blocks have wrong `line_range`
- HTML table or code block consumes too much text
- `hierarchical_structure_view` is wrong

Tests:

```powershell
python -m pytest tests\test_structure_analyzer.py -q
python -m pytest tests\test_edge_cases.py -q
```

### `core/semantic_chunk_planner.py`

Classes: `SemanticChunkPlanner`, `SemanticRegion`, `ChunkPlan`, `PlannedBlock`

Plans semantic chunking before final chunk generation.

Responsibilities:

- build semantic regions before chunking
- treat headings as soft boundaries
- identify anchors for code, tables, lists, quotes, FAQ, examples
- score density, confidence, retrieval utility
- preserve parent context, heading path, entity/alias hints
- create multi-pass chunk plan metadata

Debug here when:

- `semantic_chunk_plan` is missing or wrong
- H1/H2/H3 merge decisions are wrong
- anchors around table/code/list/example are wrong
- tiny sections are not expanded
- context injection planning is wrong

Tests:

```powershell
python -m pytest tests\test_semantic_chunk_planner.py -q
python -m pytest tests\test_semantic_chunk_planner_rules_201_236.py -q
```

### `core/chunk_preparer.py`

Class: `ChunkPreparer`

Creates actual chunk candidates in `flat_chunk_view`.

Responsibilities:

- convert `StructureAnalysis.ordered_blocks` into `ChunkCandidate`
- assign chunk ids
- estimate tokens
- classify intent and content type
- add semantic tags
- calculate density/coherence metadata
- merge sparse adjacent chunks
- filter OCR boilerplate/decorative chunks
- build adjacency links

Debug here when:

- chunks are too tiny
- chunks contain author-only, TOC, acknowledgements, or decorative OCR text
- sparse sections are not merging
- `flat_chunk_view` has wrong `chunk_id`
- chunk `text` is wrong
- adjacency is wrong

Tests:

```powershell
python -m pytest tests\test_chunk_preparer.py -q
```

## Output Layer

### `output/markdown_writer.py`

Class: `MarkdownWriter`

Writes normalized Markdown to:

```text
output_dir/clean.md
```

Debug here when:

- `clean.md` is not written
- output directory is not created

### `output/metadata_exporter.py`

Class: `MetadataExporter`

Builds and writes `index.json`.

Main index fields:

- `compiler`
- `outputs`
- `flat_chunk_view`
- `hierarchical_structure_view`
- `semantic_chunk_plan`
- `chunk_review`
- optional `chunk_repair_report`
- `observability`

Debug here when:

- chunks are created but not saved
- `index.json` structure is wrong
- `flat_chunk_view` is missing
- report fields need to be added to index

## Chunk Review And Repair

### `agents/chunk_review_agent.py`

Class: `ChunkReviewAgent`

Reviews chunks after they are created and decides whether they are ready for embedding.

Rule groups:

- `301-310`: structural checks
- `311-315`: token checks
- `316-325`: semantic checks
- `326-332`: retrieval checks
- `333-339`: noise detection
- `340-343`: context quality
- `344-350`: embedding-specific checks

Output:

```text
chunk_review_report.json
index.json["chunk_review"]
```

Debug here when:

- a chunk is incorrectly `APPROVE`, `REPAIR`, `RECHUNK`, or `REJECT`
- score is too harsh or too weak
- E5 `passage:` prefix validation is wrong
- duplicate/noise/context rules are wrong

Tests:

```powershell
python -m pytest tests\test_chunk_review_agent.py -q
```

### `agents/chunk_repair_agent.py`

Class: `ChunkRepairAgent`

Optional LLM repair for chunks in `index.json`.

Responsibilities:

- read `index["chunk_review"]`
- send selected decisions to LLM, default `REPAIR,RECHUNK`
- skip `REJECT` by default
- use rule-specific prompts
- validate model output
- replace chunk text in `flat_chunk_view`
- rerun `ChunkReviewAgent`
- write `chunk_repair_report`

Debug here when:

- `--repair-chunks` does not change `index.json`
- LLM output is rejected
- repaired chunk is worse than original
- `chunk_repair_report.json` is missing
- Ollama encoding or timeout fails

Tests:

```powershell
python -m pytest tests\test_chunk_repair_agent.py -q
```

### `agents/rule_repair_prompt_agent.py`

Class: `RuleRepairPromptAgent`

Builds rule-specific LLM prompts for both Markdown healing and chunk repair.

Covers:

- rules `1-70`
- validation rules `101-150`
- chunk planning rules `201-236`
- chunk review rules `301-350`

Debug here when:

- LLM prompts are too generic
- a rule needs a better repair instruction
- a new rule id appears in reports
- prompt should target `clean.md` vs `index.json`

Tests:

```powershell
python -m pytest tests\test_rule_repair_prompt_agent.py -q
```

### `agents/self_healing_agent.py`

Class: `SelfHealingAgent`

Optional LLM repair for source/fixed Markdown files based on validation reports.

Responsibilities:

- read `result.json` or qualification output
- map issues/warnings to source lines
- extract context window
- build rule-specific prompt
- call Ollama or injected `llm_client`
- validate model output
- replace line/context in output Markdown

Debug here when:

- `heal` finds no fixable items
- issue line mapping is wrong
- LLM output is skipped
- fixed Markdown is not written
- `--scope line` or `--scope context` behaves incorrectly

Tests:

```powershell
python -m pytest tests\test_self_healing_agent.py -q
```

### `agents/smart_structure_fixer.py`

Class: `SmartStructureFixer`

Deterministic structural fixer used by the legacy CLI `--auto-fix`.

Debug here when:

- non-LLM structural auto-fix is expected
- headings/code/tables are repaired without LLM

## Phase 2 Qualification Agents

Phase 2 is separate from chunk review. It qualifies the whole structured document for chunking, embedding, and retrieval.

Orchestrator:

```text
agents.orchestrator.DocumentQualificationEngine
```

### `agents/structure_agent.py`

Class: `StructureAgent`

Rules `101-110`.

Checks:

- heading level gaps
- heading hierarchy
- empty headings
- section length
- document order
- code fences
- tables
- links
- images

Debug here when:

- Phase 2 says document structure is broken
- heading hierarchy warning is wrong
- broken table/link/image warning is wrong

### `agents/semantic_agent.py`

Class: `SemanticAgent`

Rules `111-120`.

Checks:

- empty or near-empty blocks
- low density
- boilerplate
- navigation
- footer/header noise
- badges
- repeated content

Debug here when:

- boilerplate/noise warnings are wrong
- useful content is marked as noise
- repeated content detection is too aggressive

### `agents/retrieval_agent.py`

Class: `RetrievalAgent`

Rules `121-130`.

Checks:

- answerability
- definitions
- procedures
- API docs
- requirements
- configuration
- troubleshooting
- FAQ
- references
- cross references

Debug here when:

- document is incorrectly marked not retrievable
- definitions/procedures/API content are not detected
- cross references are not captured

### `agents/chunk_planning_agent.py`

Classes: `ChunkPlanningAgent`, `EmbeddingReadinessAgent`

Rules:

- `131-140`: chunk planning and anchoring
- `141-150`: embedding readiness

Checks:

- code/table/list/quote/example anchoring
- heading/parent/sibling context
- continuations
- topic boundaries
- min/max semantic mass
- entity/keyword density
- self-sufficiency
- retrieval utility

Debug here when:

- Phase 2 recommends wrong chunking strategy
- document is marked not embeddable
- anchor recommendations are wrong

### `agents/orchestrator.py`

Class: `DocumentQualificationEngine`

Runs all Phase 2 agents and aggregates:

- issues
- warnings
- scores
- recommendations
- `ChunkingStrategy`

Debug here when:

- `overall_score` looks wrong
- issue/warning counts are wrong
- final `is_chunkable`, `is_embeddable`, or `is_retrievable` is wrong
- recommended chunking strategy is wrong

Tests:

```powershell
python -m pytest tests\test_phase2_qualification.py -q
```

## Configuration

### `config/settings.py`

Class: `CompilerSettings`

Current settings:

- `max_candidate_tokens`
- `min_density_for_high_importance`
- `token_chars_per_token`
- `allow_header_spacing_normalization`

Debug here when:

- token estimates feel wrong
- chunk merging threshold depends on max token size
- heading spacing normalization should be disabled/enabled

## Errors

### `rag_doc_compiler/errors.py`

Error classes:

- `CompilerError`
- `NormalizationError`
- `StructureError`
- `ChunkPreparationError`

Debug here when:

- you need clearer error categories
- CLI catches an error but message is too vague

## Where Chunks Are Saved

Chunks are saved in:

```text
output_dir/index.json
```

Inside:

```json
{
  "flat_chunk_view": [
    {
      "chunk_id": "...",
      "text": "...",
      "heading_path": ["..."],
      "token_estimate": 123
    }
  ]
}
```

If chunk repair is enabled, repaired chunks replace the same `flat_chunk_view` entries in `index.json`.

## Debugging By Symptom

### `clean.md` contains OCR junk

Check:

```text
core/markdown_normalizer.py
tests/test_normalizer.py
```

Typical issues:

- mojibake not normalized
- TOC not removed
- continuation marker still present
- decorative image caption remains

### Heading path is wrong

Check:

```text
core/structure_analyzer.py
rag_doc_compiler/models.py
tests/test_structure_analyzer.py
```

### Chunks are too small or contain boilerplate

Check:

```text
core/chunk_preparer.py
tests/test_chunk_preparer.py
```

### Semantic chunk plan is wrong

Check:

```text
core/semantic_chunk_planner.py
tests/test_semantic_chunk_planner.py
tests/test_semantic_chunk_planner_rules_201_236.py
```

### Chunk is rejected incorrectly

Check:

```text
agents/chunk_review_agent.py
tests/test_chunk_review_agent.py
```

### LLM chunk repair is bad

Check:

```text
agents/chunk_repair_agent.py
agents/rule_repair_prompt_agent.py
tests/test_chunk_repair_agent.py
tests/test_rule_repair_prompt_agent.py
```

### LLM Markdown healing is bad

Check:

```text
agents/self_healing_agent.py
agents/rule_repair_prompt_agent.py
rag_doc_compiler/cli.py
tests/test_self_healing_agent.py
```

### Phase 2 says document is not good enough

Check:

```text
agents/orchestrator.py
agents/structure_agent.py
agents/semantic_agent.py
agents/retrieval_agent.py
agents/chunk_planning_agent.py
tests/test_phase2_qualification.py
```

### Output file is missing

Check:

```text
rag_doc_compiler/pipeline.py
output/markdown_writer.py
output/metadata_exporter.py
cli/main.py
```

### CLI command fails before pipeline starts

Check:

```text
cli/main.py
rag_doc_compiler/cli.py
```

## Test Map

Run all tests:

```powershell
python -m pytest -q
```

Focused tests:

```powershell
python -m pytest tests\test_normalizer.py -q
python -m pytest tests\test_structure_analyzer.py -q
python -m pytest tests\test_chunk_preparer.py -q
python -m pytest tests\test_semantic_chunk_planner.py -q
python -m pytest tests\test_semantic_chunk_planner_rules_201_236.py -q
python -m pytest tests\test_chunk_review_agent.py -q
python -m pytest tests\test_chunk_repair_agent.py -q
python -m pytest tests\test_rule_repair_prompt_agent.py -q
python -m pytest tests\test_self_healing_agent.py -q
python -m pytest tests\test_phase2_qualification.py -q
python -m pytest tests\test_phase3_pipeline_integration.py -q
python -m pytest tests\test_edge_cases.py -q
python -m pytest tests\test_rag_rules_comprehensive.py -q
```

## Current Known Output Example

The OCR sample output created during debugging is:

```text
OutputOCR/compiled_operator_fixed/
  clean.md
  index.json
  chunk_review_report.json
```

For that sample, the chunk review summary was:

```text
chunk_count: 5
decisions: APPROVE 5
average_final_score: 96.0
ready_for_embedding: True
```
