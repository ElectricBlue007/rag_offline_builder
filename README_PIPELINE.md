# 🚀 RAG Document Compiler - Complete Pipeline Guide

A state-of-the-art RAG ingestion pipeline with **70+ rules** across 7 layers for perfect markdown processing, qualification, and self-healing.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Overview](#pipeline-overview)
3. [Phase 1: Normalization & Structuring](#phase-1-normalization--structuring)
4. [Phase 2: Qualification Engine](#phase-2-qualification-engine)
5. [Phase 3: Self-Healing](#phase-3-self-healing)
6. [Complete Workflow Example](#complete-workflow-example)
7. [Rules Reference](#rules-reference)

---

## 🏃 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pytest

# Install Ollama (for self-healing)
# Visit https://ollama.ai and download for your OS

# Pull the recommended model
ollama pull qwen2.5:7b
```

### Basic Usage

```bash
# Step 1: Process document (Phase 1 + Phase 2)
python -m rag_doc_compiler.cli process your_document.md --output result.json --verbose

# Step 2: Auto-fix issues (Phase 3 - Self-Healing)
python -m rag_doc_compiler.cli heal your_document.md result.json --output fixed.md

# Step 3: Verify the fix
python -m rag_doc_compiler.cli process fixed.md --output final_result.json --verbose
```

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Document Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Normalization & Structuring (Rules 1-30)          │
│  ├─ UTF-8 Normalization                                     │
│  ├─ Noise Removal (HTML, Badges)                            │
│  ├─ Block Detection (Headings, Code, Tables, Lists)         │
│  └─ Structure Analysis                                      │
│                                                              │
│  Phase 2: Qualification Engine (Rules 101-150)              │
│  ├─ StructureAgent: Validate hierarchy, broken markdown     │
│  ├─ SemanticAgent: Detect density, boilerplate, navigation  │
│  ├─ RetrievalAgent: Identify definitions, procedures, APIs  │
│  ├─ ChunkPlanningAgent: Anchoring, context, boundaries      │
│  └─ EmbeddingReadinessAgent: Scoring & strategy             │
│                                                              │
│  Phase 3: Self-Healing (Automated Fixing)                   │
│  ├─ Load issues from Phase 2                                │
│  ├─ Extract context around each issue                       │
│  ├─ Query local LLM for surgical fixes                      │
│  └─ Apply fixes bottom-up to preserve line numbers          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Phase 1: Normalization & Structuring

**Goal:** Convert raw markdown into a structured document without losing information.

### Commands

```bash
# Process a single file
python -m rag_doc_compiler.cli process input.md --output result.json

# Process a directory
python -m rag_doc_compiler.cli process ./docs/ --output ./results/

# View statistics only
python -m rag_doc_compiler.cli stats input.md
```

### Output Structure

```json
{
  "source_file": "input.md",
  "phase_1_structure": {
    "metadata": {
      "total_blocks": 25,
      "has_front_matter": true
    },
    "blocks": [
      {
        "block_id": "h1-abc123",
        "order": 0,
        "block_type": "heading",
        "content": "# Installation",
        "level": 1,
        "heading_path": ["Installation"]
      }
    ]
  }
}
```

### Key Features (Rules 1-30)

- ✅ Preserve original meaning (no summarization)
- ✅ Maintain heading hierarchy and levels
- ✅ Extract heading paths for retrieval
- ✅ Preserve code blocks with language metadata
- ✅ Store tables as structured JSON
- ✅ Preserve list nesting
- ✅ Separate link text and URLs
- ✅ Convert images to metadata
- ✅ Remove HTML/badge noise
- ✅ Normalize whitespace
- ✅ Assign stable IDs and order
- ✅ Detect boilerplate
- ✅ UTF-8 normalization for multilingual content

---

## 🔍 Phase 2: Qualification Engine

**Goal:** Assess document quality and generate chunking strategies.

### Architecture

```
┌──────────────────────────────────────────┐
│       DocumentQualificationEngine        │
├──────────────────────────────────────────┤
│  StructureAgent   → Rules 101-110       │
│  SemanticAgent    → Rules 111-120       │
│  RetrievalAgent   → Rules 121-130       │
│  ChunkPlanningAgent → Rules 131-140     │
│  EmbeddingReadinessAgent → Rules 141-150│
└──────────────────────────────────────────┘
```

### Commands

```bash
# Run full qualification (included in 'process' command)
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
```

### Output Example

```json
{
  "phase_2_qualification": {
    "is_chunkable": true,
    "is_embeddable": true,
    "is_retrievable": true,
    "overall_score": 0.85,
    "structural_score": 0.90,
    "semantic_score": 0.80,
    "retrieval_score": 0.85,
    "embedding_readiness_score": 0.88,
    "chunking_strategy": {
      "strategy_type": "semantic_with_anchoring",
      "min_chunk_size": 100,
      "max_chunk_size": 512,
      "context_injection": true,
      "parent_context_depth": 2
    },
    "issues": [
      {
        "rule_id": "Rule 103",
        "severity": "warning",
        "message": "Empty heading detected",
        "location": {"line": 45},
        "recommendation": "Add content under this heading or merge with parent"
      }
    ],
    "recommendations": [
      "Document is well-structured and ready for advanced chunking"
    ]
  }
}
```

### Rules Summary

#### Layer A: Structural Validation (101-110)
- Rule 101: Heading level gaps
- Rule 102: Heading hierarchy breaks
- Rule 103: Empty headings
- Rule 104: Sections too small
- Rule 105: Sections too large
- Rule 106: Document order validation
- Rule 107: Broken code fences
- Rule 108: Broken tables
- Rule 109: Broken links
- Rule 110: Broken image references

#### Layer B: Semantic Validation (111-120)
- Rule 111: Empty semantic blocks
- Rule 112: Near-empty blocks
- Rule 113: Low information density
- Rule 114: Boilerplate detection
- Rule 115: Navigation detection
- Rule 116: Footer detection
- Rule 117: Header noise
- Rule 118: Badge detection
- Rule 119: Repeated content
- Rule 120: Duplicate sections

#### Layer C: Retrieval Validation (121-130)
- Rule 121: Answerability check
- Rule 122: Definition detection
- Rule 123: Procedure detection
- Rule 124: API documentation
- Rule 125: Requirements detection
- Rule 126: Configuration detection
- Rule 127: Troubleshooting detection
- Rule 128: FAQ detection
- Rule 129: Reference detection
- Rule 130: Cross-reference detection

#### Layer D: Chunk Planning (131-140)
- Rule 131: Code anchoring
- Rule 132: Table anchoring
- Rule 133: List anchoring
- Rule 134: Quote anchoring
- Rule 135: Example anchoring
- Rule 136: Heading context
- Rule 137: Parent context
- Rule 138: Sibling context
- Rule 139: Continuation detection
- Rule 140: Topic boundary candidates

#### Layer E: Embedding Readiness (141-150)
- Rule 141: Minimum semantic mass
- Rule 142: Maximum semantic mass
- Rule 143: Entity density
- Rule 144: Keyword density
- Rule 145: Definition presence
- Rule 146: Reference integrity
- Rule 147: Self-sufficiency score
- Rule 148: Context dependency score
- Rule 149: Retrieval utility score
- Rule 150: Embedding readiness score

---

## 🩺 Phase 3: Self-Healing

**Goal:** Automatically fix structural issues using local LLMs without rewriting content.

### How It Works

1. **Bottom-Up Processing**: Fixes are applied from the end of the document to the beginning to preserve line numbers
2. **Context Extraction**: Only 20 lines around each issue are sent to the model
3. **Surgical Fixes**: The model generates only the corrected snippet, not the whole document
4. **Content Preservation**: Strict prompts prevent rewriting of technical content

### Commands

```bash
# Basic healing
python -m rag_doc_compiler.cli heal input.md issues.json --output fixed.md

# With custom model
python -m rag_doc_compiler.cli heal input.md issues.json --output fixed.md --model llama3:8b

# Default output name (input_fixed.md)
python -m rag_doc_compiler.cli heal input.md issues.json
```

### Requirements

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (recommended: qwen2.5:7b for reasoning)
ollama pull qwen2.5:7b

# Alternative models
ollama pull llama3:8b
ollama pull mistral:7b
```

### Example Session

```bash
# Step 1: Process and find issues
$ python -m rag_doc_compiler.cli process Testing.md --output result.json --verbose
🚀 Starting Phase 1: Normalization & Structuring...
   ✅ Normalized content length: 6842 chars
   ✅ Detected 25 blocks
   📊 Block distribution: {'paragraph': 22, 'table': 3}
   ⚠️  Issues found: 5
      - [warning] Rule 103: Empty heading detected
      - [error] Rule 107: Unclosed code fence

# Step 2: Auto-fix issues
$ python -m rag_doc_compiler.cli heal Testing.md result.json --output Testing_fixed.md
🩺 Starting Self-Healing Process...
📄 Input: Testing.md
📋 Issues: result.json
🔍 Found 5 issues to fix.

[1/5] Fixing Rule 107 at line 145...
   🤖 Consulting qwen2.5:7b...
   ✅ Applied fix.

[2/5] Fixing Rule 103 at line 89...
   🤖 Consulting qwen2.5:7b...
   ✅ Applied fix.

🎉 Healing Complete!
   Fixed: 5/5 issues
   Output saved to: Testing_fixed.md

# Step 3: Verify
$ python -m rag_doc_compiler.cli process Testing_fixed.md --verbose
✅ Qualification Score: 0.95
✅ Chunkable: True
✅ Embeddable: True
✅ Retrievable: True
```

---

## 📊 Complete Workflow Example

### Scenario: Processing OCR-Extracted Documentation

```bash
# 1. Process the raw OCR output
python -m rag_doc_compiler.cli process \
  output_Testing_pdf2markdown/Testing/Testing.md \
  --output output_TestingNew/Testing/result.json \
  --verbose

# 2. Review the issues (optional: open result.json)
cat output_TestingNew/Testing/result.json | jq '.phase_2_qualification.issues'

# 3. Auto-fix all issues
python -m rag_doc_compiler.cli heal \
  output_Testing_pdf2markdown/Testing/Testing.md \
  output_TestingNew/Testing/result.json \
  --output output_TestingNew/Testing/clean_fixed.md

# 4. Verify the fixed document
python -m rag_doc_compiler.cli process \
  output_TestingNew/Testing/clean_fixed.md \
  --output output_TestingNew/Testing/final_result.json \
  --verbose

# 5. Check improvement
echo "Original Score: $(jq '.phase_2_qualification.overall_score' output_TestingNew/Testing/result.json)"
echo "Fixed Score: $(jq '.phase_2_qualification.overall_score' output_TestingNew/Testing/final_result.json)"
```

### Expected Output

```
Original Score: 0.45
Fixed Score: 0.92
```

---

## 📚 Rules Reference

### Top 10 Golden Rules for RAG

1. **Rule 48**: Inject heading path into chunks (`Chonkie > Installation > Pip`)
2. **Rule 34-35**: Semantic density scoring
3. **Rule 31**: Remove navigation poison
4. **Rule 50**: Preserve parent context
5. **Rule 41**: Keep code + explanation together
6. **Rule 42**: Keep table + explanation together
7. **Rule 59**: Adaptive chunk sizing
8. **Rule 60**: Semantic cohesion scoring
9. **Rule 57**: Similarity valley detection
10. **Rule 63**: Query-oriented chunk design

### All 70 Rules by Category

| Phase | Rules | Focus |
|-------|-------|-------|
| **Fidelity** | 1-30 | Preserve meaning, structure, code, tables, links |
| **Structure** | 101-110 | Validate hierarchy, detect broken markdown |
| **Semantic** | 111-120 | Density, boilerplate, navigation, badges |
| **Retrieval** | 121-130 | Definitions, procedures, APIs, troubleshooting |
| **Planning** | 131-140 | Anchoring, context, boundaries |
| **Embedding** | 141-150 | Scoring, mass, readiness |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_phase2_qualification.py -v
python -m pytest tests/test_rag_rules_comprehensive.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## 🛠 Troubleshooting

### Issue: Score is 0.00

**Cause:** Document has major structural problems.

**Solution:**
```bash
# Check detailed issues
python -m rag_doc_compiler.cli process input.md --output result.json --verbose
cat result.json | jq '.phase_2_qualification.issues[] | {rule, message, recommendation}'

# Auto-fix
python -m rag_doc_compiler.cli heal input.md result.json --output fixed.md
```

### Issue: Ollama connection error

**Solution:**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# macOS/Linux:
brew services restart ollama
# Windows: Restart Ollama from system tray

# Test model
ollama run qwen2.5:7b "Hello"
```

### Issue: Large documents (>100k lines)

**Solution:** The pipeline handles large documents efficiently because:
- Phase 1 processes line-by-line
- Phase 2 analyzes blocks, not raw text
- Phase 3 only sends 20-line contexts to the LLM

No special handling needed!

---

## 📈 Performance Tips

1. **Use `qwen2.5:7b`** for best reasoning on structural fixes
2. **Run Phase 1+2 first** to identify issues before healing
3. **Review top 5 issues** manually if score < 0.5
4. **Batch process** directories for multiple files
5. **Cache results** to avoid re-processing unchanged files

---

## 🎯 Next Steps

After completing all 3 phases:

1. **Chunking** (Future Phase 4): Use the generated strategy
2. **Embedding**: Apply `intfloat/multilingual-e5-large` or similar
3. **Indexing**: Store in vector database
4. **Retrieval**: Query with context injection

---

**Built with ❤️ for state-of-the-art RAG pipelines**
