# RAG Document Compiler - Complete Usage Guide

A production-ready pipeline for processing, validating, and automatically fixing large markdown documents for RAG (Retrieval-Augmented Generation) systems.

## 🚀 Quick Start

### Basic Workflow

```bash
# Step 1: Process and validate your document
python -m rag_doc_compiler.cli process your_document.md --output result.json --verbose

# Step 2: Auto-fix structural issues (headings, hierarchy, noise)
python -m rag_doc_compiler.cli process your_document.md \
  --output result.json \
  --auto-fix \
  --output-md fixed_document.md \
  --verbose

# Step 3: Advanced healing with local LLM (optional)
python -m rag_doc_compiler.cli heal fixed_document.md result.json --output perfect_document.md
```

## 📋 Command Reference

### `process` - Main Pipeline Command

Runs Phase 1 (Normalization & Structuring) and Phase 2 (Qualification).

**Basic Usage:**
```bash
python -m rag_doc_compiler.cli process input.md
```

**With Output Files:**
```bash
python -m rag_doc_compiler.cli process input.md --output results.json
```

**With Auto-Fix (Recommended):**
```bash
python -m rag_doc_compiler.cli process input.md \
  --output results.json \
  --output-md fixed.md \
  --auto-fix \
  --verbose
```

**Process Directory of Files:**
```bash
python -m rag_doc_compiler.cli process ./docs/ --output ./results/ --auto-fix
```

**Options:**
- `input`: Input markdown file or directory (required)
- `-o, --output`: Output JSON file or directory
- `-v, --verbose`: Show detailed logs
- `--auto-fix`: Auto-fix structural issues using smart fixer
- `--output-md`: Output path for fixed markdown (used with --auto-fix)

### `stats` - Document Statistics

```bash
python -m rag_doc_compiler.cli stats your_document.md
```

Shows:
- Original and normalized length
- Total blocks detected
- Block type distribution
- Front matter presence

### `heal` - Advanced LLM-Based Healing

Uses local Ollama models to fix complex issues.

**Prerequisites:**
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:7b
```

**Usage:**
```bash
python -m rag_doc_compiler.cli heal input.md issues.json --output fixed.md
```

**Options:**
- `input`: Input markdown file to fix (required)
- `issues`: JSON file with issues from 'process' command (required)
- `-o, --output`: Output fixed markdown file
- `-m, --model`: Ollama model name (default: qwen2.5:7b)

## 🔧 Complete Workflow Examples

### Example 1: OCR-Processed Document

Your document from OCR has no headings and poor structure:

```bash
# Process with auto-fix
python -m rag_doc_compiler.cli process \
  output_Testing_pdf2markdown/Testing/Testing.md \
  --output output_TestingNew/Testing/result.json \
  --output-md output_TestingNew/Testing/fixed_clean.md \
  --auto-fix \
  --verbose
```

**What happens:**
1. ✅ Detects missing headings
2. ✅ Injects appropriate heading levels
3. ✅ Fixes hierarchy gaps
4. ✅ Removes badges and navigation noise
5. ✅ Saves fixed version
6. ✅ Re-evaluates and shows improved scores

### Example 2: Batch Processing Multiple Files

```bash
# Process all markdown files in a directory
python -m rag_doc_compiler.cli process \
  ./documentation/ \
  --output ./qualified_results/ \
  --auto-fix \
  --verbose
```

Output files:
- `./qualified_results/file1_qualified.json` - Qualification report
- `./qualified_results/file1_fixed.md` - Fixed markdown

### Example 3: Two-Stage Healing

For maximum quality, use both smart fixer and LLM healer:

```bash
# Stage 1: Smart structural fix (fast, rule-based)
python -m rag_doc_compiler.cli process \
  input.md \
  --output stage1_result.json \
  --output-md stage1_fixed.md \
  --auto-fix \
  --verbose

# Stage 2: LLM healing (slower, context-aware)
python -m rag_doc_compiler.cli heal \
  stage1_fixed.md \
  stage1_result.json \
  --output final_perfect.md \
  --model qwen2.5:7b
```

## 📊 Understanding Output

### JSON Report Structure

```json
{
  "source_file": "your_document.md",
  "phase_1_structure": {
    "metadata": {
      "total_blocks": 25,
      "has_front_matter": false
    },
    "blocks": [...]
  },
  "phase_2_qualification": {
    "is_chunkable": true,
    "is_embeddable": true,
    "is_retrievable": true,
    "overall_score": 0.85,
    "structural_score": 0.90,
    "semantic_score": 0.80,
    "retrieval_score": 0.85,
    "embedding_readiness_score": 0.82,
    "chunking_strategy": {
      "strategy_type": "adaptive_semantic",
      "min_chunk_size": 100,
      "max_chunk_size": 512
    },
    "issues": [...],
    "warnings": [...],
    "recommendations": [...]
  }
}
```

### Score Interpretation

| Score Range | Status | Action |
|-------------|--------|--------|
| 0.9 - 1.0 | Excellent | Ready for production |
| 0.8 - 0.9 | Good | Minor improvements optional |
| 0.7 - 0.8 | Fair | Review recommended |
| < 0.7 | Poor | Auto-fix required |

### Common Issues and Fixes

#### Issue: No Headings Detected
**Symptom:** Score = 0.00, Block distribution shows only paragraphs
**Solution:** Use `--auto-fix` flag
```bash
python -m rag_doc_compiler.cli process input.md --auto-fix --output-md fixed.md
```

#### Issue: Broken Heading Hierarchy
**Symptom:** Jumps from H1 to H3, missing H2
**Solution:** Auto-fix handles this automatically

#### Issue: Navigation Noise
**Symptom:** Low semantic score, "navigation" in recommendations
**Solution:** Auto-fix removes badges and nav elements

#### Issue: Unclosed Code Fences
**Symptom:** Structural validation errors
**Solution:** 
```bash
# Smart fixer handles simple cases
python -m rag_doc_compiler.cli process input.md --auto-fix

# For complex cases, use LLM healer
python -m rag_doc_compiler.cli heal input.md issues.json
```

## 🎯 Best Practices

### 1. Always Use Auto-Fix First

The smart structural fixer is fast and handles 80% of common issues:

```bash
# ✅ Recommended
python -m rag_doc_compiler.cli process input.md --auto-fix --output-md fixed.md

# ❌ Not recommended (manual fixing required)
python -m rag_doc_compiler.cli process input.md
```

### 2. Use Verbose Mode for First Run

See exactly what's being fixed:

```bash
python -m rag_doc_compiler.cli process input.md --auto-fix --verbose
```

### 3. Check Fixed Output Before LLM Healing

Review the auto-fixed document:

```bash
# After auto-fix
cat fixed.md

# If still issues, use LLM healer
python -m rag_doc_compiler.cli heal fixed.md result.json
```

### 4. Batch Process with Directory Output

For multiple files:

```bash
python -m rag_doc_compiler.cli process \
  ./source_docs/ \
  --output ./processed_results/ \
  --auto-fix \
  --verbose
```

### 5. Integrate into CI/CD Pipeline

```yaml
# Example GitHub Actions step
- name: Validate Documentation
  run: |
    python -m rag_doc_compiler.cli process docs/ \
      --output results/ \
      --auto-fix \
      --verbose
    
    # Fail if score below threshold
    python -c "
    import json
    with open('results/report.json') as f:
        data = json.load(f)
    score = data['phase_2_qualification']['overall_score']
    if score < 0.7:
        exit(1)
    "
```

## 🔍 Troubleshooting

### Problem: "Input path does not exist"

**Solution:** Check file path is correct
```bash
# Verify file exists
ls -la your_document.md

# Use absolute path if needed
python -m rag_doc_compiler.cli process /full/path/to/document.md
```

### Problem: Fixed File Not Created

**Solution:** Ensure `--output-md` is specified with `--auto-fix`
```bash
# Correct
python -m rag_doc_compiler.cli process input.md --auto-fix --output-md fixed.md

# Incorrect (no output path specified)
python -m rag_doc_compiler.cli process input.md --auto-fix
```

### Problem: Score Still Low After Auto-Fix

**Solution:** Use LLM healer for complex issues
```bash
# First auto-fix
python -m rag_doc_compiler.cli process input.md --auto-fix --output-md fixed.md --output result.json

# Then LLM healing
python -m rag_doc_compiler.cli heal fixed.md result.json --output perfect.md
```

### Problem: Ollama Connection Error

**Solution:** Ensure Ollama is running and model is pulled
```bash
# Start Ollama service
ollama serve

# In another terminal, pull model
ollama pull qwen2.5:7b

# Verify
curl http://localhost:11434/api/tags
```

## 📈 Performance Tips

### Large Documents (100k+ lines)

The pipeline automatically handles large documents:

```bash
# No special configuration needed
python -m rag_doc_compiler.cli process large_document.md --auto-fix
```

**Processing Time Estimates:**
- 10k lines: ~30 seconds
- 50k lines: ~2 minutes  
- 100k lines: ~5 minutes

### Memory Optimization

For very large documents:

```bash
# Process in batches
find ./docs -name "*.md" | while read file; do
    python -m rag_doc_compiler.cli process "$file" --auto-fix
done
```

## 🛡️ Privacy & Security

- **100% Local Processing**: Smart fixer runs entirely offline
- **Optional LLM**: Only `heal` command uses local Ollama (no cloud APIs)
- **No Data Upload**: Your documents never leave your machine
- **Open Source**: Full transparency of all processing logic

## 📚 Additional Resources

- [RAG Pipeline Guide](RAG_PIPELINE_GUIDE.md) - Complete rule documentation
- [Phase 2 Summary](PHASE2_SUMMARY.md) - Qualification engine details
- [70 Rules Documentation](RAG_PIPELINE_GUIDE.md) - All validation rules

## 💡 Pro Tips

1. **Run on Raw OCR Output**: Don't manually clean OCR results - let the pipeline do it
2. **Save JSON Reports**: Keep qualification reports for audit trails
3. **Compare Before/After**: Run stats on original and fixed to see improvement
4. **Custom Thresholds**: Set your own quality thresholds for your use case
5. **Iterative Improvement**: Run multiple passes if needed

---

**Ready to get started?**

```bash
# Your first command - replace with your file path
python -m rag_doc_compiler.cli process your_document.md \
  --output result.json \
  --output-md fixed.md \
  --auto-fix \
  --verbose
```
