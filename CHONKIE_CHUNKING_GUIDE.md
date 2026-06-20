# Chonkie-Based Multilingual Chunking Pipeline Guide

## Overview

This document describes the rebuilt chunking step in your RAG pipeline that uses:
- **Chonkie**: A modern semantic chunking library
- **intfloat/multilingual-e5-large** (or alternatives): State-of-the-art multilingual embeddings

## Architecture

### Components

1. **MultilingualEmbeddingBackend** (`core/chunking_strategy.py`)
   - Wraps sentence-transformers models for Chonkie compatibility
   - Supports lazy loading and caching of embedding models
   - Implements Chonkie's `BaseEmbeddings` interface

2. **ChonkieChunkingStrategy** (`core/chunking_strategy.py`)
   - Main chunking strategy using Chonkie's SemanticChunker
   - Creates hierarchy-aware chunks with section headers
   - Applies semantic similarity-based chunking within sections
   - Enriches chunks with metadata, keywords, and language detection

## Installation

```bash
# Install required dependencies
pip install chonkie>=1.6.0 sentence-transformers>=3.0

# Or from requirements file
pip install -r requirements-ml.txt
```

## Recommended Embedding Models

### Primary Recommendation
- **`intfloat/multilingual-e5-large`** 
  - Dimension: 1024
  - Disk: ~2.2GB
  - RAM: ~2GB during inference
  - Best for: Production quality, 100+ languages

### Alternatives (if resources are limited)

1. **`intfloat/multilingual-e5-small`**
   - Dimension: 384
   - Disk: ~470MB
   - RAM: ~1GB
   - Good balance of quality and resources

2. **`BAAI/bge-m3`**
   - Dimension: 1024
   - Disk: ~2.2GB
   - Excellent multilingual performance with retrieval optimization

3. **`minishlab/potion-base-32M`**
   - Dimension: 32
   - Disk: ~32MB
   - Ultra-lightweight for testing/development

## Usage

### Basic Usage

```python
from core.structure_analyzer import StructureAnalyzer
from core.chunking_strategy import ChonkieChunkingStrategy
from config.settings import DEFAULT_SETTINGS, DEFAULT_MODEL_SETTINGS

# Analyze document structure
analyzer = StructureAnalyzer()
analysis = analyzer.analyze(markdown_text)

# Create chunking strategy with desired embedding model
chunker = ChonkieChunkingStrategy(
    settings=DEFAULT_SETTINGS,
    model_settings=DEFAULT_MODEL_SETTINGS,
    embedding_model='intfloat/multilingual-e5-large',  # or other model
    threshold=0.7,        # Semantic similarity threshold (0.5-0.8)
    chunk_size=512,       # Maximum tokens per chunk
    min_sentences_per_chunk=1,
    similarity_window=3,
)

# Generate chunks
result = chunker.chunk(analysis)

print(f"Generated {len(result.chunks)} chunks")
print(f"Strategy: {result.strategy['name']}")
```

### Advanced Configuration

```python
chunker = ChonkieChunkingStrategy(
    settings=DEFAULT_SETTINGS,
    model_settings=DEFAULT_MODEL_SETTINGS,
    
    # Embedding model configuration
    embedding_model='intfloat/multilingual-e5-large',
    
    # Semantic chunking parameters
    threshold=0.7,              # Lower = more chunks, Higher = fewer chunks
    chunk_size=512,             # Max tokens per chunk
    min_sentences_per_chunk=1,  # Minimum sentences before forcing break
    similarity_window=3,        # Sentences to consider for similarity
    
    # Optional overlap (for context preservation)
    include_overlap=False,
    overlap_size=50,
)
```

### Environment Variables

Configure via environment variables in `.env` or shell:

```bash
# Embedding configuration
export EMBEDDING_PROVIDER=sentence_transformers
export EMBEDDING_MODEL=intfloat/multilingual-e5-large
export EMBEDDING_CACHE_DIR=models/embeddings

# For lighter setup
# export EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

## How It Works

### 1. Document Structure Analysis
- Parses Markdown to identify sections, headings, and content blocks
- Builds hierarchical tree structure
- Extracts heading paths for context

### 2. Section-Level Chunking
- Creates dedicated chunks for each section header
- Enables hierarchy navigation in retrieval
- Preserves document structure

### 3. Semantic Chunking with Chonkie
Within each section:
1. **Sentence Splitting**: Text split into sentences using delimiters
2. **Embedding Generation**: Each sentence embedded using multilingual E5
3. **Similarity Computation**: Semantic similarity between adjacent sentences
4. **Chunk Boundary Detection**: Breaks where similarity drops below threshold
5. **Chunk Assembly**: Groups sentences into coherent semantic units

### 4. Chunk Enrichment
Each chunk includes:
- **Metadata**: IDs, line ranges, parent relationships
- **Content Type**: narrative, technical, procedural, reference
- **Intent**: definition, explanation, list, warning, example
- **Keywords**: Lexical and semantic keyword extraction
- **Language Detection**: Automatic language identification
- **Embedding Text**: Optimized text for vector embedding
- **Adjacency**: Links to previous/next chunks
- **Provenance**: Source tracking and coherence scores

## Integration with Your Pipeline

Your existing pipeline flow:

```
PDF → Chandra OCR → Markdown Cleaning → [NEW CHUNKING] → Embedding → Vector Store
```

### Example Integration

```python
from adapters.chandra_runner import ChandraOCRRunner
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from core.chunking_strategy import ChonkieChunkingStrategy
from core.embedding_engine import build_embedding_engine

# Step 1: OCR extraction
ocr_runner = ChandraOCRRunner()
raw_text = ocr_runner.extract_from_pdf("document.pdf")

# Step 2: Clean Markdown
normalizer = MarkdownNormalizer()
normalized = normalizer.normalize(raw_text)

# Step 3: Structure analysis
analyzer = StructureAnalyzer()
analysis = analyzer.analyze(normalized.markdown)

# Step 4: Semantic chunking (NEW)
chunker = ChonkieChunkingStrategy(
    embedding_model='intfloat/multilingual-e5-large',
    threshold=0.7,
    chunk_size=512,
)
chunking_result = chunker.chunk(analysis)

# Step 5: Generate embeddings
embedding_engine = build_embedding_engine(
    provider='sentence_transformers',
    settings=model_settings,
)

for chunk in chunking_result.chunks:
    embedding = embedding_engine.embed_text(chunk.embedding_text)
    # Store in vector DB...
```

## Configuration Options

### Similarity Threshold
- **0.5-0.6**: More granular chunks, better for detailed queries
- **0.7-0.8** (default): Balanced chunking, good general purpose
- **0.8-0.9**: Larger chunks, better for broad context queries

### Chunk Size
- **256-384**: Smaller chunks, faster retrieval, less context
- **512** (default): Balanced size
- **768-1024**: Larger chunks, more context, slower retrieval

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Production RAG | `intfloat/multilingual-e5-large` | Best accuracy |
| Limited RAM | `intfloat/multilingual-e5-small` | Good quality/size ratio |
| Retrieval-focused | `BAAI/bge-m3` | Optimized for retrieval tasks |
| Testing/Dev | `minishlab/potion-base-32M` | Fast, minimal resources |
| English-only | `BAAI/bge-base-en-v1.5` | Specialized for English |

## Benefits Over Previous Approach

### Previous (HierarchyAwareChunkingStrategy)
- Rule-based chunk boundaries
- No semantic understanding
- Fixed chunk sizes
- Limited multilingual support

### New (ChonkieChunkingStrategy)
- ✅ Semantic similarity-based boundaries
- ✅ Context-aware chunking
- ✅ Adaptive chunk sizes
- ✅ Full multilingual support (100+ languages)
- ✅ Better retrieval quality
- ✅ Coherence scoring
- ✅ Flexible configuration

## Troubleshooting

### Memory Issues
If you encounter OOM errors:
1. Use smaller model: `intfloat/multilingual-e5-small`
2. Reduce batch sizes in embedding
3. Increase swap space
4. Process documents sequentially

### Slow Performance
1. Ensure GPU acceleration if available
2. Use model caching (enabled by default)
3. Adjust `similarity_window` parameter
4. Consider `minishlab/potion-base-32M` for development

### Poor Chunk Quality
1. Adjust `threshold` parameter (try 0.6-0.8 range)
2. Check input Markdown quality
3. Verify language detection is correct
4. Tune `chunk_size` for your content type

## Testing

```python
# Test with sample document
test_markdown = """
# Test Document

## Introduction
This is a test document with multiple sections.

## Features
- Feature 1: Description
- Feature 2: Details

## Conclusion
Summary here.
"""

analyzer = StructureAnalyzer()
analysis = analyzer.analyze(test_markdown)

chunker = ChonkieChunkingStrategy(
    embedding_model='minishlab/potion-base-32M',  # Lightweight for testing
    threshold=0.5,
    chunk_size=256,
)

result = chunker.chunk(analysis)
assert len(result.chunks) > 0
print(f"✓ Generated {len(result.chunks)} chunks successfully")
```

## References

- [Chonkie Documentation](https://github.com/chonkie-ai/chonkie)
- [Multilingual E5 Models](https://huggingface.co/intfloat/multilingual-e5-large)
- [Sentence Transformers](https://www.sbert.net/)

## Support

For issues or questions:
1. Check memory/disk space availability
2. Verify model download completed successfully
3. Review Chonkie and sentence-transformers documentation
4. Check environment variable configuration
