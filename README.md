# RAG Document Intelligence Compiler

Production-grade ingestion scaffold for converting OCR-generated Markdown into:

1. strictly preserved clean Markdown
2. an advanced hybrid JSON index for RAG orchestration, observability, debugging, visualization, and evaluation
3. high-quality multilingual embeddings using intfloat/multilingual-e5-large with Chonkie-inspired chunking strategy

## Architecture

- `core/markdown_normalizer.py`: preserves Markdown and only performs traceable structural repairs such as ATX heading spacing.
- `core/structure_analyzer.py`: builds a heading-aware tree, flat reference map, and ordered block stream.
- `core/chunk_preparer.py`: creates deterministic, hierarchy-aware chunk candidates without making final chunking decisions.
- `embedding/model_loader.py`: manages downloading and caching of the multilingual-e5-large embedding model locally.
- `embedding/chunk_embedder.py`: generates embeddings for chunks using Chonkie-inspired semantic chunking strategy.
- `adapters/`: isolates file loading and OCR payload compatibility.
- `output/`: writes preserved Markdown and exports the hybrid JSON index.
- `cli/main.py`: provides deterministic `run`, `validate`, `debug`, and `download-model` commands.

## Installation

```bash
pip install -r requirements-dev.txt
```

## CLI

### Basic Commands

```bash
pipeline run input.md output_dir/
pipeline validate input.md
pipeline debug input.md
```

### Embedding Commands

```bash
# Download the embedding model once (cached locally)
pipeline download-model

# Force redownload the model
pipeline download-model --force

# Run with embeddings enabled
pipeline run input.md output_dir/ --embeddings

# Debug with embeddings
pipeline debug input.md --embeddings
```

Without installation:

```bash
python -m cli.main run input.md output_dir/
python -m cli.main download-model
```

## Model Caching

The `intfloat/multilingual-e5-large` model is automatically downloaded and cached in the `./models` directory on first use. Subsequent runs will use the cached model without redownloading. You can also explicitly download it using:

```bash
pipeline download-model
```

This ensures the model is available for offline use and avoids repeated downloads.

## Example Input

```markdown
#RAG Pipeline
RAG is a retrieval augmented generation pattern.

## Steps
- Load OCR Markdown
- Analyze structure
- Prepare chunk candidates
```

## Example Output Files

- `clean.md`: Markdown with only traceable structural normalization.
- `index.json`: hybrid index with `flat_chunk_view` and `hierarchical_structure_view`.

## JSON Schema Summary

Each chunk includes deterministic `chunk_id`, `parent_section_id`, `heading_path`, `line_range`, `content_type`, `intent`, `semantic_tags`, `importance_score`, `token_estimate`, adjacency links, `hierarchy_level`, `debug_metadata`, and `evaluation_metadata`.

The index intentionally excludes vector-store and embedding-model decisions. Downstream retrieval systems can consume `flat_chunk_view` for retrieval and `hierarchical_structure_view` for visualization and evaluation.

## Debugging and Visualization Strategy

- Use `pipeline debug input.md` to print the parsed tree, chunk candidates, and semantic labels.
- Visualizers should render `hierarchical_structure_view.tree.children` as a document tree.
- Chunk quality dashboards should use `evaluation_metadata.density_score`, `chunk_coherence_score`, and line ranges to inspect candidate quality.
- Normalization audits should inspect `debug_metadata.normalization_flags` per chunk.
