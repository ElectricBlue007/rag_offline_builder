# Chonkie RAG Library

![PyPI Version](https://img.shields.io/pypi/v/chonkie)
![Downloads](https://img.shields.io/pypi/dm/chonkie)

<div align="center">
  <strong>Advanced Semantic Chunking for RAG</strong>
</div>

---

## Installation

Install Chonkie using pip:

```bash
pip install chonkie
```

Or using uv:

```bash
uv add chonkie
```

### Requirements

- Python 3.8+
- numpy
- transformers

## Usage

### Basic Example

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    embedding_model="intfloat/multilingual-e5-large",
    chunk_size=512
)

text = "Your long document text here..."
chunks = chunker.chunk(text)
```

### Advanced Configuration

You can customize the chunking behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| chunk_size | int | 512 | Maximum tokens per chunk |
| similarity_threshold | float | 0.5 | Minimum semantic similarity |
| min_chunk_size | int | 50 | Minimum tokens per chunk |

## API Reference

### SemanticChunker

The main class for semantic chunking.

#### Methods

##### `chunk(text: str) -> List[Chunk]`

Splits text into semantic chunks.

**Parameters:**
- `text`: The input text to chunk

**Returns:**
- List of Chunk objects

##### `embed(chunks: List[Chunk]) -> np.ndarray`

Generates embeddings for chunks.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Steps

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Last updated: 2024*
