"""
normalizers/chunk_normalizer.py

Pipeline stage that applies table_normalizer.py to every "table"-type
SemanticChunk, populating its `normalized_text` field with natural-
language prose while leaving `content` (the raw HTML) untouched.

Runs AFTER semantic chunking (so tables are already atomic single
chunks) and BEFORE metadata enrichment.

Text and image_description chunks are left alone — their
normalized_text already equals content from SemanticChunker._wrap().
"""

from dataclasses import dataclass
from typing import Optional

from rag_pipeline.normalizers.table_normalizer import (
    normalize_table_html,
    TableNormalizationError,
)
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


@dataclass
class TableNormalizerConfig:
    # Master switch — if False, table chunks keep normalized_text == content
    # (i.e. the raw HTML gets embedded as-is, matching pre-normalization
    # behaviour). Useful for A/B comparing retrieval quality with/without
    # normalization.
    enabled: bool = True

    # If a table fails to parse (malformed HTML), fall back to using the
    # raw HTML content as normalized_text rather than raising and halting
    # the whole pipeline run. Set to False to fail loudly instead.
    fallback_to_raw_on_error: bool = True

    # Print a one-line warning to stdout when a table fails to normalize.
    warn_on_fallback: bool = True


class TableChunkNormalizer:
    """
    Usage:
        normalizer = TableChunkNormalizer(config)
        chunks = normalizer.normalize_chunks(chunks)
    """

    def __init__(self, config: Optional[TableNormalizerConfig] = None):
        self.cfg = config or TableNormalizerConfig()
        self._fallback_count = 0

    def normalize_chunks(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """
        Return a new list where every "table" chunk has normalized_text
        set to its natural-language prose form. Non-table chunks pass
        through unchanged.
        """
        if not self.cfg.enabled:
            return chunks

        for chunk in chunks:
            if chunk.section_type != "table":
                continue
            chunk.normalized_text = self._normalize_one(chunk)

        return chunks

    def _normalize_one(self, chunk: SemanticChunk) -> str:
        try:
            result = normalize_table_html(chunk.content)
            if result.strip():
                return result
            # Parsed successfully but produced nothing usable — still
            # fall back rather than embedding an empty string.
            return self._fallback(chunk, reason="empty normalization result")
        except TableNormalizationError as exc:
            return self._fallback(chunk, reason=str(exc))
        except Exception as exc:
            # Any unexpected parsing error — never let a single malformed
            # table halt the whole pipeline run.
            return self._fallback(chunk, reason=f"unexpected error: {exc}")

    def _fallback(self, chunk: SemanticChunk, reason: str) -> str:
        self._fallback_count += 1
        if not self.cfg.fallback_to_raw_on_error:
            raise TableNormalizationError(
                f"Table normalization failed ({reason}) and "
                f"fallback_to_raw_on_error=False."
            )
        if self.cfg.warn_on_fallback:
            preview = chunk.content[:60].replace("\n", " ")
            print(
                f"  [TableChunkNormalizer] WARNING: falling back to raw "
                f"HTML for table chunk ({reason}). Preview: {preview}..."
            )
        return chunk.content
