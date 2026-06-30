"""
splitters/semantic_chunker.py

Applies semantic chunking WITHIN each structural text section.

Rules:
  - "table" and "image_description" sections are NEVER semantically split —
    they are returned as-is (they are already atomic units).
  - "text" sections below skip_if_small threshold skip semantic splitting too.
  - All chunks inherit the breadcrumb and metadata of their parent section.
  - Chunks that exceed max_chunk_chars are hard-split at sentence boundaries
    as a safety net.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from Chunker.config import SemanticChunkerConfig
from Chunker.splitters.structure_splitter import StructuralSection


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class SemanticChunk:
    """A single chunk ready to be embedded and stored."""
    content: str
    section_type: str       # "text" | "table" | "image_description"
    breadcrumb: str
    header_levels: list[int]
    header_texts: list[str]
    source_path: str
    chunk_index: int = 0    # filled in later by the pipeline


# ── Hard sentence splitter (safety net for oversized chunks) ──────────────────

_RE_SENTENCE = re.compile(r'(?<=[.!?])\s+')


def _hard_split(text: str, max_chars: int) -> list[str]:
    """
    Split `text` at sentence boundaries until every piece is ≤ max_chars.
    If a single sentence exceeds max_chars it is kept whole (we don't break
    mid-sentence).
    """
    if len(text) <= max_chars:
        return [text]

    sentences = _RE_SENTENCE.split(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current:
        chunks.append(current.strip())

    return chunks


# ── Semantic chunker ──────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Wraps LangChain's SemanticChunker to apply it section-by-section.

    Lazy-loads the embedding model and LangChain components on first use
    so that importing this module has zero startup cost.
    """

    def __init__(self, config: Optional[SemanticChunkerConfig] = None):
        self.cfg = config or SemanticChunkerConfig()
        self._lc_chunker = None     # lazy-loaded

    # ── Lazy loader ──────────────────────────────────────────────────────────

    def _get_chunker(self):
        """Build and cache the LangChain SemanticChunker."""
        if self._lc_chunker is not None:
            return self._lc_chunker

        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n"
                "  pip install langchain-experimental langchain-community fastembed"
            ) from e

        self._validate_model_name(TextEmbedding)

        print(f"  [SemanticChunker] Loading embedding model: {self.cfg.embedding_model}")
        embed_model = FastEmbedEmbeddings(model_name=self.cfg.embedding_model)

        self._lc_chunker = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type=self.cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.cfg.breakpoint_threshold_amount,
        )
        return self._lc_chunker

    def _validate_model_name(self, text_embedding_cls) -> None:
        """
        Check the configured embedding_model against FastEmbed's own
        supported-model list BEFORE attempting to load it. This turns a
        confusing pydantic ValidationError (raised deep inside
        FastEmbedEmbeddings) into one clear, actionable message.
        """
        try:
            supported = {
                m.get("model", "") for m in text_embedding_cls.list_supported_models()
            }
        except Exception:
            # If we can't even list supported models (e.g. no network yet),
            # don't block — let the real load attempt fail/succeed normally.
            return

        if supported and self.cfg.embedding_model not in supported:
            example = next(iter(sorted(supported)), "BAAI/bge-small-en-v1.5")
            raise ValueError(
                f"Embedding model '{self.cfg.embedding_model}' is not "
                f"supported by FastEmbed's TextEmbedding class.\n"
                f"Fix this in rag_pipeline/config.py "
                f"(SemanticChunkerConfig.embedding_model), e.g.:\n"
                f"  embedding_model: str = \"{example}\"\n"
                f"Full supported list: TextEmbedding.list_supported_models()"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def chunk_section(self, section: StructuralSection) -> list[SemanticChunk]:
        """
        Convert one StructuralSection into one or more SemanticChunks.
        """
        # Tables and image descriptions: always return as single atomic chunk
        if section.section_type in ("table", "image_description"):
            return [self._wrap(section.content, section)]

        # Text sections smaller than the threshold: skip semantic splitting
        if (
            self.cfg.skip_semantic_if_small
            and len(section.content) < self.cfg.min_chunk_chars
        ):
            content = section.content.strip()
            return [self._wrap(content, section)] if content else []

        # Apply semantic chunking
        try:
            chunker = self._get_chunker()
            lc_docs = chunker.create_documents([section.content])
            raw_texts = [doc.page_content for doc in lc_docs]
        except Exception as exc:
            # Graceful degradation: fall back to hard split
            print(f"  [SemanticChunker] WARNING: semantic split failed ({exc}), "
                  "falling back to hard split.")
            raw_texts = _hard_split(section.content, self.cfg.max_chunk_chars)

        # Post-process: enforce min/max size
        final_texts = self._enforce_size_limits(raw_texts)

        return [self._wrap(t, section) for t in final_texts if t.strip()]

    def chunk_sections(self, sections: list[StructuralSection]) -> list[SemanticChunk]:
        """Process a list of sections and return flat list of SemanticChunks."""
        all_chunks: list[SemanticChunk] = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section))
        # Assign sequential indices
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        return all_chunks

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _wrap(self, content: str, section: StructuralSection) -> SemanticChunk:
        return SemanticChunk(
            content=content.strip(),
            section_type=section.section_type,
            breadcrumb=section.breadcrumb,
            header_levels=list(section.header_levels),
            header_texts=list(section.header_texts),
            source_path=section.source_path,
        )

    def _enforce_size_limits(self, texts: list[str]) -> list[str]:
        """
        1. Merge consecutive chunks that are below min_chunk_chars.
        2. Hard-split chunks that exceed max_chunk_chars.
        """
        # Pass 1: hard-split oversized
        split: list[str] = []
        for t in texts:
            split.extend(_hard_split(t, self.cfg.max_chunk_chars))

        # Pass 2: merge undersized (greedy, left-to-right)
        merged: list[str] = []
        buffer = ""
        for t in split:
            if len(t) < self.cfg.min_chunk_chars:
                buffer = (buffer + " " + t).strip() if buffer else t
            else:
                if buffer:
                    # Try to attach the buffer to this chunk if it fits
                    if len(buffer) + len(t) + 1 <= self.cfg.max_chunk_chars:
                        t = (buffer + " " + t).strip()
                        buffer = ""
                    else:
                        merged.append(buffer)
                        buffer = ""
                merged.append(t)

        if buffer:
            if merged:
                candidate = merged[-1] + " " + buffer
                if len(candidate) <= self.cfg.max_chunk_chars:
                    merged[-1] = candidate.strip()
                else:
                    merged.append(buffer)
            else:
                merged.append(buffer)

        return merged
