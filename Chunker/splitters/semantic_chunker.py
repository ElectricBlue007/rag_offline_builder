"""
splitters/semantic_chunker.py

Applies semantic chunking WITHIN each structural text section.

Rules:
  - "table" and "image_description" sections are NEVER semantically split —
    they are returned as-is (they are already atomic units).
  - "text" sections below the minimum size threshold skip semantic
    splitting too.
  - All chunks inherit the breadcrumb and metadata of their parent section.
  - Chunks that exceed the maximum size are hard-split at sentence
    boundaries as a safety net.
  - Sizing is TOKEN-based by default (matched to a tokenizer for the
    embedding model via TokenCounter), not character-based — character
    count is an unreliable proxy for token count across languages and
    content types (tables especially). See token_counter.py for full
    rationale. Set SemanticChunkerConfig.use_token_based_sizing=False to
    revert to legacy character-based limits.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from Chunker.config import SemanticChunkerConfig
from Chunker.splitters.structure_splitter import StructuralSection
from Chunker.splitters.token_counter import TokenCounter


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
    # Embedding-ready text. For "text" and "image_description" chunks this
    # equals `content`. For "table" chunks it is the natural-language
    # prose produced by normalizers/table_normalizer.py — `content` keeps
    # the raw HTML for display, this field is what actually gets embedded.
    # Populated by TableChunkNormalizer in the pipeline; defaults to
    # `content` until that stage runs.
    normalized_text: str = ""
    # Contextual-retrieval text: normalized_text with a short situating
    # blurb prepended ("From Binder2.pdf, section 'System reports': ...").
    # This is what actually gets embedded once contextual enrichment runs —
    # normalized_text stays as the "clean" version for other uses (e.g.
    # BM25 indexing may prefer normalized_text without the LLM's framing).
    # Defaults to normalized_text until ContextualEnricher runs.
    contextualized_text: str = ""
    # Final embedding input: contextualized_text with the E5 "passage: "
    # instruction prefix applied (see enrichers/embed_text_builder.py).
    # THIS is the exact string that must be passed to the embedding
    # model — never contextualized_text/normalized_text/content directly,
    # and never a "query: "-prefixed string (that prefix is for search
    # queries at retrieval time only, via prepare_query_text()).
    # Defaults to contextualized_text until EmbedTextBuilder runs.
    embed_text: str = ""


# ── Hard sentence splitter (safety net for oversized chunks) ──────────────────

_RE_SENTENCE = re.compile(r'(?<=[.!?])\s+')


def _hard_split(
    text: str,
    max_units: int,
    count_fn: Callable[[str], int] = len,
) -> list[str]:
    """
    Split `text` at sentence boundaries until every piece is ≤ max_units
    as measured by count_fn (default: character count via len; pass a
    TokenCounter.count_tokens bound method for token-based splitting).
    If a single sentence exceeds max_units it is kept whole (we don't
    break mid-sentence).
    """
    if count_fn(text) <= max_units:
        return [text]

    sentences = _RE_SENTENCE.split(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if current and count_fn(candidate) > max_units:
            chunks.append(current.strip())
            current = sent
        else:
            current = candidate

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
        self._token_counter: Optional[TokenCounter] = None  # lazy-loaded

    # ── Size measurement (token- or char-based, per config) ─────────────────

    @property
    def _count_fn(self) -> Callable[[str], int]:
        """
        The active size-measuring function. Token-based by default —
        returns len() (character count) only when use_token_based_sizing
        is explicitly disabled in config.
        """
        if not self.cfg.use_token_based_sizing:
            return len
        if self._token_counter is None:
            self._token_counter = TokenCounter(
                model_name=self.cfg.tokenizer_hf_id,
                attempt_real_tokenizer=self.cfg.attempt_real_tokenizer,
            )
        return self._token_counter.count_tokens

    @property
    def _min_units(self) -> int:
        return (
            self.cfg.min_chunk_tokens
            if self.cfg.use_token_based_sizing
            else self.cfg.min_chunk_chars
        )

    @property
    def _max_units(self) -> int:
        return (
            self.cfg.max_chunk_tokens
            if self.cfg.use_token_based_sizing
            else self.cfg.max_chunk_chars
        )

    # ── Lazy loader ──────────────────────────────────────────────────────────

    def _get_chunker(self):
        """Build and cache the LangChain SemanticChunker."""
        if self._lc_chunker is not None:
            return self._lc_chunker

        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_community.embeddings.ollama import OllamaEmbeddings
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n"
                "  pip install langchain-experimental langchain-community ollama"
            ) from e

        print(
            f"  [SemanticChunker] Using Ollama embedding model: "
            f"{self.cfg.embedding_model} ({self.cfg.embedding_host})"
        )
        embed_model = OllamaEmbeddings(
            model=self.cfg.embedding_model,
            base_url=self.cfg.embedding_host,
        )

        self._lc_chunker = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type=self.cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.cfg.breakpoint_threshold_amount,
        )
        return self._lc_chunker

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
            and self._count_fn(section.content) < self._min_units
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
            raw_texts = _hard_split(section.content, self._max_units, self._count_fn)

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
        stripped = content.strip()
        return SemanticChunk(
            content=stripped,
            section_type=section.section_type,
            breadcrumb=section.breadcrumb,
            header_levels=list(section.header_levels),
            header_texts=list(section.header_texts),
            source_path=section.source_path,
            # Default: same as content. Table chunks get this overwritten
            # with normalized prose by TableChunkNormalizer later in the
            # pipeline; text/image_description chunks never need it changed.
            normalized_text=stripped,
            # Default: same as content too. Overwritten by ContextualEnricher
            # later in the pipeline (which itself reads from normalized_text,
            # not content, so table chunks get contextualized prose too).
            contextualized_text=stripped,
            # Default: same as content. Overwritten by EmbedTextBuilder
            # (applies the "passage: " prefix on top of contextualized_text).
            embed_text=stripped,
        )

    def _enforce_size_limits(self, texts: list[str]) -> list[str]:
        """
        1. Merge consecutive chunks that are below the minimum size.
        2. Hard-split chunks that exceed the maximum size.
        Sizing uses self._count_fn (tokens by default, chars if
        use_token_based_sizing=False) and self._min_units / self._max_units.
        """
        count_fn = self._count_fn
        min_units = self._min_units
        max_units = self._max_units

        # Pass 1: hard-split oversized
        split: list[str] = []
        for t in texts:
            split.extend(_hard_split(t, max_units, count_fn))

        # Pass 2: merge undersized (greedy, left-to-right)
        merged: list[str] = []
        buffer = ""
        for t in split:
            if count_fn(t) < min_units:
                buffer = (buffer + " " + t).strip() if buffer else t
            else:
                if buffer:
                    # Try to attach the buffer to this chunk if it fits
                    candidate = (buffer + " " + t).strip()
                    if count_fn(candidate) <= max_units:
                        t = candidate
                        buffer = ""
                    else:
                        merged.append(buffer)
                        buffer = ""
                merged.append(t)

        if buffer:
            if merged:
                candidate = (merged[-1] + " " + buffer).strip()
                if count_fn(candidate) <= max_units:
                    merged[-1] = candidate
                else:
                    merged.append(buffer)
            else:
                merged.append(buffer)

        return merged
