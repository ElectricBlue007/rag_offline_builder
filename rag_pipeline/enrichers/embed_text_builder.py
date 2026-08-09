"""
enrichers/embed_text_builder.py

Single source of truth for the embedding model's instruction-prefix
contract — the exact prefix strings live in EmbedTextConfig
(config.py), so switching embedding models means changing them in
ONE place, not hunting through the codebase.

Many embedding models are trained with instruction prefixes: E5-family
models (e.g. the previous intfloat/multilingual-e5-large) expect every
STORED passage to start with "passage: " and every SEARCH QUERY with
"query: ". The current qwen3-embedding models use a different contract
— no passage prefix, an instruction-style query prefix only (see
EmbedTextConfig). Skipping a prefix a model expects — or swapping two
prefixes — runs the model outside its training distribution. Cosine
similarity scores still come back looking plausible, so this failure
mode produces no error and no warning; it just silently ranks results
in the wrong order. This is documented in each model's card and is one
of the most common mistakes when standing up a new embedding model.

This module is the ONLY place either prefix string should be defined,
for exactly that reason: one source of truth means it's impossible for
the storage side and the query side to drift out of sync.

  - EmbedTextBuilder   → pipeline stage, storage side. Applies
                         "passage: " to every chunk's contextualized_text,
                         producing the final embed_text field.
  - prepare_query_text → free function, query side. The future retriever
                         module (roadmap step 6) must call this — and
                         ONLY this — to prefix a user's search query
                         before embedding it. Never hand-roll "query: "
                         + text elsewhere in the codebase.
"""

from dataclasses import dataclass
from typing import Optional

from rag_pipeline.config import EmbedTextConfig
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


class EmbedTextBuilder:
    """
    Usage:
        builder = EmbedTextBuilder(config)
        chunks = builder.build(chunks)
    """

    def __init__(self, config: Optional[EmbedTextConfig] = None):
        self.cfg = config or EmbedTextConfig()

    def build(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """
        Populate embed_text for every chunk: passage_prefix + contextualized_text.
        If disabled via config, embed_text becomes an exact copy of
        contextualized_text (no prefix) — matches the "off" pattern used
        by TableChunkNormalizer/ContextualEnricher for consistency.
        """
        prefix = self.cfg.passage_prefix if self.cfg.enabled else ""
        for chunk in chunks:
            chunk.embed_text = f"{prefix}{chunk.contextualized_text}".strip()
        return chunks


def prepare_query_text(query: str, config: Optional[EmbedTextConfig] = None) -> str:
    """
    Prefix a user's search query with "query: " for embedding at
    retrieval time. This is the query-side counterpart to
    EmbedTextBuilder.build() — the retriever module MUST use this
    function (not a hand-rolled prefix) so the query prefix can never
    drift out of sync with the passage prefix defined here.

    Usage (in the future retriever module):
        from rag_pipeline.enrichers.embed_text_builder import prepare_query_text
        query_embed_text = prepare_query_text(user_question)
        query_vector = embed_generator.embed_texts([query_embed_text])[0]
    """
    cfg = config or EmbedTextConfig()
    prefix = cfg.query_prefix if cfg.enabled else ""
    return f"{prefix}{query.strip()}".strip()
