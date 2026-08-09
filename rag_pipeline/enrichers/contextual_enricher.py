"""
enrichers/contextual_enricher.py

Implements "Contextual Retrieval": prepends a short blurb to each chunk
BEFORE it gets embedded, situating it within its source document. This
matters because chunking necessarily strips context — a table row like
"CMI Report: (no restrictions)" is meaningless in isolation, but the
document it came from makes clear it's from "SERVICE: Great circle >
System reports". Without this, the embedding model has to guess; with
it, the relationship is spelled out in the text that actually gets
embedded (not just tucked away in metadata the embedding model never
sees).

This is the same core technique Anthropic published research on
(reducing retrieval failures substantially by adding situating context
per-chunk before embedding) — implemented here for a fully offline
pipeline.

Two modes
----------
"deterministic" (default) — builds the situating text from data already
  on the chunk: its breadcrumb and source filename. Zero cost, zero
  external dependency, works immediately with no setup. Captures most
  of the value: the embedding model learns WHERE a chunk is from.

"llm" — sends the surrounding document + the chunk to a local Ollama
  model, which writes 1-2 sentences of genuine situating prose (what
  this passage is about and how it relates to the rest of the
  document — not just "where" but "why it matters here"). Meaningfully
  higher quality, but requires Ollama running locally, and is much
  slower: one LLM call per chunk. For a 50-chunk document, expect real
  wall-clock minutes on CPU-only local hardware. Falls back to
  deterministic mode automatically on any failure (Ollama not running,
  timeout, model not pulled, etc.) unless configured not to.

"off" — contextualized_text becomes an exact copy of normalized_text,
  no prefix added at all.

Design invariant: contextualized_text is ALWAYS derived fresh from
normalized_text for every chunk, regardless of mode. This matters
because normalized_text itself gets overwritten upstream by
TableChunkNormalizer for table chunks — if this stage only touched
some chunks (e.g. skipped when mode="off"), table chunks would leak
stale raw HTML through contextualized_text. Always re-deriving avoids
that entirely.
"""

import re
from dataclasses import dataclass
from typing import Optional

from rag_pipeline.config import ContextualEnricherConfig
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


_VALID_MODES = {"deterministic", "llm", "off"}


class ContextualEnricher:
    """
    Usage:
        enricher = ContextualEnricher(config)
        chunks = enricher.enrich_chunks(chunks, document_text, source_filename)
    """

    def __init__(self, config: Optional[ContextualEnricherConfig] = None):
        self.cfg = config or ContextualEnricherConfig()
        if self.cfg.mode not in _VALID_MODES:
            raise ValueError(
                f"ContextualEnricherConfig.mode must be one of {_VALID_MODES}, "
                f"got {self.cfg.mode!r}"
            )
        self._ollama_client = None   # lazy-loaded; settable directly in tests
        self._llm_fallback_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def enrich_chunks(
        self,
        chunks: list[SemanticChunk],
        document_text: str,
        source_filename: str,
    ) -> list[SemanticChunk]:
        """
        Populate contextualized_text for every chunk in `chunks`.
        `document_text` is the full cleaned markdown of the source file
        (used only in "llm" mode, to give the model surrounding context).
        `source_filename` is the .md filename (used in "deterministic"
        mode and as a fallback label in "llm" mode).
        """
        for chunk in chunks:
            prefix = self._build_prefix(chunk, document_text, source_filename)
            base = chunk.normalized_text
            chunk.contextualized_text = (
                f"{prefix} {base}".strip() if prefix else base.strip()
            )
        return chunks

    # ── Prefix construction, dispatched by mode ─────────────────────────────

    def _build_prefix(
        self, chunk: SemanticChunk, document_text: str, source_filename: str
    ) -> str:
        if self.cfg.mode == "off":
            return ""
        elif self.cfg.mode == "deterministic":
            return self._deterministic_prefix(chunk, source_filename)
        else:  # "llm"
            return self._llm_prefix(chunk, document_text, source_filename)

    def _deterministic_prefix(self, chunk: SemanticChunk, source_filename: str) -> str:
        parts = []
        if self.cfg.include_source_filename and source_filename:
            parts.append(f"From {source_filename}")
        if (
            self.cfg.include_breadcrumb
            and chunk.breadcrumb
            and chunk.breadcrumb != "(root)"
        ):
            parts.append(f"section '{self._strip_hashes(chunk.breadcrumb)}'")

        if not parts:
            return ""
        return ", ".join(parts) + ":"

    @staticmethod
    def _strip_hashes(breadcrumb: str) -> str:
        """
        Breadcrumbs are stored like "### SERVICE: Great circle > ### System
        reports" — strip the markdown '#' markers for a clean readable
        prefix: "SERVICE: Great circle > System reports".
        """
        segments = breadcrumb.split(" > ")
        cleaned = [re.sub(r'^#+\s*', '', s).strip() for s in segments]
        return " > ".join(cleaned)

    def _llm_prefix(
        self, chunk: SemanticChunk, document_text: str, source_filename: str
    ) -> str:
        try:
            excerpt = document_text[: self.cfg.max_document_chars_for_llm]
            prompt = self._build_llm_prompt(excerpt, chunk.normalized_text)
            response_text = self._call_ollama(prompt).strip()
            if response_text:
                return response_text
            return self._fallback(chunk, source_filename, reason="empty LLM response")
        except Exception as exc:
            return self._fallback(chunk, source_filename, reason=str(exc))

    def _build_llm_prompt(self, document_excerpt: str, chunk_text: str) -> str:
        return (
            "You will be given an excerpt from a larger document, followed "
            "by one specific passage taken from that document.\n\n"
            f'Document excerpt:\n"""\n{document_excerpt}\n"""\n\n'
            f'Passage to contextualize:\n"""\n{chunk_text}\n"""\n\n'
            "Write one or two short sentences that place this passage in "
            "context — mention what section, system, or topic it belongs "
            "to and what role it plays in the document. Do not summarize "
            "the passage itself; only provide the surrounding context a "
            "reader would need to understand where it fits. Reply with "
            "only the context sentences, nothing else."
        )

    # ── Ollama client (lazy, mockable) ──────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        if self._ollama_client is None:
            try:
                import ollama
            except ImportError as e:
                raise ImportError(
                    "Missing dependency. Run:\n  pip install ollama\n"
                    "and ensure the Ollama service is running locally "
                    f"at {self.cfg.llm_host} with model "
                    f"'{self.cfg.llm_model}' pulled."
                ) from e
            self._ollama_client = ollama.Client(host=self.cfg.llm_host)

        options = {}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = self.cfg.num_ctx
        response = self._ollama_client.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return self._extract_response_text(response)

    @staticmethod
    def _extract_response_text(response) -> str:
        """
        ollama-python's response shape has varied slightly across
        versions (dict-like vs. attribute-style objects) — try both
        rather than assuming one and crashing on the other.
        """
        try:
            return response["message"]["content"]
        except (TypeError, KeyError, IndexError):
            pass
        try:
            return response.message.content
        except AttributeError:
            pass
        return str(response)

    # ── Fallback handling ────────────────────────────────────────────────────

    def _fallback(self, chunk: SemanticChunk, source_filename: str, reason: str) -> str:
        self._llm_fallback_count += 1
        if not self.cfg.fallback_to_deterministic_on_error:
            raise RuntimeError(f"Contextual LLM enrichment failed ({reason}).")
        if self.cfg.warn_on_fallback:
            preview = chunk.normalized_text[:60].replace("\n", " ")
            print(
                f"  [ContextualEnricher] WARNING: LLM context failed "
                f"({reason}), falling back to deterministic context. "
                f"Preview: {preview}..."
            )
        return self._deterministic_prefix(chunk, source_filename)
