"""
splitters/llm_chunk_refiner.py

Optional refinement pass that runs immediately after SemanticChunker.
See LlmChunkRefinerConfig (config.py) for the full rationale — in short:
embedding-similarity chunking sometimes cuts a passage right before the
sentence that would have made it independently answerable. This stage
asks a local Ollama model, per group of adjacent same-section chunks,
which ones should be merged so that each resulting chunk stands alone as
a complete answer to whatever question it's relevant to.

Hard invariant: the LLM decides MERGE BOUNDARIES ONLY. It never rewrites
chunk text — merged chunks are the original content strings concatenated
verbatim, nothing paraphrased or summarized. This pipeline embeds
technical manuals; an LLM silently "improving" a procedure step or a
table value would corrupt the source of truth being retrieved, which is
a far worse failure than an imperfect chunk boundary.
"""

import json
import re
from typing import Optional

from Chunker.config import LlmChunkRefinerConfig
from Chunker.splitters.semantic_chunker import SemanticChunk, _hard_split


class LlmChunkRefiner:
    """
    Usage:
        refiner = LlmChunkRefiner(config)
        chunks = refiner.refine(chunks)   # chunks fresh from SemanticChunker

    Only "text" section_type chunks are ever touched — "table" and
    "image_description" chunks pass through unchanged, matching the same
    atomic-unit rule SemanticChunker itself already applies to them.
    """

    def __init__(self, config: Optional[LlmChunkRefinerConfig] = None):
        self.cfg = config or LlmChunkRefinerConfig()
        self._ollama_client = None    # lazy-loaded; settable directly in tests
        self._fallback_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def refine(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """
        Re-groups `chunks` (in place order, same list shape SemanticChunker
        produces) by merging adjacent "text" chunks the LLM judges as
        belonging together. Returns a new list; re-indexing (chunk_index)
        is the caller's responsibility, same as SemanticChunker.chunk_sections.
        """
        if not self.cfg.enabled or not chunks:
            return chunks

        result: list[SemanticChunk] = []
        for group in self._group_by_section(chunks):
            result.extend(self._refine_group(group))
        return result

    # ── Grouping ─────────────────────────────────────────────────────────────

    @staticmethod
    def _group_by_section(chunks: list[SemanticChunk]) -> list[list[SemanticChunk]]:
        """
        Splits the flat chunk list into consecutive runs that came from
        the same structural section — SemanticChunker.chunk_sections()
        always emits a section's chunks contiguously, so grouping by a
        change in (breadcrumb, section_type, source_path) reconstructs
        section boundaries without needing the original sections passed
        back in separately.
        """
        groups: list[list[SemanticChunk]] = []
        current: list[SemanticChunk] = []
        current_key = None

        for chunk in chunks:
            key = (chunk.breadcrumb, chunk.section_type, chunk.source_path)
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(chunk)
            current_key = key

        if current:
            groups.append(current)
        return groups

    # ── Per-section refinement ──────────────────────────────────────────────

    def _refine_group(self, group: list[SemanticChunk]) -> list[SemanticChunk]:
        # Tables and image descriptions are always atomic — never touched.
        if group[0].section_type != "text":
            return group

        if len(group) < self.cfg.min_chunks_to_consider:
            return group

        refined: list[SemanticChunk] = []
        for i in range(0, len(group), self.cfg.max_chunks_per_call):
            batch = group[i : i + self.cfg.max_chunks_per_call]
            refined.extend(self._refine_batch(batch))
        return refined

    def _refine_batch(self, batch: list[SemanticChunk]) -> list[SemanticChunk]:
        if len(batch) < self.cfg.min_chunks_to_consider:
            return batch

        try:
            prompt = self._build_prompt(batch)
            response_text = self._call_ollama(prompt)
            merge_groups = self._parse_merge_groups(response_text, len(batch))
            return self._apply_merge_groups(batch, merge_groups)
        except Exception as exc:
            return self._fallback(batch, reason=str(exc))

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_prompt(self, batch: list[SemanticChunk]) -> str:
        numbered = "\n\n".join(
            f"[{i}]\n{chunk.content}" for i, chunk in enumerate(batch)
        )
        return (
            "You will be shown several consecutive passages taken from the "
            "same section of a technical document, in their original order. "
            "They were split automatically and the split points are not "
            "always correct.\n\n"
            "For each passage, ask: if a user asked a question that only "
            "this passage could answer, would this passage ALONE contain "
            "everything needed to answer it clearly — or does it feel "
            "incomplete, like it's missing a step, a conclusion, or context "
            "that spilled into the passage right before or after it?\n\n"
            "Group the passage numbers so that each group forms one "
            "complete, self-contained unit. A passage that is already "
            "complete on its own is its own group of one. Passages must "
            "stay in their original relative order — only merge ADJACENT "
            "passages, never reorder or skip.\n\n"
            f"Passages:\n{numbered}\n\n"
            "Reply with ONLY a JSON object of this exact shape, no other "
            "text, no markdown code fences:\n"
            '{"merge_groups": [[0, 1], [2], [3, 4, 5]]}'
        )

    # ── Ollama client (lazy, mockable — mirrors contextual_enricher.py) ─────

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

        response = self._ollama_client.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_response_text(response)

    @staticmethod
    def _extract_response_text(response) -> str:
        """
        ollama-python's response shape has varied slightly across
        versions (dict-like vs. attribute-style objects) — try both
        rather than assuming one and crashing on the other. Same
        defensive pattern used in llm_client.py and contextual_enricher.py.
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

    # ── Response parsing (defensive — never trust raw LLM output) ──────────

    _RE_JSON_OBJECT = re.compile(r'\{.*\}', re.DOTALL)

    def _parse_merge_groups(self, response_text: str, n: int) -> list[list[int]]:
        """
        Extract {"merge_groups": [[...], ...]} from the model's reply and
        validate it strictly: every index 0..n-1 must appear EXACTLY
        once, groups must be non-empty, and (since the prompt asks for
        adjacency only) each group's indices must be a contiguous
        ascending run. Any violation raises, which the caller treats
        exactly like an LLM/network failure — fall back to no merging
        rather than trust a malformed or hallucinated grouping.
        """
        match = self._RE_JSON_OBJECT.search(response_text)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {response_text[:200]!r}")

        parsed = json.loads(match.group(0))
        groups = parsed["merge_groups"]

        seen: set[int] = set()
        for group in groups:
            if not group:
                raise ValueError("merge_groups contained an empty group")
            if list(group) != list(range(group[0], group[0] + len(group))):
                raise ValueError(f"merge_groups entry {group} is not a contiguous ascending run")
            for idx in group:
                if idx in seen:
                    raise ValueError(f"index {idx} appeared in more than one group")
                seen.add(idx)

        if seen != set(range(n)):
            raise ValueError(
                f"merge_groups covered indices {sorted(seen)}, expected exactly 0..{n - 1}"
            )

        return [list(g) for g in groups]

    # ── Applying the merge decision (verbatim concatenation only) ──────────

    def _apply_merge_groups(
        self, batch: list[SemanticChunk], merge_groups: list[list[int]]
    ) -> list[SemanticChunk]:
        merged: list[SemanticChunk] = []
        for group in merge_groups:
            if len(group) == 1:
                merged.append(batch[group[0]])
                continue

            pieces = [batch[i] for i in group]
            combined_content = "\n\n".join(p.content for p in pieces).strip()
            first = pieces[0]
            merged_chunk = SemanticChunk(
                content=combined_content,
                section_type=first.section_type,
                breadcrumb=first.breadcrumb,
                header_levels=list(first.header_levels),
                header_texts=list(first.header_texts),
                source_path=first.source_path,
                normalized_text=combined_content,
                contextualized_text=combined_content,
                embed_text=combined_content,
            )
            merged.extend(self._resplit_if_oversized(merged_chunk))
        return merged

    def _resplit_if_oversized(self, chunk: SemanticChunk) -> list[SemanticChunk]:
        """
        A merge can push a chunk above max_chunk_tokens. Re-applies the
        SAME deterministic sentence-boundary hard-split SemanticChunker
        uses as its own safety net — this is a size safeguard, not a
        content decision, so it stays deterministic rather than going
        back to the LLM.
        """
        # Uses character count (len) here deliberately: this module has no
        # tokenizer wiring of its own, and a generous character ceiling is
        # a reasonable safety net for the rare case a merge overshoots —
        # exact token-precision isn't the point of this fallback.
        max_chars = 4000
        if len(chunk.content) <= max_chars:
            return [chunk]

        pieces = _hard_split(chunk.content, max_chars, len)
        return [
            SemanticChunk(
                content=piece,
                section_type=chunk.section_type,
                breadcrumb=chunk.breadcrumb,
                header_levels=list(chunk.header_levels),
                header_texts=list(chunk.header_texts),
                source_path=chunk.source_path,
                normalized_text=piece,
                contextualized_text=piece,
                embed_text=piece,
            )
            for piece in pieces
        ]

    # ── Fallback handling (mirrors contextual_enricher.py) ──────────────────

    def _fallback(self, batch: list[SemanticChunk], reason: str) -> list[SemanticChunk]:
        self._fallback_count += 1
        if not self.cfg.fallback_on_error:
            raise RuntimeError(f"LLM chunk refinement failed ({reason}).")
        if self.cfg.warn_on_fallback:
            preview = batch[0].content[:60].replace("\n", " ")
            print(
                f"  [LlmChunkRefiner] WARNING: refinement failed ({reason}), "
                f"keeping original chunk boundaries. Preview: {preview}..."
            )
        return batch
