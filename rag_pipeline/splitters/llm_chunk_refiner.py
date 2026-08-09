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

from rag_pipeline.config import LlmChunkRefinerConfig
from rag_pipeline.splitters.semantic_chunker import SemanticChunk, _hard_split


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
        n = len(batch)
        numbered = "\n\n".join(
            f"[{i}]\n{chunk.content}" for i, chunk in enumerate(batch)
        )
        example = self._example_group_ids(n)
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
            "Passages that belong together should share the same group "
            "number; a passage that is already complete on its own gets "
            "its own group number.\n\n"
            f"Passages:\n{numbered}\n\n"
            f"There are {n} passages above, numbered [0] to [{n - 1}]. "
            f"Reply with ONLY a JSON object of this exact shape: a "
            f'"group_ids" array with EXACTLY {n} integers — no more, no '
            "fewer — one per passage IN ORDER (the i-th number is passage "
            "[i]'s group id). Group ids must never decrease as you move "
            "through the array — start at 0 and only increase by 0 or 1 "
            f"at each step. Since there are {n} passages here, your array "
            f"must have exactly {n} numbers in it, for example:\n"
            f'{{"group_ids": {example}}}\n\n'
            "No other text, no markdown code fences — just that JSON "
            "object, with an array of the correct length for the "
            "passages actually shown above."
        )

    @staticmethod
    def _example_group_ids(n: int) -> list[int]:
        """
        Builds an illustrative example ALWAYS the same length as the
        real batch (n), so the model has no fixed-length example to
        anchor/copy instead of the actual passage count. A previous,
        static example (always length 6) caused smaller models to
        return arrays matching the example's length rather than the
        batch's real length on small batches. Shows one merge (indices
        0 and 1 share a group) followed by all-separate groups, which
        is enough to demonstrate the "repeat = merge" rule without
        needing the array to be trivial ([0, 1, 2, ...], i.e. no merges
        at all, which wouldn't demonstrate the rule at all).
        """
        if n <= 1:
            return [0] * n
        return [0, 0] + list(range(1, n - 1))

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

        options = {
            # Deterministic on purpose: this is a structured extraction
            # task (which adjacent passages belong together), not a
            # creative one — sampling temperature only adds variance and
            # a higher chance of a hallucinated/miscounted index, with
            # zero upside. Keep this at 0 even if the model is swapped.
            "temperature": 0,
        }
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
        Extract {"group_ids": [...]} from the model's reply and validate
        it strictly, then convert it into the same [[0,1],[2],...] shape
        _apply_merge_groups expects.

        Design note: earlier this asked the model to directly BUILD the
        nested group lists (e.g. {"merge_groups": [[0,1],[2]]}). Weaker
        models routinely skipped, duplicated, or hallucinated indices
        that way (e.g. inventing passage "6" in a 3-passage batch) —
        because nothing about that shape forces the model to visit every
        index exactly once. Asking for one group id PER PASSAGE, in
        order, removes that failure mode almost entirely: the model just
        walks passages 0..n-1 and labels each one, so it structurally
        cannot skip or duplicate an index — the only way this call still
        raises is if the ids aren't well-formed (wrong length, or not
        the required "start at 0, only ever step up by 0 or 1" run,
        which is what "contiguous ascending adjacency" collapses to in
        this representation). Any violation raises, which the caller
        treats exactly like an LLM/network failure — fall back to no
        merging rather than trust a malformed or hallucinated grouping.
        """
        match = self._RE_JSON_OBJECT.search(response_text)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {response_text[:200]!r}")

        parsed = json.loads(match.group(0))
        group_ids = parsed["group_ids"]

        if len(group_ids) != n:
            raise ValueError(f"group_ids has length {len(group_ids)}, expected {n}")

        for idx, gid in enumerate(group_ids):
            if not isinstance(gid, int):
                raise ValueError(f"group_ids[{idx}]={gid!r} is not an integer")
            if idx == 0:
                if gid != 0:
                    raise ValueError(f"group_ids must start at 0, got {gid}")
            elif gid not in (group_ids[idx - 1], group_ids[idx - 1] + 1):
                # Each id must either repeat the previous one (still the
                # same group) or step up by exactly 1 (a new group) —
                # never jump ahead and never drop back to an earlier one.
                raise ValueError(
                    f"group_ids {group_ids} is not a valid 0-based, "
                    "step-by-0-or-1 sequence"
                )

        groups: list[list[int]] = []
        for idx, gid in enumerate(group_ids):
            if idx > 0 and gid == group_ids[idx - 1]:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        return groups

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
