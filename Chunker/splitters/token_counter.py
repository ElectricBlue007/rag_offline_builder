"""
splitters/token_counter.py

Provides accurate token counts matched to the actual embedding model's
tokenizer, instead of using character count as a rough proxy.

Why this matters
------------------
intfloat/multilingual-e5-large has a HARD 512-token limit. Text beyond
that is SILENTLY TRUNCATED at embedding time — no error, no warning —
and everything past token 512 is invisible to retrieval. Character
count is a poor proxy for token count: English averages ~4 chars/token,
but this varies with language (Spanish, with its longer words and
accented characters, tokenizes differently) and content type (a table's
"CMI<br/>CMI Report" tokenizes very differently from prose).

Once a "passage: " prefix (step 2 of the roadmap) and/or an LLM-written
contextual header (see contextual_enricher.py) get prepended before
embedding, the margin for error shrinks further. Token-based sizing
with a safety margin under 512 is what actually prevents silent
truncation — character-based sizing only approximates it.

Design
-------
Attempts to load the REAL tokenizer for the configured embedding model
via HuggingFace `transformers.AutoTokenizer`. This requires a one-time
download of a small tokenizer file (~1-5MB, NOT the full model) unless
already cached locally.

If `transformers` isn't installed, or the download fails (fully offline
environment, no HF_TOKEN, etc.), falls back to a word-count-based
heuristic. This mirrors the same graceful-degradation pattern already
used for the embedding model itself elsewhere in this pipeline — the
pipeline never hard-fails just because a nice-to-have isn't available.
"""

from typing import Optional


class TokenCounter:
    """
    Usage:
        counter = TokenCounter(model_name="intfloat/multilingual-e5-large")
        n = counter.count_tokens("some text")
        counter.using_real_tokenizer  # True if the real HF tokenizer loaded
    """

    def __init__(self, model_name: str, attempt_real_tokenizer: bool = True):
        self.model_name = model_name
        self.attempt_real_tokenizer = attempt_real_tokenizer
        self._tokenizer = None
        self._load_attempted = False
        self._using_real = False

    @property
    def using_real_tokenizer(self) -> bool:
        """True once a real HF tokenizer has been successfully loaded."""
        if not self._load_attempted:
            self._try_load()
        return self._using_real

    def count_tokens(self, text: str) -> int:
        """Return the token count for `text` using the best available method."""
        if not text:
            return 0

        if not self._load_attempted:
            self._try_load()

        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                # Tokenizer loaded but failed on this specific input
                # (shouldn't normally happen) — fall through to heuristic
                # rather than crashing the pipeline over a size estimate.
                pass

        return self._heuristic_count(text)

    def count_batch(self, texts: list[str]) -> list[int]:
        """Convenience batch wrapper — same per-item behaviour as count_tokens."""
        return [self.count_tokens(t) for t in texts]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        self._load_attempted = True

        if not self.attempt_real_tokenizer:
            return

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._using_real = True
        except Exception:
            # No network, package missing, model name not on HF Hub, etc.
            # Silently fall back — this is expected in fully offline setups
            # and should not print noise on every pipeline run.
            self._tokenizer = None
            self._using_real = False

    def _heuristic_count(self, text: str) -> int:
        """
        Word-count-based approximation, used when the real tokenizer isn't
        available. Word-based scales more consistently across languages
        than a flat chars/4 ratio (which is biased toward English) — E5's
        underlying XLM-RoBERTa-style tokenizer typically produces roughly
        1.3-1.5 subword tokens per whitespace-separated word for Latin-
        script text such as English and Spanish.

        This is explicitly an approximation. For exact counts, ensure
        `transformers` is installed and the tokenizer can be downloaded
        once (cached locally afterward, fully offline from then on).
        """
        word_count = len(text.split())
        if word_count == 0:
            return 0
        return max(1, round(word_count * 1.4))
