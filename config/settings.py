"""Compiler settings kept independent from retrieval/vector systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompilerSettings:
    max_candidate_tokens: int = 350
    min_density_for_high_importance: float = 0.62
    token_chars_per_token: int = 4
    allow_header_spacing_normalization: bool = True


DEFAULT_SETTINGS = CompilerSettings()
