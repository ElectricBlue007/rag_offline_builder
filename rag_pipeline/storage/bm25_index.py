"""
storage/bm25_index.py

Builds and queries a BM25 keyword index — the sparse-retrieval half of
hybrid search (roadmap step 6 layers Reciprocal Rank Fusion on top of
this + ChromaDB's dense search).

Why this matters for THIS project specifically: the source documents
are full of domain abbreviations an embedding model has never seen in
training and can only treat as near-meaningless tokens — CMI, MAC,
GCI, LSSCONF, TiempFMAC, Vel.MAC. Dense (vector) search handles
paraphrase well ("how do I delete a MAC?" ≈ "Eliminate the CMI") but
under-matches on exact technical terms. BM25 finds exact matches
regardless of what the embedding model "understands," which is
precisely the gap dense search leaves open here.

Indexes contextualized_text, not embed_text
---------------------------------------------
contextualized_text carries the situating blurb (breadcrumb/filename,
or LLM-written context) WITHOUT the E5 "passage: " instruction prefix.
That prefix is the same 9 characters in literally every single
document — zero keyword-search value, pure noise for BM25's term
statistics — while the situating blurb itself IS valuable ("System
reports" appearing in the blurb helps a query for "system reports"
match). So this is the one place in the pipeline that deliberately
does NOT use embed_text, and instead uses contextualized_text.

A BM25 quirk worth knowing: BM25's IDF term can go NEGATIVE for very
common terms (this is a documented property of the standard Okapi
formula, confirmed empirically against this project's own rank_bm25
installation). That means "score > 0" is NOT a valid relevance filter
— a legitimate keyword match can still score negative in a small
corpus. This module deliberately never filters by an absolute score
threshold; it only ever returns a RANKED list, which is what the
downstream RRF fusion step actually needs (rank position, not the raw
score value).
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from rag_pipeline.config import BM25IndexConfig

_RE_TOKEN = re.compile(r'\w+', re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Lowercase, split on non-word-character boundaries.

    Deliberately NO stopword removal: a stopword list risks discarding
    meaningful short technical tokens (e.g. Binder2.md's "NT" action
    code is exactly the kind of 2-letter token a generic stopword list
    might target).

    Deliberately NO stemming: stemming works against exact-term
    matching, which is the entire reason to run BM25 alongside dense
    search in the first place.
    """
    if not text:
        return []
    return _RE_TOKEN.findall(text.lower())


def _get_field(chunk: Union[dict, object], field_name: str, default=""):
    if isinstance(chunk, dict):
        return chunk.get(field_name, default)
    return getattr(chunk, field_name, default)


@dataclass
class BM25SearchResult:
    chunk_id: str
    score: float


class BM25Index:
    """
    Usage:
        index = BM25Index(config)
        index.build(chunks)          # chunks: list of dicts or EnrichedChunk
        index.save()                 # persist to config.persist_path
        ...later, possibly a new process...
        index = BM25Index(config)
        index.load()                 # restore from disk
        results = index.search("What is a CMI?", n_results=10)
    """

    def __init__(self, config: Optional[BM25IndexConfig] = None):
        self.cfg = config or BM25IndexConfig()
        self._bm25 = None
        self._chunk_ids: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Union[dict, object]]) -> None:
        """
        Tokenize every chunk's contextualized_text and build the BM25
        index over the resulting corpus. Chunk order doesn't matter —
        chunk_ids are tracked in parallel and used to identify results,
        never positional index alone.
        """
        if not chunks:
            self._bm25 = None
            self._chunk_ids = []
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install rank-bm25"
            ) from e

        self._chunk_ids = [_get_field(c, "chunk_id") for c in chunks]
        texts = [_get_field(c, "contextualized_text") for c in chunks]
        tokenized_corpus = [tokenize(t) for t in texts]

        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.cfg.k1, b=self.cfg.b)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 10) -> list[BM25SearchResult]:
        """
        Return the top n_results chunks ranked by BM25 score, highest
        first. Returns an empty list for an empty/whitespace-only query
        or an unbuilt/empty index — never raises for those cases.

        Deliberately does NOT filter by score value (e.g. "score > 0")
        — see module docstring for why that would be an incorrect
        relevance filter given BM25's negative-IDF behaviour on common
        terms. Callers wanting a relevance cutoff should apply their
        own threshold with full awareness of that quirk, or better,
        rely on rank position via the future RRF fusion step.
        """
        if not self.is_built or self.size == 0:
            return []
        if not query or not query.strip():
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            BM25SearchResult(chunk_id=cid, score=float(score))
            for cid, score in ranked[:n_results]
        ]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """Persist the built index + parallel chunk_id list via pickle."""
        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt index — call build() first.")
        target = Path(path or self.cfg.persist_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "chunk_ids": self._chunk_ids},
                f,
            )

    def load(self, path: Optional[str] = None) -> None:
        """Restore a previously-saved index from disk."""
        source = Path(path or self.cfg.persist_path)
        if not source.exists():
            raise FileNotFoundError(f"No BM25 index found at: {source}")
        with open(source, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
