"""
retrieval/hybrid_retriever.py

Combines dense (ChromaDB) and sparse (BM25) retrieval via Reciprocal
Rank Fusion (RRF) — the standard, calibration-free way to merge two
ranked lists that live on completely incomparable score scales (cosine
similarity vs. BM25's term-statistics score).

Query-time flow:
  1. Detect the query's language (reuses preprocessors/language_detector.py)
  2. Apply the configured query-side instruction prefix via
     prepare_query_text() — the single source of truth for that contract
     (enrichers/embed_text_builder.py, EmbedTextConfig)
  3. Embed the prefixed query
  4. Dense search: ChromaDB, optionally pre-filtered by detected language
  5. Sparse search: BM25, using the RAW (unprefixed) query text — an
     instruction prefix has zero keyword-matching value and would just
     inject spurious common tokens into every single BM25 query, the
     same reasoning that keeps BM25's index itself built from
     contextualized_text rather than embed_text (see bm25_index.py)
  6. Fuse both ranked lists via RRF — rank position only, never the raw
     scores, since dense and sparse scores are not on comparable scales
  7. Backfill document/metadata for any chunk found only by BM25 (i.e.
     outside the dense search's top-N) via a direct ChromaDB ID lookup,
     so every final result has full display data regardless of which
     retriever actually found it
"""

from dataclasses import dataclass
from typing import Optional

from rag_pipeline.config import HybridRetrieverConfig
from rag_pipeline.preprocessors.language_detector import detect_language
from rag_pipeline.enrichers.embed_text_builder import prepare_query_text
from rag_pipeline.embedding.embed_generator import EmbedGenerator
from rag_pipeline.storage.chroma_ingestion import ChromaIngestor
from rag_pipeline.storage.bm25_index import BM25Index


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """
    score(doc) = Σ 1/(k + rank) across every ranked_list containing doc,
    where rank is the doc's 1-indexed position in that particular list.

    A document absent from one of the lists simply contributes 0 from
    that list — there's no penalty term, and this is the correct,
    standard RRF behaviour (not an approximation of it).

    Generalizes to any number of ranked lists, not just two — kept this
    way deliberately in case a third retrieval signal gets added later
    without needing to touch this function.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str            # normalized_text — clean prose, from ChromaDB's documents
    metadata: dict
    rrf_score: float
    dense_rank: Optional[int] = None   # None if not found by dense search
    sparse_rank: Optional[int] = None  # None if not found by sparse search


class HybridRetriever:
    """
    Usage:
        retriever = HybridRetriever(chroma_ingestor, bm25_index, embed_generator)
        results = retriever.retrieve("What is a CMI?")
    """

    def __init__(
        self,
        chroma_ingestor: ChromaIngestor,
        bm25_index: BM25Index,
        embed_generator: EmbedGenerator,
        config: Optional[HybridRetrieverConfig] = None,
    ):
        self.chroma = chroma_ingestor
        self.bm25 = bm25_index
        self.embedder = embed_generator
        self.cfg = config or HybridRetrieverConfig()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        language = detect_language(query)

        # ── Dense search ─────────────────────────────────────────────────────
        query_embed_text = prepare_query_text(query)
        query_vector = self.embedder.embed_texts([query_embed_text])[0]

        where = None
        if self.cfg.language_filter and language != "unknown":
            where = {"language": language}

        dense_raw = self.chroma.query(
            query_vector, n_results=self.cfg.dense_n_results, where=where
        )
        dense_ids = self._unwrap(dense_raw.get("ids"))
        dense_documents = self._unwrap(dense_raw.get("documents"))
        dense_metadatas = self._unwrap(dense_raw.get("metadatas"))

        # ── Sparse search (raw query text — no E5 prefix) ───────────────────
        sparse_results = self.bm25.search(query, n_results=self.cfg.sparse_n_results)
        sparse_ids = [r.chunk_id for r in sparse_results]

        # ── Fuse ─────────────────────────────────────────────────────────────
        fused_scores = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.cfg.rrf_k)
        if not fused_scores:
            return []

        dense_rank_map = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_rank_map = {cid: i + 1 for i, cid in enumerate(sparse_ids)}

        ranked_ids = sorted(
            fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True
        )
        top_ids = ranked_ids[: self.cfg.final_n_results]

        # ── Resolve display text/metadata for every result ──────────────────
        # Dense hits already carry this data for free from the query above;
        # sparse-only hits (found by BM25 but outside the dense top-N) need
        # a direct ID fetch to get their document/metadata.
        info_by_id: dict[str, dict] = {
            cid: {"document": doc, "metadata": meta}
            for cid, doc, meta in zip(dense_ids, dense_documents, dense_metadatas)
        }
        missing_ids = [cid for cid in top_ids if cid not in info_by_id]
        if missing_ids:
            fetched = self.chroma.get_by_ids(missing_ids)
            for cid, doc, meta in zip(
                fetched.get("ids", []),
                fetched.get("documents", []),
                fetched.get("metadatas", []),
            ):
                info_by_id[cid] = {"document": doc, "metadata": meta}

        results = []
        for cid in top_ids:
            info = info_by_id.get(cid, {"document": "", "metadata": {}})
            results.append(RetrievedChunk(
                chunk_id=cid,
                text=info["document"],
                metadata=info["metadata"],
                rrf_score=fused_scores[cid],
                dense_rank=dense_rank_map.get(cid),
                sparse_rank=sparse_rank_map.get(cid),
            ))
        return results

    @staticmethod
    def _unwrap(nested: Optional[list]) -> list:
        """
        ChromaDB's .query() wraps every field in an outer list — one
        entry per query embedding sent in the batch. This retriever
        always sends exactly one query embedding, so unwrap that single
        entry. Handles None/empty input gracefully (e.g. an empty
        collection returns {"ids": [[]], ...} or similar edge shapes).
        """
        if not nested:
            return []
        return nested[0] if nested else []
