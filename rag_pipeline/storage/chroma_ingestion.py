"""
storage/chroma_ingestion.py

Ingests embedded chunks into a local, persistent ChromaDB collection —
no server required, everything lives in a directory on disk.

Deliberately decoupled from the main chunking pipeline (pipeline.py)
and from embedding generation (embedding/embed_generator.py), matching
the same separation-of-concerns used throughout this project: each
stage can be re-run independently. Typical flow:

    1. pipeline.py            → chunks/*.chunks.json
    2. embed_generator.py     → EmbeddingResult objects (chunk_id + vector)
    3. chroma_ingestion.py     → persisted, queryable ChromaDB collection

ChromaDB metadata constraint
-----------------------------
ChromaDB only accepts scalar metadata values (str / int / float / bool)
— NOT lists, NOT None. Our chunk metadata includes page_numbers as a
list[int] (from metadata_enricher.py), which would raise at ingestion
time if passed through unchanged. _sanitize_metadata_for_chroma()
below handles this: lists become comma-separated strings, None values
are dropped, and everything else passes through as-is. This is exactly
the kind of interface mismatch that's invisible until you actually try
to ingest real data — which is why it's handled here explicitly rather
than assumed away.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rag_pipeline.config import ChromaIngestionConfig


@dataclass
class IngestionSummary:
    ingested_count: int
    collection_count_after: int
    skipped_no_vector: int = 0


def _sanitize_metadata_for_chroma(metadata: dict) -> dict:
    """
    Convert a chunk's metadata dict into a form ChromaDB will accept:
      - list values  → comma-separated string (e.g. page_numbers=[1,2] → "1,2")
      - None values  → dropped entirely (key omitted)
      - str/int/float/bool → passed through unchanged
      - anything else → stringified as a fallback, rather than raising
    """
    sanitized: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, list):
            sanitized[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def _get_field(chunk: Union[dict, object], field: str, default=""):
    if isinstance(chunk, dict):
        return chunk.get(field, default)
    return getattr(chunk, field, default)


class ChromaIngestor:
    """
    Usage:
        ingestor = ChromaIngestor(config)
        summary = ingestor.ingest_chunks(chunks, embedding_results)
        results = ingestor.query(query_vector, n_results=10)
    """

    def __init__(self, config: Optional[ChromaIngestionConfig] = None):
        self.cfg = config or ChromaIngestionConfig()
        self._client = None       # lazy-loaded; settable directly in tests
        self._collection = None  # lazy-loaded; settable directly in tests

    # ── Lazy client/collection ──────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install chromadb"
            ) from e
        self._client = chromadb.PersistentClient(path=self.cfg.persist_directory)
        return self._client

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self.cfg.collection_name
        )
        return self._collection

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_chunks(
        self,
        chunks: list[Union[dict, object]],
        embedding_results: list,
    ) -> IngestionSummary:
        """
        Upsert `chunks` (with their precomputed vectors from
        embedding_results) into the ChromaDB collection.

        Matches chunks to vectors by chunk_id (not positional order) —
        this is deliberately robust to embedding_results arriving in a
        different order than chunks, or a subset of chunks having failed
        to embed for some reason.

        `documents` (what ChromaDB returns as the primary retrieved text)
        is normalized_text — clean prose without the "passage: " prefix
        or the situating blurb, which would be redundant/awkward if
        surfaced directly as a retrieved answer. `content` (raw HTML for
        tables) and `contextualized_text` are preserved in metadata for
        anyone who needs the original representation.
        """
        vector_by_id = {r.chunk_id: r.vector for r in embedding_results}

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        skipped = 0

        for chunk in chunks:
            chunk_id = _get_field(chunk, "chunk_id")
            vector = vector_by_id.get(chunk_id)
            if vector is None:
                skipped += 1
                continue

            base_metadata = dict(_get_field(chunk, "metadata", {}) or {})
            base_metadata["content"] = _get_field(chunk, "content")
            base_metadata["contextualized_text"] = _get_field(chunk, "contextualized_text")

            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(_get_field(chunk, "normalized_text"))
            metadatas.append(_sanitize_metadata_for_chroma(base_metadata))

        collection = self._get_collection()
        self._upsert_in_batches(collection, ids, embeddings, documents, metadatas)

        return IngestionSummary(
            ingested_count=len(ids),
            collection_count_after=collection.count(),
            skipped_no_vector=skipped,
        )

    def _upsert_in_batches(
        self, collection, ids, embeddings, documents, metadatas
    ) -> None:
        batch_size = self.cfg.batch_size
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    # ── Convenience query (smoke-testing / lightweight retrieval) ───────────

    def query(
        self,
        query_vector: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Thin wrapper around collection.query() — similarity search.
        Building block for the hybrid retriever (retrieval/hybrid_retriever.py),
        which layers BM25 + RRF on top of this dense-only lookup.
        """
        collection = self._get_collection()
        kwargs = {"query_embeddings": [query_vector], "n_results": n_results}
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    def get_by_ids(self, ids: list[str]) -> dict:
        """
        Direct ID lookup — NOT similarity search. Used by the hybrid
        retriever to fetch document/metadata for chunks that BM25 found
        but that weren't present in the dense search's top-N results
        (i.e. a sparse-only hit still needs its display text from
        somewhere, and this is the somewhere).

        Note the response SHAPE differs from query(): .get() returns a
        flat single-level dict ({"ids": [...], "documents": [...], ...}),
        NOT the nested list-of-lists that .query() returns (that nesting
        exists specifically to support batched similarity queries, which
        doesn't apply here). Mixing these two shapes up is an easy
        mistake — handled explicitly rather than assumed.
        """
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        collection = self._get_collection()
        return collection.get(ids=ids, include=["documents", "metadatas"])

    def count(self) -> int:
        """Number of items currently in the collection."""
        return self._get_collection().count()
