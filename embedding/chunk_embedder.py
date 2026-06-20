"""Chunk embedding using Chonkie and multilingual-e5-large."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import SentenceTransformer

from embedding.model_loader import EmbeddingModelLoader
from rag_doc_compiler.models import ChunkCandidate


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""
    
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


class ChunkEmbedder:
    """Embed chunks using Chonkie-style strategy with multilingual-e5-large.
    
    This embedder uses a semantic chunking approach inspired by Chonkie,
    where chunks are embedded with context awareness using the multilingual
    e5-large model for high-quality multilingual embeddings.
    """
    
    def __init__(
        self,
        model_loader: EmbeddingModelLoader | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the chunk embedder.
        
        Args:
            model_loader: Optional custom model loader. Creates default if None.
            batch_size: Batch size for embedding generation.
            normalize_embeddings: Whether to normalize embedding vectors.
        """
        self.model_loader = model_loader or EmbeddingModelLoader()
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = self.model_loader.load_model(auto_download=True)
        return self._model
    
    def embed_chunks(
        self,
        chunks: list[ChunkCandidate],
        include_metadata: bool = True,
    ) -> list[EmbeddedChunk]:
        """Embed a list of chunk candidates.
        
        Args:
            chunks: List of ChunkCandidate objects to embed.
            include_metadata: Whether to include chunk metadata in output.
            
        Returns:
            List of EmbeddedChunk objects with vectors.
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 10,
        )
        
        # Create EmbeddedChunk objects
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {}
            if include_metadata:
                metadata = {
                    "parent_section_id": chunk.parent_section_id,
                    "heading_path": chunk.heading_path,
                    "line_range": chunk.line_range,
                    "content_type": chunk.content_type,
                    "intent": chunk.intent,
                    "semantic_tags": chunk.semantic_tags,
                    "importance_score": chunk.importance_score,
                    "token_estimate": chunk.token_estimate,
                    "hierarchy_level": chunk.hierarchy_level,
                }
            
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding.tolist(),
                metadata=metadata,
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension size (1024 for multilingual-e5-large).
        """
        return self.model.get_sentence_embedding_dimension()
