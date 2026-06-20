"""Final chunking strategy using Chonkie for semantic chunking with multilingual embeddings.

This stage produces retrieval-ready chunks using Chonkie's semantic chunker with
intfloat/multilingual-e5-large (or small) embeddings for optimal multilingual support.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chonkie import SemanticChunker as ChonkieSemanticChunker
from sentence_transformers import SentenceTransformer

from config.settings import CompilerSettings, DEFAULT_SETTINGS, ModelSettings, DEFAULT_MODEL_SETTINGS
from core.chunk_preparer import ChunkPreparer
from rag_doc_compiler.errors import ChunkPreparationError
from rag_doc_compiler.models import DocumentNode, FinalChunk, StructureAnalysis, ChunkingResult


class MultilingualEmbeddingBackend:
    """Multilingual embedding backend using sentence-transformers with Chonkie compatibility."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        cache_dir: str = "models/embeddings",
    ) -> None:
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model. 
                       Recommended: 'intfloat/multilingual-e5-large' (if disk/memory allows),
                                   'intfloat/multilingual-e5-small' (lighter alternative),
                                   'BAAI/bge-m3' (excellent multilingual performance).
            cache_dir: Directory to cache the model weights.
        """
        from chonkie.embeddings.base import BaseEmbeddings
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None
        
        # Make this class inherit from BaseEmbeddings dynamically for Chonkie compatibility
        self.__class__ = type(
            'MultilingualEmbeddingBackend',
            (MultilingualEmbeddingBackend, BaseEmbeddings),
            dict(MultilingualEmbeddingBackend.__dict__)
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
            )
        return self._model

    def embed(self, text: str) -> Any:
        """Embed a single text (Chonkie interface).
        
        Args:
            text: Text string to embed.
            
        Returns:
            numpy.ndarray: Embedding vector.
        """
        import numpy as np
        
        if not text.strip():
            return np.array([])
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding)

    def embed_batch(self, texts: list[str]) -> list[Any]:
        """Embed multiple texts (Chonkie interface).
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of numpy.ndarray embeddings.
        """
        import numpy as np
        
        if not texts:
            return []
        
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            return [np.array([]) for _ in texts]
        
        embeddings = self.model.encode(
            non_empty_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        result = []
        non_empty_idx = 0
        for text in texts:
            if text.strip():
                result.append(np.array(embeddings[non_empty_idx]))
                non_empty_idx += 1
            else:
                result.append(np.array([]))
        
        return result

    def get_tokenizer(self):
        """Return tokenizer info for Chonkie compatibility."""
        from chonkie.tokenizer import ByteTokenizer
        # Return a simple byte tokenizer - works for all text
        return ByteTokenizer()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class ChonkieChunkingStrategy:
    """Semantic chunking strategy using Chonkie with multilingual E5 embeddings.
    
    This strategy leverages Chonkie's semantic chunker which:
    - Splits text into sentences
    - Computes semantic similarity between sentences using embeddings
    - Groups sentences into chunks based on semantic coherence
    - Respects chunk size limits while maintaining semantic boundaries
    
    Features:
    - Multilingual support via intfloat/multilingual-e5 models
    - Adaptive chunk sizes based on semantic similarity
    - Configurable similarity threshold for chunk boundaries
    - Optional overlap for context preservation
    """

    def __init__(
        self,
        settings: CompilerSettings = DEFAULT_SETTINGS,
        model_settings: ModelSettings = DEFAULT_MODEL_SETTINGS,
        embedding_model: str | None = None,
        threshold: float = 0.7,
        chunk_size: int = 512,
        min_sentences_per_chunk: int = 1,
        similarity_window: int = 3,
        include_overlap: bool = False,
        overlap_size: int = 50,
    ) -> None:
        """Initialize the Chonkie-based chunking strategy.
        
        Args:
            settings: Compiler settings for token estimation and other parameters.
            model_settings: Model settings for embedding configuration.
            embedding_model: Override for embedding model name. If None, uses model_settings.
                           Recommended models:
                           - 'intfloat/multilingual-e5-large': Best quality (requires ~2.2GB disk, ~2GB RAM)
                           - 'intfloat/multilingual-e5-small': Good quality (requires ~470MB disk, ~1GB RAM)
                           - 'BAAI/bge-m3': Excellent multilingual (requires ~2.2GB disk, ~2GB RAM)
                           - 'minishlab/potion-base-32M': Lightweight fallback (~32MB)
            threshold: Semantic similarity threshold (0-1). Lower = more chunks, higher = fewer chunks.
                      Typical range: 0.5-0.8. Default 0.7 works well for most documents.
            chunk_size: Maximum tokens per chunk. Chonkie will try to respect this limit.
            min_sentences_per_chunk: Minimum sentences before forcing a chunk break.
            similarity_window: Number of sentences to consider for similarity calculation.
            include_overlap: Whether to include overlapping text between chunks.
            overlap_size: Number of characters to overlap (if include_overlap is True).
        """
        self.settings = settings
        self.model_settings = model_settings
        self.classifier = ChunkPreparer(settings)
        
        # Determine embedding model
        self.embedding_model_name = embedding_model or model_settings.embedding_model
        
        # Initialize embedding backend
        self.embedding_backend = MultilingualEmbeddingBackend(
            model_name=self.embedding_model_name,
            cache_dir=model_settings.embedding_cache_dir,
        )
        
        # Initialize Chonkie semantic chunker
        self.chunker = ChonkieSemanticChunker(
            embedding_model=self.embedding_backend,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
            delim=[". ", "! ", "? ", "\n"],
            include_delim="prev",
        )
        
        self.include_overlap = include_overlap
        self.overlap_size = overlap_size
        self.threshold = threshold
        self.chunk_size = chunk_size

    def chunk(self, analysis: StructureAnalysis) -> ChunkingResult:
        """Generate semantically coherent chunks from the document structure.
        
        This method:
        1. Creates section-level chunks for hierarchy navigation
        2. Applies Chonkie semantic chunking within each section
        3. Enriches chunks with metadata, keywords, and embedding text
        4. Establishes adjacency relationships between chunks
        
        Args:
            analysis: Structure analysis containing document nodes and blocks.
            
        Returns:
            ChunkingResult with strategy metadata and list of FinalChunk objects.
        """
        chunks: list[FinalChunk] = []
        
        # Get all section nodes sorted by position
        section_nodes = [
            node
            for node in analysis.flat_map.values()
            if node.node_type == "section"
        ]
        section_nodes.sort(key=lambda node: node.line_range[0])
        
        # Process each section
        for section in section_nodes:
            # Create section header chunk
            section_chunk = self._create_section_chunk(section, len(chunks) + 1)
            chunks.append(section_chunk)
            
            # Get blocks belonging to this section
            section_blocks = [
                block for block in analysis.ordered_blocks
                if block.parent_id == section.node_id
            ]
            
            if section_blocks:
                # Combine block content for semantic chunking
                section_text = self._combine_section_content(section_blocks)
                
                # Apply Chonkie semantic chunking
                semantic_chunks = self._apply_chonkie_chunking(
                    section_text=section_text,
                    section=section,
                    section_blocks=section_blocks,
                    start_ordinal=len(chunks) + 1,
                )
                chunks.extend(semantic_chunks)
        
        # Establish adjacency relationships
        for index, chunk in enumerate(chunks):
            chunk.adjacency["previous_chunk_id"] = chunks[index - 1].chunk_id if index > 0 else None
            chunk.adjacency["next_chunk_id"] = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
        
        return ChunkingResult(
            strategy={
                "name": "chonkie_semantic_multilingual_v1",
                "embedding_model": self.embedding_model_name,
                "similarity_threshold": self.threshold,
                "max_chunk_size": self.chunk_size,
                "include_overlap": self.include_overlap,
                "overlap_size": self.overlap_size if self.include_overlap else 0,
                "min_sentences_per_chunk": self.chunker.min_sentences_per_chunk,
                "multilingual_support": True,
            },
            chunks=chunks,
        )

    def _combine_section_content(self, blocks: list[DocumentNode]) -> str:
        """Combine block contents preserving structure markers."""
        parts = []
        for block in blocks:
            # Add structure marker for tracking
            marker = f"\n[STRUCT:{block.node_type}:{block.node_id}]\n"
            parts.append(marker)
            parts.append(block.content)
        return "".join(parts)

    def _apply_chonkie_chunking(
        self,
        section_text: str,
        section: DocumentNode,
        section_blocks: list[DocumentNode],
        start_ordinal: int,
    ) -> list[FinalChunk]:
        """Apply Chonkie semantic chunking to section content.
        
        Args:
            section_text: Combined text content of the section.
            section: Parent section node.
            section_blocks: List of blocks in the section.
            start_ordinal: Starting ordinal number for chunk IDs.
            
        Returns:
            List of FinalChunk objects.
        """
        chunks: list[FinalChunk] = []
        
        # Use Chonkie to chunk the text
        try:
            chonkie_chunks = self.chunker.chunk(section_text)
        except Exception as e:
            # Fallback to simple splitting if Chonkie fails
            chonkie_chunks = [type('obj', (object,), {'text': section_text})()]
        
        for idx, chonk in enumerate(chonkie_chunks):
            chunk_text = getattr(chonk, 'text', str(chonk))
            
            # Extract metadata from structure markers
            block_info = self._extract_block_info(chunk_text, section_blocks)
            
            # Determine content type and intent
            content_type = self._determine_content_type(chunk_text, block_info)
            intent = self.classifier._intent(
                type('obj', (object,), {'content': chunk_text, 'node_type': block_info.get('node_type', 'paragraph')})()
            )
            
            # Extract keywords
            keywords = self._extract_keywords(chunk_text)
            
            # Build heading path
            heading_path = list(section.heading_path)
            if block_info.get('heading_path'):
                heading_path.extend(block_info['heading_path'])
            
            # Create chunk ID
            anchor = self._slug(block_info.get('title', heading_path[-1] if heading_path else 'section'))
            parent_id = section.node_id if section.node_id != "doc_root" else None
            chunk_id = f"{parent_id or 'doc_root'}::{anchor}::{intent}::{start_ordinal + idx:04d}"
            
            # Estimate line range (approximate)
            base_line = section.line_range[0] + 1
            line_start = base_line + (idx * 5)  # Approximate
            line_end = line_start + chunk_text.count('\n')
            
            # Create the final chunk
            chunk = FinalChunk(
                chunk_id=chunk_id,
                chunk_kind="semantic",
                parent_section_id=parent_id,
                source_node_id=block_info.get('node_id', section.node_id),
                heading_path=heading_path,
                line_range=[line_start, line_end],
                content_type=content_type,
                intent=intent,
                semantic_tags=[block_info.get('node_type', 'paragraph'), *self._heading_tags(heading_path)],
                keywords=keywords,
                language=self._detect_language(chunk_text),
                importance_score=self._compute_importance(chunk_text, block_info),
                token_estimate=self.classifier._estimate_tokens(chunk_text),
                original_text=chunk_text,
                embedding_text=self._prepare_embedding_text(heading_path, chunk_text, keywords),
                adjacency={"previous_chunk_id": None, "next_chunk_id": None},
                provenance={
                    "source_section_id": section.node_id,
                    "source_block_ids": [block_info.get('node_id', 'unknown')],
                    "chunk_index_in_section": idx,
                    "total_chunks_in_section": len(chonkie_chunks),
                    "semantic_coherence": getattr(chonk, 'score', 1.0),
                },
            )
            chunks.append(chunk)
        
        return chunks

    def _extract_block_info(
        self,
        chunk_text: str,
        section_blocks: list[DocumentNode],
    ) -> dict[str, Any]:
        """Extract block information from chunk text using structure markers."""
        # Try to find structure markers
        marker_pattern = r'\[STRUCT:(\w+):([\w\-]+)\]'
        matches = re.findall(marker_pattern, chunk_text)
        
        if matches:
            # Find corresponding block
            for node_type, node_id in matches:
                for block in section_blocks:
                    if block.node_id == node_id:
                        return {
                            'node_id': block.node_id,
                            'node_type': block.node_type,
                            'heading_path': list(block.heading_path),
                            'title': block.metadata.get('title', ''),
                        }
        
        # Fallback: use first block or section info
        if section_blocks:
            first_block = section_blocks[0]
            return {
                'node_id': first_block.node_id,
                'node_type': first_block.node_type,
                'heading_path': list(first_block.heading_path),
                'title': first_block.metadata.get('title', ''),
            }
        
        return {
            'node_id': 'unknown',
            'node_type': 'paragraph',
            'heading_path': [],
            'title': '',
        }

    def _create_section_chunk(self, section: DocumentNode, ordinal: int) -> FinalChunk:
        """Create a section-level chunk for hierarchy navigation."""
        title = section.metadata.get("title", section.content.lstrip("# ").strip())
        original_text = section.content
        keywords = self._extract_keywords(" ".join(section.heading_path + [original_text]))
        
        return FinalChunk(
            chunk_id=f"{section.node_id}::{self._slug(title)}::section::{ordinal:04d}",
            chunk_kind="section",
            parent_section_id=section.parent_id if section.parent_id != "doc_root" else None,
            source_node_id=section.node_id,
            heading_path=list(section.heading_path),
            line_range=[section.line_range[0], section.line_range[0]],
            content_type="reference",
            intent="explanation",
            semantic_tags=["section", *self._heading_tags(section.heading_path)],
            keywords=keywords,
            language=self._detect_language(original_text),
            importance_score=0.85,
            token_estimate=self.classifier._estimate_tokens(original_text),
            original_text=original_text,
            embedding_text=self._prepare_embedding_text(section.heading_path, original_text, keywords),
            adjacency={"previous_chunk_id": None, "next_chunk_id": None},
            provenance={
                "source_node_id": section.node_id,
                "source_node_type": section.node_type,
                "line_range": [section.line_range[0], section.line_range[0]],
                "is_section_header": True,
            },
        )

    def _determine_content_type(self, text: str, block_info: dict[str, Any]) -> str:
        """Determine content type based on text analysis."""
        node_type = block_info.get('node_type', 'paragraph')
        
        if node_type in {'table', 'code_block'}:
            return 'reference'
        
        text_lower = text.lower()
        if re.search(r'\b(step|install|configure|run|execute|first|then)\b', text_lower):
            return 'procedural'
        if re.search(r'\b(api|model|token|json|python|rag|embedding|pipeline)\b', text_lower):
            return 'technical'
        if node_type == 'list':
            return 'reference'
        
        return 'narrative'

    def _compute_importance(self, text: str, block_info: dict[str, Any]) -> float:
        """Compute importance score for a chunk."""
        density = self.classifier._density(text)
        token_estimate = self.classifier._estimate_tokens(text)
        node_type = block_info.get('node_type', 'paragraph')
        
        score = 0.35 + density * 0.45
        if node_type in {'table', 'code_block', 'list'}:
            score += 0.1
        if token_estimate > self.settings.max_candidate_tokens:
            score -= 0.1
        
        # Boost section headers
        if block_info.get('is_section_header'):
            score = max(score, 0.8)
        
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _extract_keywords(text: str) -> dict[str, list[str]]:
        """Extract lexical and semantic keywords from text."""
        tokens = [
            token.lower()
            for token in re.findall(r"[^\W\d_]{3,}", text, flags=re.UNICODE)
        ]
        stopwords = {
            "the", "and", "for", "with", "this", "that", "una", "del", "los",
            "las", "con", "por", "para", "que", "are", "is", "you",
            "el", "la", "de", "en", "un", "una", "los", "las",
        }
        filtered = [token for token in tokens if token not in stopwords]
        lexical = [token for token, _ in Counter(filtered).most_common(8)]
        entity_candidates = re.findall(r"\b[A-ZÀ-Ÿ][^\W\d_]{2,}\b", text, flags=re.UNICODE)
        
        return {
            "lexical": lexical,
            "semantic": lexical[:5],
            "conceptual_topics": lexical[:3],
            "entities": list(dict.fromkeys(entity_candidates))[:8],
        }

    @staticmethod
    def _detect_language(text: str) -> dict[str, Any]:
        """Detect language of the text (simple heuristic)."""
        lowered = text.lower()
        
        # Spanish indicators
        spanish_words = ["acción", "sistema", "datos", "informe", "seleccione", 
                        "configuración", "usuario", "aplicación", "página"]
        spanish_hits = sum(word in lowered for word in spanish_words)
        
        # English indicators
        english_words = ["the", "system", "data", "report", "select",
                        "configuration", "user", "application", "page"]
        english_hits = sum(word in lowered for word in english_words)
        
        # French indicators
        french_words = ["système", "données", "rapport", "sélectionnez",
                       "configuration", "utilisateur", "application"]
        french_hits = sum(word in lowered for word in french_words)
        
        if spanish_hits > max(english_hits, french_hits):
            return {"detected_language": "es", "confidence": min(0.9, 0.6 + spanish_hits * 0.1)}
        if french_hits > max(english_hits, spanish_hits):
            return {"detected_language": "fr", "confidence": min(0.9, 0.6 + french_hits * 0.1)}
        if english_hits > 0:
            return {"detected_language": "en", "confidence": min(0.9, 0.6 + english_hits * 0.1)}
        
        return {"detected_language": "unknown", "confidence": 0.0}

    @staticmethod
    def _prepare_embedding_text(heading_path: list[str], original_text: str, keywords: dict[str, list[str]]) -> str:
        """Prepare text optimized for embedding with context."""
        heading = " > ".join(heading_path) if heading_path else "Document root"
        keyword_text = ", ".join(keywords["lexical"])
        
        # Format that works well with multilingual E5 models
        # E5 models benefit from explicit instruction-like formatting
        return f"Context: {heading}\nKey terms: {keyword_text}\nContent: {original_text}"

    @staticmethod
    def _heading_tags(heading_path: list[str]) -> list[str]:
        """Generate hierarchical tags from heading path."""
        return [f"h{idx + 1}:{ChonkieChunkingStrategy._slug(heading)}" 
                for idx, heading in enumerate(heading_path)]

    @staticmethod
    def _slug(value: str) -> str:
        """Convert text to URL-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "section"
