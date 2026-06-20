"""Model loader for intfloat/multilingual-e5-large with caching."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingModelLoader:
    """Load and cache the multilingual-e5-large embedding model.
    
    This loader ensures the model is downloaded once to a local directory
    and reused on subsequent runs without redownloading.
    """
    
    MODEL_NAME = "intfloat/multilingual-e5-large"
    DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "models"
    
    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize the model loader.
        
        Args:
            cache_dir: Directory to store the downloaded model. 
                      Defaults to ./models in the project root.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.model_path = self.cache_dir / self.MODEL_NAME.replace("/", "_")
        self._model: SentenceTransformer | None = None
    
    def download_model(self, force: bool = False) -> Path:
        """Download the model if not already cached.
        
        Args:
            force: If True, force redownload even if model exists.
            
        Returns:
            Path to the cached model directory.
        """
        if self.model_path.exists() and not force:
            return self.model_path
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and save the model to our cache directory
        model = SentenceTransformer(self.MODEL_NAME)
        model.save(str(self.model_path))
        
        return self.model_path
    
    def load_model(self, auto_download: bool = True) -> SentenceTransformer:
        """Load the embedding model.
        
        Args:
            auto_download: If True and model not cached, download it first.
            
        Returns:
            Loaded SentenceTransformer model.
            
        Raises:
            FileNotFoundError: If model not found and auto_download is False.
        """
        if self._model is not None:
            return self._model
        
        if not self.model_path.exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Set auto_download=True to download automatically."
                )
            self.download_model()
        
        # Load from local cache
        self._model = SentenceTransformer(str(self.model_path))
        return self._model
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.MODEL_NAME,
            "cache_dir": str(self.cache_dir),
            "model_path": str(self.model_path),
            "is_cached": self.model_path.exists(),
        }
