# viralStoryGenerator/utils/embedding_generator.py
"""Handles text embedding generation using sentence-transformers."""

from typing import Optional
from chromadb.utils import embedding_functions

from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger

_embedding_function = None

def get_embedding_function():
    """Initializes and returns the configured sentence-transformer embedding function."""
    global _embedding_function
    if _embedding_function is None:
        model_name = app_config.rag.EMBEDDING_MODEL
        _logger.info(f"Initializing sentence-transformer embedding model: {model_name}")
        try:
            # ChromaDB's helper function simplifies using sentence-transformers
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            # Test the function
            _embedding_function(["test"])
            _logger.info(f"Successfully initialized embedding model: {model_name}")
        except Exception as e:
            _logger.exception(f"Failed to initialize embedding model '{model_name}': {e}")
            _embedding_function = None
    return _embedding_function
