# viralStoryGenerator/utils/vector_db_manager.py
"""Manages interactions with the ChromaDB vector database."""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional, Dict, Any

from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger
from .embedding_generator import get_embedding_function

# Initialize ChromaDB client
try:
    # Use persistent storage
    client = chromadb.PersistentClient(path=app_config.rag.VECTOR_DB_PATH)
    _logger.info(f"ChromaDB client initialized. Storage path: {app_config.rag.VECTOR_DB_PATH}")
except Exception as e:
    _logger.exception(f"Failed to initialize ChromaDB client: {e}")
    client = None

# Get the embedding function based on config
embed_func = get_embedding_function()

def add_chunks_to_collection(collection_name: str, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
    """Adds text chunks and their metadata to a ChromaDB collection."""
    if not client:
        _logger.error("ChromaDB client not available. Cannot add chunks.")
        return False
    if not embed_func:
        _logger.error("Embedding function not available. Cannot add chunks.")
        return False
    if not (len(chunks) == len(metadatas) == len(ids)):
         _logger.error("Mismatch between chunks, metadatas, and ids count.")
         return False
    if not chunks:
         _logger.warning(f"No chunks provided to add to collection '{collection_name}'.")
         return True # Nothing to add is not an error

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_func
            # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric
        )
        _logger.info(f"Adding {len(chunks)} chunks to collection '{collection_name}'...")
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        _logger.info(f"Successfully added chunks to collection '{collection_name}'.")
        return True
    except Exception as e:
        _logger.exception(f"Failed to add chunks to ChromaDB collection '{collection_name}': {e}")
        return False

def query_collection(collection_name: str, query_texts: List[str], n_results: int = 5) -> Optional[Dict[str, Any]]:
    """Queries a ChromaDB collection for relevant documents."""
    if not client:
        _logger.error("ChromaDB client not available. Cannot query collection.")
        return None
    if not embed_func:
        _logger.error("Embedding function not available. Cannot query collection.")
        return None

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_func
        )
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Include distances for potential relevance scoring
        )
        _logger.info(f"Query successful for collection '{collection_name}'. Found results.")
        return results
    except Exception as e:
        # Handle case where collection might not exist yet if query happens before add
        if "does not exist" in str(e):
             _logger.warning(f"Collection '{collection_name}' not found during query.")
             return None
        _logger.exception(f"Failed to query ChromaDB collection '{collection_name}': {e}")
        return None

def delete_collection(collection_name: str) -> bool:
    """Deletes a ChromaDB collection."""
    if not client:
        _logger.error("ChromaDB client not available. Cannot delete collection.")
        return False
    try:
        client.delete_collection(name=collection_name)
        _logger.info(f"Successfully deleted ChromaDB collection: {collection_name}")
        return True
    except Exception as e:
        _logger.exception(f"Failed to delete ChromaDB collection '{collection_name}': {e}")
        return False

def close_client():
    """Resets the ChromaDB client, releasing resources."""
    global client
    if client:
        try:
            _logger.info("Resetting ChromaDB client...")
            client.reset() # Resets the client state, effective for PersistentClient
            client = None # Allow reinitialization if needed
            _logger.info("ChromaDB client reset successfully.")
        except Exception as e:
            _logger.exception(f"Error resetting ChromaDB client: {e}")

