# viralStoryGenerator/utils/vector_db_manager.py
"""Vector database manager using ChromaDB."""
import os
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings

from viralStoryGenerator.utils.config import config as app_config
import logging

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

_client = None
_embedding_function = None

def get_client():
    """Get or initialize the ChromaDB client."""
    global _client
    if _client is not None:
        return _client

    # Only initialize if RAG is enabled
    if not app_config.rag.ENABLED:
        _logger.warning("RAG is disabled in configuration. Vector DB operations will not work.")
        return None

    try:
        persist_directory = app_config.rag.VECTOR_DB_PATH
        os.makedirs(persist_directory, exist_ok=True)

        _logger.info(f"ChromaDB client initialized. Storage path: {persist_directory}")
        _client = chromadb.PersistentClient(
            path=persist_directory
        )
        return _client
    except Exception as e:
        _logger.exception(f"Failed to initialize ChromaDB client: {e}")
        return None

def get_embedding_function():
    """Get or initialize the embedding function."""
    global _embedding_function
    if _embedding_function is not None:
        return _embedding_function

    # Only initialize if RAG is enabled and client exists
    if not app_config.rag.ENABLED or get_client() is None:
        return None

    try:
        embedding_model_name = app_config.rag.EMBEDDING_MODEL
        _logger.info(f"Initializing sentence-transformer embedding model: {embedding_model_name}")

        from chromadb.utils import embedding_functions
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        _logger.info(f"Successfully initialized embedding model: {embedding_model_name}")
        return _embedding_function
    except Exception as e:
        _logger.exception(f"Failed to initialize embedding function: {e}")
        return None

def add_chunks_to_collection(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> bool:
    """Add chunks to a collection with proper lazy initialization."""
    client = get_client()
    if not client:
        _logger.error("ChromaDB client not available. Cannot add chunks.")
        return False

    embedding_func = get_embedding_function()
    if not embedding_func:
        _logger.error("Embedding function not available. Cannot add chunks.")
        return False

    try:
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
            _logger.debug(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
            _logger.debug(f"Created new collection: {collection_name}")

        _logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'...")

        # Add documents in a single batch
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids or [f"id_{i}" for i in range(len(documents))]
        )

        _logger.info(f"Successfully added documents to collection '{collection_name}'.")
        return True
    except Exception as e:
        _logger.exception(f"Failed to add documents to collection '{collection_name}': {e}")
        return False

def query_collection(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Query a collection with proper lazy initialization."""
    client = get_client()
    if not client:
        _logger.error("ChromaDB client not available. Cannot query collection.")
        return {"documents": [], "metadatas": [], "distances": []}

    embedding_func = get_embedding_function()
    if not embedding_func:
        _logger.error("Embedding function not available. Cannot query collection.")
        return {"documents": [], "metadatas": [], "distances": []}

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # TODO: Include 'distances' in chromadb.collection.query include_list for relevance scoring and add handling for it in consuming functions.
        )
        _logger.info(f"Query successful for collection '{collection_name}'. Found results.")
        return results
    except Exception as e:
        # Handle case where collection might not exist yet if query happens before add
        if "does not exist" in str(e):
             _logger.warning(f"Collection '{collection_name}' not found during query.")
             return {"documents": [], "metadatas": [], "distances": []}
        _logger.exception(f"Failed to query ChromaDB collection '{collection_name}': {e}")
        return {"documents": [], "metadatas": [], "distances": []}

def delete_collection(collection_name: str) -> bool:
    """Delete a collection with proper lazy initialization."""
    client = get_client()
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
    """Explicitly close the Chroma client and release its resources."""
    global _client, _embedding_function
    if _client:
        try:
            # The ChromaDB client doesn't always have a close method
            # Try several known cleanup approaches

            # First try close method (if available in newer versions)
            if hasattr(_client, "close"):
                _client.close()
                _logger.info("ChromaDB client closed via close() method")
            # Try alternate cleanup methods if available
            elif hasattr(_client, "_clean_up"):
                _client._clean_up()
                _logger.info("ChromaDB client closed via _clean_up() method")
            # If no explicit close method, try to cleanup known resources
            elif hasattr(_client, "_server") and hasattr(_client._server, "close"):
                _client._server.close()
                _logger.info("ChromaDB server component closed")

            # Force Python's garbage collection
            import gc
            gc.collect()

            _logger.info("ChromaDB client resources released")
        except Exception as e:
            _logger.error(f"Error during ChromaDB client cleanup: {e}")
        finally:
            _client = None
            _embedding_function = None
    else:
        _logger.debug("No ChromaDB client to close")

def get_vector_db_client():
    """Get or initialize the ChromaDB client."""
    return get_client()

