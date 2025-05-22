import pytest
import os
from unittest.mock import patch, MagicMock

# Assuming the module is viralStoryGenerator.utils.vector_db_manager
from viralStoryGenerator.utils import vector_db_manager as vdbm_module
from viralStoryGenerator.utils.config import app_config # For patching config values
# Import chromadb specific errors if they are caught and handled, e.g.
# from chromadb.exceptions import CollectionNotFoundError # Example

# --- Global Mocks & Fixtures ---

MOCK_VECTOR_DB_PATH = "/tmp/mock_chroma_db"

@pytest.fixture(autouse=True)
def mock_appconfig_vdb_defaults(monkeypatch):
    """Set default app_config values for vector_db_manager tests."""
    monkeypatch.setattr(app_config.rag, 'ENABLED', True) # Default to RAG enabled
    monkeypatch.setattr(app_config.rag, 'VECTOR_DB_PATH', MOCK_VECTOR_DB_PATH)
    monkeypatch.setattr(app_config.rag, 'EMBEDDING_MODEL', "test_embedding_model_vdb")
    
    # Ensure the mock path is "clean" for some tests, though os functions will be mocked.
    # For tests involving actual directory creation (if any, though unlikely in unit tests),
    # this might need actual cleanup. For now, os.makedirs will be mocked.
    yield
    # Cleanup if any test actually creates the mock path and doesn't clean itself
    # if os.path.exists(MOCK_VECTOR_DB_PATH) and "mock_chroma_db" in MOCK_VECTOR_DB_PATH:
    #     import shutil
    #     shutil.rmtree(MOCK_VECTOR_DB_PATH, ignore_errors=True)


@pytest.fixture
def mock_vdb_logger():
    """Fixture to mock the _logger in vector_db_manager.py."""
    with patch('viralStoryGenerator.utils.vector_db_manager._logger') as mock_logger:
        yield mock_logger

@pytest.fixture(autouse=True)
def reset_vdb_module_globals(monkeypatch):
    """Reset global client and embedding function in vector_db_manager.py before each test."""
    monkeypatch.setattr(vdbm_module, '_client', None)
    monkeypatch.setattr(vdbm_module, '_embedding_function', None)
    # If CHROMADB_AVAILABLE is dynamically set and tested, reset it too.
    # For now, assume tests will control it directly if needed.
    monkeypatch.setattr(vdbm_module, 'CHROMADB_AVAILABLE', True) # Default to True for most tests


# --- Mocks for chromadb library ---
@pytest.fixture
def mock_persistent_client_class():
    """Mocks chromadb.PersistentClient class."""
    # This mock will be used to control the instance returned by PersistentClient()
    with patch('chromadb.PersistentClient') as MockClientClass:
        mock_client_instance = MagicMock()
        # Add expected methods/attributes to the instance if needed by get_client or other functions
        # e.g., mock_client_instance.get_collection = MagicMock()
        # mock_client_instance.create_collection = MagicMock()
        # mock_client_instance.delete_collection = MagicMock()
        # mock_client_instance.reset = MagicMock() # For close_client tests
        # mock_client_instance.clear_pending_collections_for_deletion = MagicMock() # For close_client tests
        MockClientClass.return_value = mock_client_instance
        yield MockClientClass, mock_client_instance

@pytest.fixture
def mock_embedding_generator_func():
    """Mocks embedding_generator.get_embedding_function."""
    with patch('viralStoryGenerator.utils.embedding_generator.get_embedding_function') as mock_get_ef:
        # Default behavior: return a mock embedding function instance
        mock_ef_instance = MagicMock(name="MockEmbeddingFunctionInstance")
        mock_get_ef.return_value = mock_ef_instance
        yield mock_get_ef, mock_ef_instance

# --- Tests for get_client (Scenario 1) ---

@patch('os.makedirs')
@patch('os.path.exists')
def test_get_client_successful_initialization(
    mock_os_exists, mock_os_makedirs,
    mock_persistent_client_class, # Fixture providing (MockClass, mock_instance)
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    MockClientClass, mock_client_instance = mock_persistent_client_class
    mock_os_exists.return_value = False # Simulate DB path does not exist

    client = vdbm_module.get_client()

    assert client is mock_client_instance
    mock_os_exists.assert_called_once_with(MOCK_VECTOR_DB_PATH)
    mock_os_makedirs.assert_called_once_with(MOCK_VECTOR_DB_PATH, exist_ok=True)
    MockClientClass.assert_called_once_with(path=MOCK_VECTOR_DB_PATH)
    mock_vdb_logger.info.assert_any_call(f"ChromaDB client initialized. DB Path: {MOCK_VECTOR_DB_PATH}")


@patch('os.makedirs') # Should not be called if path exists
@patch('os.path.exists', return_value=True) # Simulate DB path already exists
def test_get_client_path_already_exists(
    mock_os_exists, mock_os_makedirs,
    mock_persistent_client_class, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    MockClientClass, _ = mock_persistent_client_class
    
    client = vdbm_module.get_client()

    assert client is not None
    mock_os_exists.assert_called_once_with(MOCK_VECTOR_DB_PATH)
    mock_os_makedirs.assert_not_called() # Not called because path exists
    MockClientClass.assert_called_once_with(path=MOCK_VECTOR_DB_PATH)


def test_get_client_memoization(
    mock_persistent_client_class, mock_appconfig_vdb_defaults, monkeypatch
):
    MockClientClass, mock_client_instance_first_call = mock_persistent_client_class
    
    # First call
    with patch('os.path.exists', return_value=True), patch('os.makedirs'): # Ensure path logic doesn't interfere
        client1 = vdbm_module.get_client()
    
    assert client1 is mock_client_instance_first_call
    MockClientClass.assert_called_once() # Called once for first client

    # Second call - should return the same instance
    # Reset the call count of the class mock to ensure it's not called again for instantiation
    MockClientClass.reset_mock() 
    
    with patch('os.path.exists', return_value=True), patch('os.makedirs'):
        client2 = vdbm_module.get_client()
        
    assert client2 is client1 # Same instance
    MockClientClass.assert_not_called() # Not called again


def test_get_client_rag_disabled(
    mock_persistent_client_class, mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', False)
    MockClientClass, _ = mock_persistent_client_class

    client = vdbm_module.get_client()

    assert client is None
    MockClientClass.assert_not_called()
    mock_vdb_logger.info.assert_any_call("RAG is disabled. ChromaDB client will not be initialized.")


@patch('os.path.exists', return_value=True) # Assume path exists
def test_get_client_chromadb_not_available(
    mock_os_exists, mock_persistent_client_class, mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(vdbm_module, 'CHROMADB_AVAILABLE', False) # Simulate chromadb not installed
    MockClientClass, _ = mock_persistent_client_class

    client = vdbm_module.get_client()

    assert client is None
    MockClientClass.assert_not_called()
    mock_vdb_logger.critical.assert_any_call(
        "ChromaDB library is not installed, but RAG is enabled. "
        "Vector DB functionality will be unavailable. Please install with `pip install chromadb`."
    )


@patch('os.path.exists', return_value=True)
def test_get_client_persistent_client_instantiation_error(
    mock_os_exists, mock_persistent_client_class, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    MockClientClass, _ = mock_persistent_client_class
    init_exception = Exception("Failed to initialize PersistentClient")
    MockClientClass.side_effect = init_exception

    client = vdbm_module.get_client()

    assert client is None
    MockClientClass.assert_called_once_with(path=MOCK_VECTOR_DB_PATH)
    mock_vdb_logger.error.assert_any_call(
        f"Failed to initialize ChromaDB PersistentClient: {init_exception}", exc_info=True
    )

# --- Tests for get_embedding_function (Scenario 2) ---

@patch.object(vdbm_module, 'get_client') # Mock get_client within the vdb_manager module
def test_get_embedding_function_rag_enabled_client_available(
    mock_vdb_get_client, mock_embedding_generator_func, # Fixtures
    mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    mock_vdb_get_client.return_value = MagicMock() # Simulate client is available
    
    mock_get_ef_func, mock_ef_instance = mock_embedding_generator_func
    mock_get_ef_func.return_value = mock_ef_instance # Ensure it returns the mock EF

    ef = vdbm_module.get_embedding_function()

    assert ef is mock_ef_instance
    mock_vdb_get_client.assert_called_once()
    mock_get_ef_func.assert_called_once() # embedding_generator.get_embedding_function was called


def test_get_embedding_function_rag_disabled(
    mock_embedding_generator_func, mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', False)
    mock_get_ef_func, _ = mock_embedding_generator_func

    # Patch get_client to ensure it's not the reason for returning None
    with patch.object(vdbm_module, 'get_client', return_value=MagicMock()):
        ef = vdbm_module.get_embedding_function()

    assert ef is None
    mock_get_ef_func.assert_not_called()
    mock_vdb_logger.debug.assert_any_call(
        "RAG is disabled. Embedding function will not be initialized."
    )


@patch.object(vdbm_module, 'get_client', return_value=None) # Mock get_client to return None
def test_get_embedding_function_client_not_available(
    mock_vdb_get_client_none, mock_embedding_generator_func, 
    mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', True) # RAG is enabled
    mock_get_ef_func, _ = mock_embedding_generator_func

    ef = vdbm_module.get_embedding_function()

    assert ef is None
    mock_vdb_get_client_none.assert_called_once() # Attempted to get client
    mock_get_ef_func.assert_not_called() # But didn't proceed to get EF
    mock_vdb_logger.warning.assert_any_call(
        "ChromaDB client is not available. Cannot initialize embedding function."
    )


@patch.object(vdbm_module, 'get_client')
def test_get_embedding_function_memoization(
    mock_vdb_get_client, mock_embedding_generator_func,
    mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    mock_vdb_get_client.return_value = MagicMock() # Client is available
    
    mock_get_ef_func, mock_ef_instance = mock_embedding_generator_func
    mock_get_ef_func.return_value = mock_ef_instance

    # First call
    ef1 = vdbm_module.get_embedding_function()
    assert ef1 is mock_ef_instance
    mock_get_ef_func.assert_called_once()
    
    # Second call
    ef2 = vdbm_module.get_embedding_function()
    assert ef2 is mock_ef_instance
    mock_get_ef_func.assert_called_once() # Still called only once due to memoization in embedding_generator
                                         # and _embedding_function in vdb_manager
    # Check that vdb_manager's own _embedding_function is memoized
    assert vdbm_module._embedding_function is mock_ef_instance

# --- Tests for add_chunks_to_collection (Scenario 3) ---

@pytest.fixture
def mock_collection_instance():
    """Provides a mock chromadb.Collection instance."""
    collection = MagicMock()
    collection.add = MagicMock() # Mock the add method
    return collection

# 3.1: Collection exists
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_add_chunks_collection_exists(
    mock_get_ef, mock_get_client, mock_collection_instance,
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection_instance
    mock_get_client.return_value = mock_client_instance
    
    mock_embedding_function_instance = MagicMock()
    mock_get_ef.return_value = mock_embedding_function_instance

    collection_name = "test_collection_exists"
    chunks = ["chunk1", "chunk2"]
    metadata = [{"source": "doc1"}, {"source": "doc2"}]
    ids = ["id1", "id2"]

    result = vdbm_module.add_chunks_to_collection(collection_name, chunks, metadata, ids)

    assert result is True
    mock_client_instance.get_collection.assert_called_once_with(
        name=collection_name, embedding_function=mock_embedding_function_instance
    )
    mock_client_instance.create_collection.assert_not_called()
    mock_collection_instance.add.assert_called_once_with(
        documents=chunks,
        metadatas=metadata,
        ids=ids
    )
    mock_vdb_logger.info.assert_any_call(
        f"Adding {len(chunks)} chunks to existing collection '{collection_name}'."
    )


# 3.2: Collection does not exist (creation)
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_add_chunks_collection_creates_new(
    mock_get_ef, mock_get_client, mock_collection_instance,
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    # Simulate get_collection raising an exception (e.g., CollectionNotFoundError, or just generic Exception as per code)
    # The code catches generic Exception for get_collection.
    mock_client_instance.get_collection.side_effect = Exception("Collection not found")
    mock_client_instance.create_collection.return_value = mock_collection_instance # create_collection returns the new collection
    mock_get_client.return_value = mock_client_instance
    
    mock_embedding_function_instance = MagicMock()
    mock_get_ef.return_value = mock_embedding_function_instance

    collection_name = "test_collection_new"
    chunks = ["new_chunk"]
    metadata = [{"source": "new_doc"}]
    ids = ["new_id"]

    result = vdbm_module.add_chunks_to_collection(collection_name, chunks, metadata, ids)

    assert result is True
    mock_client_instance.get_collection.assert_called_once_with(
        name=collection_name, embedding_function=mock_embedding_function_instance
    )
    mock_client_instance.create_collection.assert_called_once_with(
        name=collection_name, embedding_function=mock_embedding_function_instance
    )
    mock_collection_instance.add.assert_called_once_with(
        documents=chunks, metadatas=metadata, ids=ids
    )
    mock_vdb_logger.info.assert_any_call(
        f"Collection '{collection_name}' not found or error accessing, attempting to create."
    )
    mock_vdb_logger.info.assert_any_call(
        f"Successfully created and got collection '{collection_name}'."
    )
    mock_vdb_logger.info.assert_any_call(
        f"Adding {len(chunks)} chunks to new collection '{collection_name}'."
    )


# 3.3: Client not available
@patch.object(vdbm_module, 'get_client', return_value=None) # get_client returns None
def test_add_chunks_client_not_available(
    mock_get_client_none, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    result = vdbm_module.add_chunks_to_collection("no_client_coll", ["c"], [{"s":"d"}], ["id"])
    assert result is False
    mock_vdb_logger.error.assert_called_once_with("ChromaDB client not available. Cannot add chunks.")


# 3.4: Embedding function not available
@patch.object(vdbm_module, 'get_client', return_value=MagicMock()) # Client is available
@patch.object(vdbm_module, 'get_embedding_function', return_value=None) # EF is not
def test_add_chunks_embedding_function_not_available(
    mock_get_ef_none, mock_get_client_ef_test, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    result = vdbm_module.add_chunks_to_collection("no_ef_coll", ["c"], [{"s":"d"}], ["id"])
    assert result is False
    mock_vdb_logger.error.assert_called_once_with("Embedding function not available. Cannot add chunks.")


# 3.5: Error during collection.add
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_add_chunks_collection_add_error(
    mock_get_ef, mock_get_client, mock_collection_instance,
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection_instance
    mock_get_client.return_value = mock_client_instance
    mock_get_ef.return_value = MagicMock() # EF is available
    
    add_exception = Exception("Failed to add documents to Chroma")
    mock_collection_instance.add.side_effect = add_exception

    collection_name = "test_collection_add_err"
    result = vdbm_module.add_chunks_to_collection(collection_name, ["c"], [{"s":"d"}], ["id"])

    assert result is False
    mock_collection_instance.add.assert_called_once()
    mock_vdb_logger.error.assert_called_once_with(
        f"Failed to add chunks to collection '{collection_name}'. Error: {add_exception}", exc_info=True
    )


# 3.6: Error during client.create_collection
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_add_chunks_create_collection_error(
    mock_get_ef, mock_get_client, # Don't use mock_collection_instance as it won't be returned
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.side_effect = Exception("Collection not found initially")
    
    create_exception = Exception("Failed to create collection in Chroma")
    mock_client_instance.create_collection.side_effect = create_exception
    mock_get_client.return_value = mock_client_instance
    mock_get_ef.return_value = MagicMock()

    collection_name = "test_collection_create_err"
    result = vdbm_module.add_chunks_to_collection(collection_name, ["c"], [{"s":"d"}], ["id"])

    assert result is False
    mock_client_instance.get_collection.assert_called_once()
    mock_client_instance.create_collection.assert_called_once()
    mock_vdb_logger.error.assert_any_call( # Updated to any_call due to multiple logs
        f"Failed to create collection '{collection_name}'. Error: {create_exception}", exc_info=True
    )

# --- Tests for query_collection (Scenario 4) ---

# 4.1: Successful query
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_query_collection_successful(
    mock_get_ef, mock_get_client, mock_collection_instance, # mock_collection_instance from fixture
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection_instance
    mock_get_client.return_value = mock_client_instance
    
    mock_embedding_function_instance = MagicMock()
    mock_get_ef.return_value = mock_embedding_function_instance

    collection_name = "test_query_collection_ok"
    query_texts = ["query text 1"]
    n_results = 5
    where_filter = {"source": "doc_source"}
    include_fields = ["metadatas", "documents", "distances"]
    
    # Mock collection.query response structure
    # Example: {'ids': [['id1']], 'distances': [[0.1]], 'metadatas': [[{'source': 'doc1'}]], 'documents': [['text1']], 'uris': None, 'data': None}
    mock_query_results_dict = {
        "ids": [["res_id1", "res_id2"]],
        "documents": [["doc text 1", "doc text 2"]],
        "metadatas": [[{"source": "doc_source"}, {"source": "another"}]],
        "distances": [[0.1, 0.2]]
    }
    mock_collection_instance.query.return_value = mock_query_results_dict

    results = vdbm_module.query_collection(
        collection_name, query_texts, n_results, where_filter, include_fields
    )

    assert results == mock_query_results_dict
    mock_client_instance.get_collection.assert_called_once_with(
        name=collection_name, embedding_function=mock_embedding_function_instance
    )
    mock_collection_instance.query.assert_called_once_with(
        query_texts=query_texts,
        n_results=n_results,
        where=where_filter,
        include=include_fields
    )
    mock_vdb_logger.debug.assert_any_call(
        f"Querying collection '{collection_name}' with {len(query_texts)} query texts. N_results: {n_results}."
    )


# 4.2: Collection not found
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_query_collection_not_found(
    mock_get_ef, mock_get_client, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    # Simulate get_collection raising an exception (e.g., CollectionNotFoundError or generic)
    get_coll_exception = Exception("Collection 'query_coll_not_found' does not exist.")
    mock_client_instance.get_collection.side_effect = get_coll_exception
    mock_get_client.return_value = mock_client_instance
    
    mock_get_ef.return_value = MagicMock() # EF is available

    collection_name = "query_coll_not_found"
    results = vdbm_module.query_collection(collection_name, ["q"], 3)

    assert results == {"ids": [], "documents": [], "metadatas": [], "distances": []} # Empty results
    mock_client_instance.get_collection.assert_called_once()
    mock_vdb_logger.warning.assert_any_call(
        f"Collection '{collection_name}' not found or error accessing. Error: {get_coll_exception}"
    )


# 4.3: Client or embedding function not available
@pytest.mark.parametrize("client_available, ef_available", [
    (False, True), # Client None, EF available (mocked)
    (True, False), # Client available (mocked), EF None
])
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_query_collection_client_or_ef_not_available(
    mock_get_ef, mock_get_client, client_available, ef_available,
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    if not client_available:
        mock_get_client.return_value = None
    else:
        mock_get_client.return_value = MagicMock() # Client is available
        
    if not ef_available:
        mock_get_ef.return_value = None
    else:
        mock_get_ef.return_value = MagicMock() # EF is available

    results = vdbm_module.query_collection("coll_no_client_ef", ["q"], 3)
    
    assert results == {"ids": [], "documents": [], "metadatas": [], "distances": []}
    if not client_available:
        mock_vdb_logger.error.assert_any_call("ChromaDB client not available. Cannot query collection.")
    elif not ef_available:
        mock_vdb_logger.error.assert_any_call("Embedding function not available. Cannot query collection.")


# 4.4: Error during collection.query
@patch.object(vdbm_module, 'get_client')
@patch.object(vdbm_module, 'get_embedding_function')
def test_query_collection_query_error(
    mock_get_ef, mock_get_client, mock_collection_instance,
    mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection_instance
    mock_get_client.return_value = mock_client_instance
    mock_get_ef.return_value = MagicMock()

    query_exception = Exception("Chroma query failed")
    mock_collection_instance.query.side_effect = query_exception
    
    collection_name = "test_collection_query_err"
    results = vdbm_module.query_collection(collection_name, ["q"], 3)

    assert results == {"ids": [], "documents": [], "metadatas": [], "distances": []}
    mock_collection_instance.query.assert_called_once()
    mock_vdb_logger.error.assert_any_call(
        f"Failed to query collection '{collection_name}'. Error: {query_exception}", exc_info=True
    )

# --- Tests for delete_collection (Scenario 5) ---

@patch.object(vdbm_module, 'get_client')
def test_delete_collection_successful(
    mock_get_client, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    mock_client_instance.delete_collection = MagicMock() # delete_collection is synchronous
    mock_get_client.return_value = mock_client_instance
    
    collection_name = "test_coll_to_delete"
    result = vdbm_module.delete_collection(collection_name)

    assert result is True
    mock_client_instance.delete_collection.assert_called_once_with(name=collection_name)
    mock_vdb_logger.info.assert_any_call(f"Successfully deleted collection '{collection_name}'.")


@patch.object(vdbm_module, 'get_client', return_value=None) # Client not available
def test_delete_collection_client_not_available(
    mock_get_client_none, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    result = vdbm_module.delete_collection("no_client_delete_coll")
    assert result is False
    mock_vdb_logger.error.assert_called_once_with("ChromaDB client not available. Cannot delete collection.")


@patch.object(vdbm_module, 'get_client')
def test_delete_collection_delete_error(
    mock_get_client, mock_vdb_logger, mock_appconfig_vdb_defaults
):
    mock_client_instance = MagicMock()
    delete_exception = Exception("Chroma delete_collection failed")
    mock_client_instance.delete_collection.side_effect = delete_exception
    mock_get_client.return_value = mock_client_instance
    
    collection_name = "test_coll_delete_err"
    result = vdbm_module.delete_collection(collection_name)

    assert result is False
    mock_client_instance.delete_collection.assert_called_once_with(name=collection_name)
    mock_vdb_logger.error.assert_called_once_with(
        f"Failed to delete collection '{collection_name}'. Error: {delete_exception}", exc_info=True
    )


# --- Tests for close_client (Scenario 6) ---

@patch.object(vdbm_module, '_client', new=None) # Ensure _client is None initially for some tests
def test_close_client_no_client_initialized(mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch):
    # Ensure _client is None by resetting via fixture or explicitly for this test
    monkeypatch.setattr(vdbm_module, '_client', None)
    
    vdbm_module.close_client()
    mock_vdb_logger.info.assert_any_call("ChromaDB client not initialized or already closed. No action taken.")


@patch.object(vdbm_module, 'get_client') # To ensure _client is set via a call to get_client first
def test_close_client_calls_reset_and_clears_pending_collections(
    mock_get_client_for_close, mock_persistent_client_class, # Use this to get the instance that get_client would set
    mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    # This test assumes ChromaDB client has a reset() method and potentially a way to clear pending deletions.
    # The actual methods depend on the chromadb version.
    # `_client.clear_pending_collections_for_deletion()` was in 0.4.23+
    # `_client._system.delete_pending_collections()` was for < 0.4.23
    # `_client.reset()` is a general method.
    
    MockClientClass, mock_client_instance_internal = mock_persistent_client_class
    
    # Simulate get_client() having been called and _client is set
    # Patch os.path.exists for get_client() if it's called
    with patch('os.path.exists', return_value=True), patch('os.makedirs'):
        # First, call get_client() to set the global _client to our mock_client_instance_internal
        # We need to ensure that the _client that close_client operates on is our mock.
        # The mock_persistent_client_class fixture ensures PersistentClient() returns mock_client_instance_internal.
        # So, the first call to vdbm_module.get_client() will set vdbm_module._client.
        vdbm_module.get_client() 
    
    # Ensure the global _client is indeed our mock instance
    assert vdbm_module._client is mock_client_instance_internal

    # Add specific methods to the mock instance for testing close_client paths
    mock_client_instance_internal.reset = MagicMock()
    
    # For ChromaDB >= 0.4.23
    mock_client_instance_internal.clear_pending_collections_for_deletion = MagicMock() 
    
    # For ChromaDB < 0.4.23 (mocking a nested structure)
    mock_system_obj = MagicMock()
    mock_system_obj.delete_pending_collections = MagicMock()
    mock_client_instance_internal._system = mock_system_obj

    vdbm_module.close_client()

    mock_client_instance_internal.reset.assert_called_once() # General cleanup
    
    # Check which cleanup path was taken based on attribute presence
    # The code tries `clear_pending_collections_for_deletion` first.
    if hasattr(mock_client_instance_internal, 'clear_pending_collections_for_deletion'):
        mock_client_instance_internal.clear_pending_collections_for_deletion.assert_called_once()
        mock_system_obj.delete_pending_collections.assert_not_called()
        mock_vdb_logger.info.assert_any_call("ChromaDB client: Cleared pending collections for deletion (>=0.4.23 method).")
    elif hasattr(mock_client_instance_internal, '_system') and hasattr(mock_client_instance_internal._system, 'delete_pending_collections'):
        mock_system_obj.delete_pending_collections.assert_called_once()
        mock_client_instance_internal.clear_pending_collections_for_deletion.assert_not_called()
        mock_vdb_logger.info.assert_any_call("ChromaDB client: Deleted pending collections (_system.delete_pending_collections <0.4.23 method).")
        
    assert vdbm_module._client is None # Should be reset
    mock_vdb_logger.info.assert_any_call("ChromaDB client resources released and client reset.")


@patch.object(vdbm_module, 'get_client')
def test_close_client_handles_exception_during_cleanup(
    mock_get_client_for_close_exc, mock_persistent_client_class,
    mock_vdb_logger, mock_appconfig_vdb_defaults, monkeypatch
):
    MockClientClass, mock_client_instance_internal_exc = mock_persistent_client_class
    with patch('os.path.exists', return_value=True), patch('os.makedirs'):
        vdbm_module.get_client() # Sets the global _client
    
    assert vdbm_module._client is mock_client_instance_internal_exc

    # Simulate reset() raising an error
    reset_exception = Exception("Error during client.reset()")
    mock_client_instance_internal_exc.reset = MagicMock(side_effect=reset_exception)
    # Ensure other cleanup methods don't also error for this specific test
    mock_client_instance_internal_exc.clear_pending_collections_for_deletion = MagicMock()

    vdbm_module.close_client()

    mock_client_instance_internal_exc.reset.assert_called_once()
    # clear_pending_collections_for_deletion should still be called if reset() fails, due to try/except blocks
    mock_client_instance_internal_exc.clear_pending_collections_for_deletion.assert_called_once()
    
    mock_vdb_logger.error.assert_any_call(
        f"Error during ChromaDB client cleanup (reset/clear_pending): {reset_exception}", exc_info=True
    )
    assert vdbm_module._client is None # Still reset even if cleanup had errors
