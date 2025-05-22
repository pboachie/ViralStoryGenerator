import pytest
from unittest.mock import patch, MagicMock

# Assuming the module is viralStoryGenerator.utils.embedding_generator
from viralStoryGenerator.utils import embedding_generator as eg_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_embedding_defaults(monkeypatch):
    """Set default app_config values for embedding_generator tests."""
    monkeypatch.setattr(app_config.rag, 'ENABLED', True) # Default to RAG enabled
    monkeypatch.setattr(app_config.rag, 'EMBEDDING_MODEL', "mock_embedding_model_name")
    monkeypatch.setattr(app_config.rag, 'DEVICE', "mock_device") # e.g., "cpu", "cuda"

@pytest.fixture
def mock_embedding_logger():
    """Fixture to mock the _logger in embedding_generator.py."""
    with patch('viralStoryGenerator.utils.embedding_generator._logger') as mock_logger:
        yield mock_logger

@pytest.fixture(autouse=True)
def reset_embedding_function_singleton(monkeypatch):
    """
    Reset the global _embedding_function in embedding_generator.py before each test
    to ensure memoization is tested correctly.
    """
    monkeypatch.setattr(eg_module, '_embedding_function', None)
    # Also reset SENTENCE_TRANSFORMERS_AVAILABLE if it's dynamically set and affects logic
    # Assuming it's set at import time based on try-except. For tests, we might need to control it.
    # For now, focusing on _embedding_function. If SENTENCE_TRANSFORMERS_AVAILABLE is critical,
    # we might need to reload the module or patch it directly.
    # The tests will primarily mock SentenceTransformerEmbeddingFunction itself.

# --- Tests for get_embedding_function ---

# Scenario 1: Successful initialization
@patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
def test_get_embedding_function_successful_initialization(
    MockSentenceTransformerEF, mock_embedding_logger, mock_appconfig_embedding_defaults
):
    # Ensure SENTENCE_TRANSFORMERS_AVAILABLE is True for this test path
    # This might require monkeypatching if it's determined dynamically and could be False
    with patch.object(eg_module, 'SENTENCE_TRANSFORMERS_AVAILABLE', True):
        mock_ef_instance = MagicMock() # Mock instance of the embedding function
        MockSentenceTransformerEF.return_value = mock_ef_instance

        ef = eg_module.get_embedding_function()

        assert ef is mock_ef_instance
        MockSentenceTransformerEF.assert_called_once_with(
            model_name=app_config.rag.EMBEDDING_MODEL,
            device=app_config.rag.DEVICE
        )
        mock_embedding_logger.info.assert_called_once_with(
            f"Initialized SentenceTransformerEmbeddingFunction with model: {app_config.rag.EMBEDDING_MODEL} on device: {app_config.rag.DEVICE}"
        )

# Scenario 2: Memoization (singleton behavior)
@patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
def test_get_embedding_function_memoization(
    MockSentenceTransformerEF, mock_embedding_logger, mock_appconfig_embedding_defaults
):
    with patch.object(eg_module, 'SENTENCE_TRANSFORMERS_AVAILABLE', True):
        mock_ef_instance = MagicMock()
        MockSentenceTransformerEF.return_value = mock_ef_instance

        # First call - should initialize
        ef1 = eg_module.get_embedding_function()
        assert ef1 is mock_ef_instance
        MockSentenceTransformerEF.assert_called_once()
        mock_embedding_logger.info.assert_called_once_with(
            f"Initialized SentenceTransformerEmbeddingFunction with model: {app_config.rag.EMBEDDING_MODEL} on device: {app_config.rag.DEVICE}"
        )
        
        # Reset logger mock for the second call to check it's not called again for initialization
        mock_embedding_logger.reset_mock()

        # Second call - should return existing instance
        ef2 = eg_module.get_embedding_function()
        assert ef2 is mock_ef_instance
        MockSentenceTransformerEF.assert_called_once() # Still only called once in total
        mock_embedding_logger.info.assert_not_called() # No new initialization log

# Scenario 3: Error during initialization
@patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
def test_get_embedding_function_initialization_error(
    MockSentenceTransformerEF, mock_embedding_logger, mock_appconfig_embedding_defaults
):
    with patch.object(eg_module, 'SENTENCE_TRANSFORMERS_AVAILABLE', True):
        init_exception = Exception("Model load failed")
        MockSentenceTransformerEF.side_effect = init_exception

        # First call - should fail
        ef1 = eg_module.get_embedding_function()
        assert ef1 is None
        MockSentenceTransformerEF.assert_called_once() # Attempted initialization
        mock_embedding_logger.error.assert_called_once_with(
            f"Failed to initialize SentenceTransformerEmbeddingFunction: {init_exception}", exc_info=True
        )
        
        # Reset mocks for second call check
        MockSentenceTransformerEF.reset_mock() # Reset call count and side_effect if needed
        MockSentenceTransformerEF.side_effect = init_exception # Re-apply side effect
        mock_embedding_logger.reset_mock()

        # Second call - should also return None and not try to re-initialize if it failed before
        # (based on current implementation checking if _embedding_function is None *before* trying)
        # The current code does: `if _embedding_function is None and RAG_ENABLED and SENTENCE_TRANSFORMERS_AVAILABLE:`
        # So if _embedding_function remains None after a failure, it *will* try again.
        # This test will verify that behavior.
        ef2 = eg_module.get_embedding_function()
        assert ef2 is None
        MockSentenceTransformerEF.assert_called_once() # Attempted initialization again
        mock_embedding_logger.error.assert_called_once_with(
            f"Failed to initialize SentenceTransformerEmbeddingFunction: {init_exception}", exc_info=True
        )

# Scenario 4: RAG disabled
@patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
def test_get_embedding_function_rag_disabled(
    MockSentenceTransformerEF, mock_embedding_logger, mock_appconfig_embedding_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', False) # Disable RAG

    with patch.object(eg_module, 'SENTENCE_TRANSFORMERS_AVAILABLE', True): # Assume lib is available
        ef = eg_module.get_embedding_function()

        assert ef is None
        MockSentenceTransformerEF.assert_not_called()
        # Check for a specific log if RAG is disabled.
        # The current code in get_embedding_function doesn't log when RAG is disabled.
        # It just doesn't proceed. So, no specific log to check here for that condition.
        # If the calling code (e.g., vector_db_manager) logs, that's tested elsewhere.
        mock_embedding_logger.info.assert_not_called()
        mock_embedding_logger.warning.assert_not_called()


# Scenario 5: SentenceTransformers library not available
@patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') # Will not be called
def test_get_embedding_function_library_not_available(
    MockSentenceTransformerEF, mock_embedding_logger, mock_appconfig_embedding_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.rag, 'ENABLED', True) # RAG is enabled

    # Simulate SENTENCE_TRANSFORMERS_AVAILABLE being False
    # This is set at module import time in embedding_generator.py based on try-except.
    # To test this, we must patch it *on the module object*.
    with patch.object(eg_module, 'SENTENCE_TRANSFORMERS_AVAILABLE', False):
        ef = eg_module.get_embedding_function()

        assert ef is None
        MockSentenceTransformerEF.assert_not_called()
        mock_embedding_logger.critical.assert_called_once_with(
            "SentenceTransformers library is not installed, but RAG is enabled. "
            "Embeddings cannot be generated. Please install with `pip install sentence-transformers`."
        )

# Placeholder for other tests if the module evolves
def test_placeholder():
    pass
