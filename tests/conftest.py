import pytest
import tempfile
import shutil
from pathlib import Path

from agentic_memory.retrievers import ChromaRetriever, PersistentChromaRetriever


@pytest.fixture
def retriever():
    """Fixture providing a clean ChromaRetriever instance."""
    retriever = ChromaRetriever(collection_name="test_memories")
    yield retriever
    # Cleanup: reset the collection after each test
    retriever.client.reset()


@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata with various types."""
    return {
        "timestamp": "2024-01-01T00:00:00",
        "tags": ["test", "memory"],
        "config": {"key": "value"},
        "count": 42,
        "score": 0.95
    }


@pytest.fixture
def temp_db_dir():
    """Fixture providing a temporary directory for persistent ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup: remove the temporary directory after test
    shutil.rmtree(temp_dir, ignore_errors=True)

    
@pytest.fixture
def existing_collection(temp_db_dir, sample_metadata):
    """Fixture that creates a pre-existing collection with data."""
    retriever = PersistentChromaRetriever(
        directory=str(temp_db_dir),
        collection_name="existing_collection"
    )
    retriever.add_document("Existing document", sample_metadata, "existing_doc")
    return temp_db_dir, "existing_collection"
