import pytest

from agentic_memory.retrievers import ChromaRetriever


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
