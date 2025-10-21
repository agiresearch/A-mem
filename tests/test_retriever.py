import pytest

from agentic_memory.retrievers import PersistentChromaRetriever


def test_initialization(retriever):
    """Test ChromaRetriever initializes correctly."""
    assert retriever.collection is not None
    assert retriever.embedding_function is not None


def test_add_document(retriever, sample_metadata):
    """Test adding a document with metadata."""
    doc_id = "test_doc_1"
    document = "This is a test document."
    
    retriever.add_document(document, sample_metadata, doc_id)
    
    results = retriever.collection.get(ids=[doc_id])
    assert len(results["ids"]) == 1
    assert results["ids"][0] == doc_id


def test_delete_document(retriever, sample_metadata):
    """Test deleting a document."""
    doc_id = "test_doc_2"
    retriever.add_document("Test document", sample_metadata, doc_id)
    
    retriever.delete_document(doc_id)
    
    results = retriever.collection.get(ids=[doc_id])
    assert len(results["ids"]) == 0


def test_search(retriever, sample_metadata):
    """Test searching for similar documents."""
    retriever.add_document(
        "Machine learning is fascinating", sample_metadata, "doc1")
    retriever.add_document(
        "Deep learning uses neural networks", sample_metadata, "doc2")
    retriever.add_document(
        "Cats are fluffy animals", sample_metadata, "doc3")
    
    results = retriever.search("artificial intelligence", k=2)
    
    assert len(results["ids"][0]) == 2
    assert len(results["documents"][0]) == 2


def test_metadata_list_conversion(retriever):
    """Test that list metadata is properly converted."""
    metadata = {"tags": ["tag1", "tag2", "tag3"]}
    retriever.add_document("Test doc", metadata, "doc_list")
    
    results = retriever.search("Test", k=1)
    
    retrieved_tags = results["metadatas"][0][0]["tags"]
    assert isinstance(retrieved_tags, list)
    assert retrieved_tags == ["tag1", "tag2", "tag3"]


def test_metadata_dict_conversion(retriever):
    """Test that dict metadata is properly converted."""
    metadata = {"config": {"nested": "value", "number": 123}}
    retriever.add_document("Test doc", metadata, "doc_dict")
    
    results = retriever.search("Test", k=1)
    
    retrieved_config = results["metadatas"][0][0]["config"]
    assert isinstance(retrieved_config, dict)
    assert retrieved_config["nested"] == "value"


@pytest.mark.parametrize("value,expected_type", [
    ("42", int),
    ("3.14", float),
    ("-10", int),
    ("hello", str),
])
def test_numeric_string_conversion(retriever, value, expected_type):
    """Test numeric string conversion in metadata."""
    metadata = {"value": value}
    retriever.add_document("Test doc", metadata, f"doc_{value}")
    
    results = retriever.search("Test", k=1)
    
    retrieved_value = results["metadatas"][0][0]["value"]
    assert isinstance(retrieved_value, expected_type)


def test_search_returns_top_k_results(retriever, sample_metadata):
    """Test that search respects the k parameter."""
    for i in range(10):
        retriever.add_document(
            f"Document number {i}", sample_metadata, f"doc_{i}")
    
    results = retriever.search("Document", k=3)
    
    assert len(results["ids"][0]) == 3


class TestPersistentChromaRetriever:
    """Test suite for PersistentChromaRetriever."""
    
    def test_creates_new_collection(self, temp_db_dir):
        """Test creating a new persistent collection."""
        retriever = PersistentChromaRetriever(
            directory=str(temp_db_dir),
            collection_name="new_collection"
        )
        
        assert retriever.collection is not None
        assert retriever.collection_name == "new_collection"
        assert temp_db_dir.exists()
    
    @pytest.mark.parametrize("collection_name,extend,should_raise", [
        ("existing_collection", False, True),   # Existing collection, no extend -> error
        ("existing_collection", True, False),   # Existing collection, extend -> success
        ("new_collection", False, False),       # New collection, no extend -> success
        ("new_collection", True, False),        # New collection, extend -> success
    ])
    def test_collection_access_control(
        self, existing_collection, collection_name, extend, should_raise
    ):
        """Test collection access with different combinations of name and extend flag."""
        temp_db_dir, existing_name = existing_collection
        
        if should_raise:
            with pytest.raises(ValueError, match="already exists"):
                PersistentChromaRetriever(
                    directory=str(temp_db_dir),
                    collection_name=collection_name,
                    extend=extend
                )
        else:
            retriever = PersistentChromaRetriever(
                directory=str(temp_db_dir),
                collection_name=collection_name,
                extend=extend
            )
            assert retriever.collection is not None
            
            # If accessing existing collection, verify data is accessible
            if collection_name == existing_name:
                results = retriever.collection.get(ids=["existing_doc"])
                assert len(results["ids"]) == 1
    
    def test_persistence_across_sessions(self, temp_db_dir, sample_metadata):
        """Test that data persists across different retriever instances."""
        collection_name = "persistent_collection"
        
        # Session 1: Create and add document
        retriever1 = PersistentChromaRetriever(
            directory=str(temp_db_dir),
            collection_name=collection_name
        )
        retriever1.add_document("Persistent data", sample_metadata, "persist_doc")
        del retriever1
        
        # Session 2: Reconnect and verify data exists
        retriever2 = PersistentChromaRetriever(
            directory=str(temp_db_dir),
            collection_name=collection_name,
            extend=True
        )
        
        results = retriever2.collection.get(ids=["persist_doc"])
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Persistent data"
    
    def test_uses_default_directory_when_none(self):
        """Test that default directory is used when none provided."""
        retriever = PersistentChromaRetriever(
            collection_name="default_dir_collection"
        )
        
        # Should use ~/.chromadb as default
        from pathlib import Path
        default_path = Path.home() / '.chromadb'
        assert default_path.exists()
        
        # Cleanup
        retriever.client.delete_collection("default_dir_collection")
