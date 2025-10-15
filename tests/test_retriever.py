import pytest


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
