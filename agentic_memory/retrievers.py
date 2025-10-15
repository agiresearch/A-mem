import json
from typing import Dict, List
import ast

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nltk.tokenize import word_tokenize


def simple_tokenize(text):
    return word_tokenize(text)


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""

    def __init__(
        self, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.add(
            documents=[document], metadatas=[processed_metadata], ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        if (results is not None) and (results.get("metadatas", [])):
            results["metadatas"] = self._convert_metadata_types(
                results["metadatas"])
        
        return results

    def _convert_metadata_types(
        self, 
        metadatas: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Convert string metadata back to original types.
        
        Args:
            metadatas: List of metadata lists from query results
            
        Returns:
            Converted metadata structure
        """
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, List):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, Dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict) -> None:
        """Convert metadata values from strings to appropriate types in-place.
        
        Args:
            metadata: Single metadata dictionary to convert
        """
        for key, value in metadata.items():
            # only attempt to convert strings
            if not isinstance(value, str):
                continue
            else:
                try:
                    metadata[key] = ast.literal_eval(value)
                except Exception:
                    pass
