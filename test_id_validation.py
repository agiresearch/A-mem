#!/usr/bin/env python
"""
Test script to verify ID validation fix in process_memory method.
This simulates the issue described in Issue #11.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Mock the dependencies that are not available
class MockLLMController:
    class MockLLM:
        def get_completion(self, prompt, response_format=None):
            # Simulate LLM response with invalid IDs
            return '''{
                "should_evolve": true,
                "actions": ["strengthen"],
                "suggested_connections": ["0", "invalid_id", "memory1"],
                "tags_to_update": ["test", "validation"],
                "new_context_neighborhood": [],
                "new_tags_neighborhood": []
            }'''
    
    def __init__(self, backend, model, api_key):
        self.llm = self.MockLLM()

class MockChromaRetriever:
    def __init__(self, collection_name, model_name):
        self.collection_name = collection_name
        self.model_name = model_name
        self.documents = {}
        
    def add_document(self, content, metadata, doc_id):
        self.documents[doc_id] = {"content": content, "metadata": metadata}
        
    def search(self, query, k):
        # Return empty results for testing
        return {
            'ids': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
    def delete_document(self, doc_id):
        if doc_id in self.documents:
            del self.documents[doc_id]
            
    @property
    def client(self):
        class MockClient:
            def reset(self):
                pass
        return MockClient()

# Patch the imports
import agentic_memory.llm_controller
import agentic_memory.retrievers
agentic_memory.llm_controller.LLMController = MockLLMController
agentic_memory.retrievers.ChromaRetriever = MockChromaRetriever

# Now import the actual classes
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote

def test_id_validation():
    """Test that invalid IDs are filtered out during memory processing."""
    print("Testing ID validation in process_memory...")
    
    # Create memory system
    memory_system = AgenticMemorySystem()
    
    # Add some valid memories first
    memory1_id = memory_system.add_note("Memory 1")
    memory2_id = memory_system.add_note("Memory 2")
    print(f"Created memories: {memory1_id}, {memory2_id}")
    
    # Create a new memory that will trigger evolution
    new_note = MemoryNote("New memory that should trigger evolution")
    
    # Override find_related_memories to return test data
    original_find_related = memory_system.find_related_memories
    def mock_find_related(query, k=5):
        return "memory index:0\ttalk start time:20240101\tmemory content: Test\n", [0]
    memory_system.find_related_memories = mock_find_related
    
    # Process the memory (this will trigger the LLM mock which returns invalid IDs)
    try:
        should_evolve, processed_note = memory_system.process_memory(new_note)
        
        # Check results
        print(f"Should evolve: {should_evolve}")
        print(f"Note links after processing: {processed_note.links}")
        
        # Verify that only valid IDs were added
        for link_id in processed_note.links:
            if link_id not in memory_system.memories:
                print(f"ERROR: Invalid ID '{link_id}' was added to links!")
                return False
                
        # Check that invalid IDs were filtered out
        if '0' in processed_note.links or 'invalid_id' in processed_note.links:
            print("ERROR: Invalid IDs were not filtered out!")
            return False
            
        print("SUCCESS: Invalid IDs were properly filtered out!")
        print(f"Valid links added: {processed_note.links}")
        return True
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        return False
    finally:
        # Restore original method
        memory_system.find_related_memories = original_find_related

if __name__ == "__main__":
    success = test_id_validation()
    sys.exit(0 if success else 1)