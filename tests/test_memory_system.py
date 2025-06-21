import unittest
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from datetime import datetime

class TestAgenticMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",
            llm_model="gpt-4o-mini"
        )
        
    def test_create_memory(self):
        """Test creating a new memory with complete metadata."""
        content = "Test memory content"
        tags = ["test", "memory"]
        keywords = ["test", "content"]
        links = ["link1", "link2"]
        context = "Test context"
        category = "Test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp
        )
        
        # Verify memory was created
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        self.assertEqual(memory.tags, tags)
        self.assertEqual(memory.keywords, keywords)
        self.assertEqual(memory.links, links)
        self.assertEqual(memory.context, context)
        self.assertEqual(memory.category, category)
        self.assertEqual(memory.timestamp, timestamp)
        
    def test_memory_metadata_persistence(self):
        """Test that memory metadata persists through ChromaDB storage and retrieval."""
        # Create a memory with complex metadata
        content = "Complex test memory"
        tags = ["test", "complex", "metadata"]
        keywords = ["test", "complex", "keywords"]
        links = ["link1", "link2", "link3"]
        context = "Complex test context"
        category = "Complex test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        evolution_history = ["evolution1", "evolution2"]
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp,
            evolution_history=evolution_history
        )
        
        # Search for the memory using ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertGreater(len(results), 0)
        
        # Verify metadata in search results
        result = results[0]
        self.assertEqual(result['content'], content)
        self.assertEqual(result['tags'], tags)
        self.assertEqual(result['keywords'], keywords)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['category'], category)
        
    def test_memory_update(self):
        """Test updating memory metadata through ChromaDB."""
        # Create initial memory
        content = "Initial content"
        memory_id = self.memory_system.add_note(content=content)
        
        # Update memory with new metadata
        new_content = "Updated content"
        new_tags = ["updated", "tags"]
        new_keywords = ["updated", "keywords"]
        new_context = "Updated context"
        
        success = self.memory_system.update(
            memory_id,
            content=new_content,
            tags=new_tags,
            keywords=new_keywords,
            context=new_context
        )
        
        self.assertTrue(success)
        
        # Verify updates in ChromaDB
        results = self.memory_system.search_agentic(new_content, k=1)
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result['content'], new_content)
        self.assertEqual(result['tags'], new_tags)
        self.assertEqual(result['keywords'], new_keywords)
        self.assertEqual(result['context'], new_context)
        
    def test_memory_relationships(self):
        """Test memory relationships and linked memories."""
        # Create related memories
        content1 = "First memory"
        content2 = "Second memory"
        content3 = "Third memory"
        
        id1 = self.memory_system.add_note(content1)
        id2 = self.memory_system.add_note(content2)
        id3 = self.memory_system.add_note(content3)
        
        # Add relationships
        memory1 = self.memory_system.read(id1)
        memory2 = self.memory_system.read(id2)
        memory3 = self.memory_system.read(id3)
        
        memory1.links.append(id2)
        memory2.links.append(id1)
        memory2.links.append(id3)
        memory3.links.append(id2)
        
        # Update memories with relationships
        self.memory_system.update(id1, links=memory1.links)
        self.memory_system.update(id2, links=memory2.links)
        self.memory_system.update(id3, links=memory3.links)
        
        # Test relationship retrieval
        results = self.memory_system.search_agentic(content1, k=3)
        self.assertGreater(len(results), 0)
        
        # Verify relationships are maintained
        memory1_updated = self.memory_system.read(id1)
        self.assertIn(id2, memory1_updated.links)
        
    def test_memory_evolution(self):
        """Test memory evolution system with ChromaDB."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            self.assertIsNotNone(memory.keywords)
            
        # Test evolution through search
        results = self.memory_system.search_agentic("neural networks", k=3)
        self.assertGreater(len(results), 0)
        
        # Verify evolution metadata
        for result in results:
            self.assertIsNotNone(result['tags'])
            self.assertIsNotNone(result['context'])
            self.assertIsNotNone(result['keywords'])
            
    def test_memory_deletion(self):
        """Test memory deletion from ChromaDB."""
        # Create and delete a memory
        content = "Memory to delete"
        memory_id = self.memory_system.add_note(content)
        
        # Verify memory exists
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        
        # Delete memory
        success = self.memory_system.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_system.read(memory_id)
        self.assertIsNone(memory)
        
        # Verify memory is removed from ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertEqual(len(results), 0)
        
    def test_memory_consolidation(self):
        """Test memory consolidation with ChromaDB."""
        # Create multiple memories
        contents = [
            "Memory 1",
            "Memory 2",
            "Memory 3"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Force consolidation
        self.memory_system.consolidate_memories()
        
        # Verify memories are still accessible
        for content in contents:
            results = self.memory_system.search_agentic(content, k=1)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]['content'], content)
            
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python",
            "Web development with JavaScript"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories
        results = self.memory_system.find_related_memories("Python", k=2)
        self.assertGreater(len(results), 0)
        
    def test_find_related_memories_raw(self):
        """Test finding related memories with raw format."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories in raw format
        results = self.memory_system.find_related_memories_raw("Python", k=2)
        self.assertIsNotNone(results)
        
    def test_process_memory(self):
        """Test memory processing and evolution."""
        # Create a test memory
        content = "Test memory for processing"
        memory_id = self.memory_system.add_note(content)
        
        # Get the memory
        memory = self.memory_system.read(memory_id)
        
        # Process the memory
        should_evolve, processed_memory = self.memory_system.process_memory(memory)
        
        # Verify processing results
        self.assertIsInstance(should_evolve, bool)
        self.assertIsInstance(processed_memory, MemoryNote)
        self.assertIsNotNone(processed_memory.tags)
        self.assertIsNotNone(processed_memory.context)
        self.assertIsNotNone(processed_memory.keywords)

if __name__ == '__main__':
    unittest.main()
