"""Test utilities for the memory system."""
from typing import List
from agentic_memory.llm_controller import BaseLLMController

class MockLLMController(BaseLLMController):
    """Mock LLM controller for testing"""
    def __init__(self):
        self.mock_response = "{}"
        
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Mock completion that returns the pre-set response"""
        return self.mock_response
        
    def get_embedding(self, text: str) -> List[float]:
        """Mock embedding that returns a zero vector"""
        return [0.0] * 384  # Mock embedding vector
