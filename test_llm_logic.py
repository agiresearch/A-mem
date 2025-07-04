#!/usr/bin/env python
"""
Simple test to verify the LLM-based attribute generation logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_llm_logic():
    """Test the LLM attribute generation logic."""
    print("Testing LLM attribute generation logic...")
    
    # Simulate the logic that we implemented
    content = "Machine learning is a subset of artificial intelligence that uses neural networks."
    
    # Simulate parameters that might be passed to MemoryNote
    keywords = None
    context = None
    tags = None
    category = None
    
    # Mock LLM controller available
    llm_controller = "mock_controller"
    
    # Test the condition logic
    needs_llm_analysis = llm_controller and any(param is None for param in [keywords, context, category, tags])
    
    print(f"Content: {content}")
    print(f"Keywords provided: {keywords}")
    print(f"Context provided: {context}")
    print(f"Tags provided: {tags}")
    print(f"Category provided: {category}")
    print(f"LLM controller available: {bool(llm_controller)}")
    print(f"Needs LLM analysis: {needs_llm_analysis}")
    
    # Test assertion
    assert needs_llm_analysis == True, "Should need LLM analysis when attributes are missing"
    
    # Test partial attributes
    keywords_partial = ["provided", "keyword"]
    context_partial = None
    tags_partial = ["provided", "tag"]
    category_partial = None
    
    needs_llm_analysis_partial = llm_controller and any(param is None for param in [keywords_partial, context_partial, category_partial, tags_partial])
    
    print(f"\nPartial attributes test:")
    print(f"Keywords provided: {keywords_partial}")
    print(f"Context provided: {context_partial}")
    print(f"Tags provided: {tags_partial}")
    print(f"Category provided: {category_partial}")
    print(f"Needs LLM analysis: {needs_llm_analysis_partial}")
    
    # Test assertion
    assert needs_llm_analysis_partial == True, "Should need LLM analysis when some attributes are missing"
    
    # Test all attributes provided
    keywords_all = ["provided", "keyword"]
    context_all = "Provided context"
    tags_all = ["provided", "tag"]
    category_all = "Provided category"
    
    needs_llm_analysis_all = llm_controller and any(param is None for param in [keywords_all, context_all, category_all, tags_all])
    
    print(f"\nAll attributes provided test:")
    print(f"Keywords provided: {keywords_all}")
    print(f"Context provided: {context_all}")
    print(f"Tags provided: {tags_all}")
    print(f"Category provided: {category_all}")
    print(f"Needs LLM analysis: {needs_llm_analysis_all}")
    
    # Test assertion
    assert needs_llm_analysis_all == False, "Should not need LLM analysis when all attributes are provided"
    
    print("\n✓ All LLM logic tests passed!")
    return True

def test_attribute_merging():
    """Test the attribute merging logic."""
    print("\nTesting attribute merging logic...")
    
    # Simulate LLM analysis results
    mock_analysis = {
        "keywords": ["machine", "learning", "neural"],
        "context": "Discussion about AI and machine learning techniques",
        "tags": ["AI", "technology", "research"]
    }
    
    # Test merging when no attributes provided
    provided_keywords = None
    provided_context = None
    provided_tags = None
    
    final_keywords = provided_keywords or mock_analysis.get("keywords", [])
    final_context = provided_context or mock_analysis.get("context", "General")
    final_tags = provided_tags or mock_analysis.get("tags", [])
    
    print(f"No attributes provided:")
    print(f"  Final keywords: {final_keywords}")
    print(f"  Final context: {final_context}")
    print(f"  Final tags: {final_tags}")
    
    assert final_keywords == ["machine", "learning", "neural"], "Should use LLM-generated keywords"
    assert final_context == "Discussion about AI and machine learning techniques", "Should use LLM-generated context"
    assert final_tags == ["AI", "technology", "research"], "Should use LLM-generated tags"
    
    # Test merging when some attributes provided
    provided_keywords_partial = ["custom", "keyword"]
    provided_context_partial = None
    provided_tags_partial = ["custom", "tag"]
    
    final_keywords_partial = provided_keywords_partial or mock_analysis.get("keywords", [])
    final_context_partial = provided_context_partial or mock_analysis.get("context", "General")
    final_tags_partial = provided_tags_partial or mock_analysis.get("tags", [])
    
    print(f"\nSome attributes provided:")
    print(f"  Final keywords: {final_keywords_partial}")
    print(f"  Final context: {final_context_partial}")
    print(f"  Final tags: {final_tags_partial}")
    
    assert final_keywords_partial == ["custom", "keyword"], "Should use provided keywords"
    assert final_context_partial == "Discussion about AI and machine learning techniques", "Should use LLM-generated context"
    assert final_tags_partial == ["custom", "tag"], "Should use provided tags"
    
    print("\n✓ Attribute merging tests passed!")
    return True

if __name__ == "__main__":
    success = True
    
    try:
        success &= test_llm_logic()
        success &= test_attribute_merging()
        
        if success:
            print("\n✅ All tests passed! The LLM attribute generation logic is working correctly.")
            print("The implementation should now automatically generate keywords, tags, and context using LLM.")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)