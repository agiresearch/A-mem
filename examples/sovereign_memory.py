import os
import sys

# Ensure we can import from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.memory_system import AgenticMemorySystem

def main():
    print("🧠 Initializing A-mem Sovereign System (Local)...")
    
    # Initialize with local backend
    # Note: Requires Ollama running with 'llama3' pulled
    try:
        memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Local embeddings (via sentence-transformers)
            llm_backend="ollama",
            llm_model="llama3" 
        )
        print("✅ System initialized.")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # Add a memory
    print("\n📝 Adding Sovereign Memory...")
    content = "The user values data sovereignty and local processing above all else."
    try:
        # Note: A-mem automatically generates tags/context via LLM here
        memory_id = memory_system.add_note(
            content=content,
            tags=["sovereign", "privacy"],
            category="Principles"
        )
        print(f"   Memory stored with ID: {memory_id}")
    except Exception as e:
        print(f"❌ Failed to store memory: {e}")
        return

    # Retrieve
    print("\n🔍 Retrieving Memory...")
    try:
        results = memory_system.search_agentic("sovereignty", k=1)
        for res in results:
            print(f"   Found: {res['content']}")
            print(f"   Tags: {res['tags']}")
            print(f"   Context (LLM Generated): {res.get('context', 'N/A')}")
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")

if __name__ == "__main__":
    main()
