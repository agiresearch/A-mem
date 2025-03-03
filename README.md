# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🧬 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Usage Examples 💡

Here's how to use the Agentic Memory system for basic operations:

```python
from memory_system import AgenticMemorySystem

# Initialize the memory system 🚀
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',  # Embedding model for semantic search
    llm_backend="openai",           # LLM backend (openai/mock/ollama)
    llm_model="gpt-4"              # LLM model name
)

# Create (Add) Memories ➕
# Simple creation
memory_id = memory_system.create("Deep learning neural networks")

# Creation with metadata
memory_id = memory_system.create(
    content="Machine learning project notes",
    tags=["ml", "project"],
    category="Research",
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Read (Retrieve) Memories 📖
# Get memory by ID
memory = memory_system.read(memory_id)
print(f"Content: {memory.content}")
print(f"Tags: {memory.tags}")
print(f"Context: {memory.context}")
print(f"Keywords: {memory.keywords}")

# Search memories
results = memory_system.search("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id, "Updated content about deep learning")

# Delete Memories ❌
memory_system.delete(memory_id)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Finding semantic relationships
# 2. Updating metadata and context
# 3. Creating connections between related memories
# This happens automatically when creating or updating memories!
```

### Advanced Features 🌟

1. **Hybrid Search** 🔍
   - Combines ChromaDB vector search and embedding-based retrieval
   - Automatically deduplicates and ranks results
   - Returns most relevant memories first

2. **Memory Evolution** 🧬
   - Automatically analyzes content relationships
   - Updates tags and context based on related memories
   - Creates semantic connections between memories

3. **Flexible Metadata** 📋
   - Custom tags and categories
   - Automatic keyword extraction
   - Context generation
   - Timestamp tracking

4. **Multiple LLM Backends** 🤖
   - OpenAI (GPT-4, GPT-3.5)
   - Mock backend (for testing)
   - Ollama (for local deployment)

### Best Practices 💪

1. **Memory Creation** ✨:
   - Provide clear, specific content
   - Add relevant tags for better organization
   - Let the system handle context and keyword generation

2. **Memory Retrieval** 🔍:
   - Use specific search queries
   - Adjust 'k' parameter based on needed results
   - Consider both exact and semantic matches

3. **Memory Evolution** 🧬:
   - Allow automatic evolution to organize memories
   - Review generated connections periodically
   - Use consistent tagging conventions

4. **Error Handling** ⚠️:
   - Always check return values
   - Handle potential KeyError for non-existent memories
   - Use try-except blocks for LLM operations

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@misc{xu2025amemagenticmemoryllm,
      title={A-MEM: Agentic Memory for LLM Agents}, 
      author={Wujiang Xu and Zujie Liang and Kai Mei and Hang Gao and Juntao Tan and Yongfeng Zhang},
      year={2025},
      eprint={2502.12110},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12110}, 
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
