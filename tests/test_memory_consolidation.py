from unittest.mock import Mock, call, patch

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


def test_consolidation_replaces_existing_retriever_entries():
    note = MemoryNote(
        content="synthetic note",
        id="target",
        context="evolved context",
        tags=["evolved tag"],
    )
    system = AgenticMemorySystem.__new__(AgenticMemorySystem)
    system.memories = {note.id: note}
    system.model_name = "test-model"
    retriever = Mock()

    with patch(
        "agentic_memory.memory_system.ChromaRetriever", return_value=retriever
    ):
        system.consolidate_memories()

    assert retriever.mock_calls[0] == call.delete_document(note.id)
    assert retriever.mock_calls[1] == call.add_document(
        note.content,
        {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
        },
        note.id,
    )
