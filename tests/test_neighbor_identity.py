"""Neighbor updates must follow retrieved memory identities, not insertion order."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


@pytest.fixture
def memory_system():
    # Replace external services before construction; keep the real business methods.
    with patch("agentic_memory.memory_system.ChromaRetriever"):
        with patch("agentic_memory.memory_system.LLMController"):
            system = AgenticMemorySystem()

    system.memories = {
        key: MemoryNote(
            id=key,
            content=f"{key}-content",
            context=f"{key}-old-context",
            tags=[f"{key}-old-tag"],
            timestamp="202611070800",
            last_accessed="202611070800",
        )
        for key in ("A", "B", "C")
    }
    return system


@pytest.mark.parametrize(
    "retrieved_ids,new_contexts,new_tags,expected_changes",
    [
        pytest.param(
            ["C"], ["C-new-context"], [["C-new-tag"]],
            {"C": {"context": "C-new-context", "tags": ["C-new-tag"]}},
            id="single-neighbor-different-from-insertion-rank",
        ),
        pytest.param(
            ["C", "A"], ["C-new-context", "A-new-context"],
            [["C-new-tag"], ["A-new-tag"]],
            {
                "C": {"context": "C-new-context", "tags": ["C-new-tag"]},
                "A": {"context": "A-new-context", "tags": ["A-new-tag"]},
            },
            id="multiple-neighbors-in-retrieval-order",
        ),
        pytest.param(
            ["C", "A"], ["C-new-context"], [["C-new-tag"], ["A-new-tag"]],
            {
                "C": {"context": "C-new-context", "tags": ["C-new-tag"]},
                "A": {"context": "A-old-context", "tags": ["A-new-tag"]},
            },
            id="short-context-array-keeps-target-neighbors-own-context",
        ),
        pytest.param(
            ["A", "B"], ["A-new-context", "B-new-context"],
            [["A-new-tag"], ["B-new-tag"]],
            {
                "A": {"context": "A-new-context", "tags": ["A-new-tag"]},
                "B": {"context": "B-new-context", "tags": ["B-new-tag"]},
            },
            id="control-retrieval-and-insertion-orders-match",
        ),
        pytest.param(
            ["C", "missing", "A"], ["C-new-context", "missing-context", "A-new-context"],
            [["C-new-tag"], ["missing-tag"], ["A-new-tag"]],
            {
                "C": {"context": "C-new-context", "tags": ["C-new-tag"]},
                "A": {"context": "A-new-context", "tags": ["A-new-tag"]},
            },
            id="missing-neighbor-does-not-shift-or-stop-remaining-updates",
        ),
    ],
)
def test_neighbor_updates_preserve_identity(
    memory_system, retrieved_ids, new_contexts, new_tags, expected_changes
):
    system = memory_system
    expected = {key: deepcopy(vars(note)) for key, note in system.memories.items()}
    for key, changes in expected_changes.items():
        expected[key].update(changes)

    system.retriever.search.return_value = {
        "ids": [retrieved_ids],
        "metadatas": [[
            deepcopy(vars(system.read(key))) if system.read(key) is not None
            else {"content": "A neighbor no longer available locally", "context": "old"}
            for key in retrieved_ids
        ]],
    }
    system.llm_controller.llm.get_completion.return_value = json.dumps({
        "should_evolve": True,
        "actions": ["update_neighbor"],
        "suggested_connections": [],
        "tags_to_update": [],
        "new_context_neighborhood": new_contexts,
        "new_tags_neighborhood": new_tags,
    })

    incoming = MemoryNote(content="New information about the retrieved neighbors")
    evolved, processed = system.process_memory(incoming)

    assert evolved is True
    assert processed is incoming
    actual = {key: deepcopy(vars(system.read(key))) for key in ("A", "B", "C")}
    # Compare all note fields: unrelated notes and original content/IDs must survive.
    assert actual == expected


def test_evolution_prompt_exposes_neighbor_ids(memory_system):
    system = memory_system
    # Keep IDs independent of the content so incidental text cannot satisfy the check.
    retrieved_ids = ["retrieved-id-123", "retrieved-id-456"]
    notes = [system.read("C"), system.read("A")]
    system.memories = dict(zip(retrieved_ids, notes))
    for memory_id, note in system.memories.items():
        note.id = memory_id
    system.retriever.search.return_value = {
        "ids": [retrieved_ids],
        "metadatas": [[
            dict(deepcopy(vars(note)), id="stale-metadata-id") for note in notes
        ]],
    }
    system.llm_controller.llm.get_completion.return_value = json.dumps({
        "should_evolve": True,
        "actions": ["strengthen"],
        "suggested_connections": [retrieved_ids[0]],
        "tags_to_update": ["connected"],
        "new_context_neighborhood": [note.context for note in notes],
        "new_tags_neighborhood": [note.tags for note in notes],
    })

    incoming = MemoryNote(content="New information about these neighbors")
    evolved, processed = system.process_memory(incoming)

    prompt = system.llm_controller.llm.get_completion.call_args[0][0]
    neighbor_lines = [line for line in prompt.splitlines() if "memory content:" in line]
    assert len(neighbor_lines) == len(notes)
    for line, memory_id, note in zip(neighbor_lines, retrieved_ids, notes):
        assert memory_id in line
        assert note.content in line
    assert evolved is True
    assert processed.links == [retrieved_ids[0]]
    assert system.read(processed.links[0]) is notes[0]
