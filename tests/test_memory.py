"""Unit tests for the memory subsystems."""

import pytest
import numpy as np
from core.memory.working_memory import WorkingMemory, GoalStatus
from core.memory.episodic_store import EpisodicStore, SemanticEmbedder
from core.memory.semantic_graph import SemanticGraph, Triple
from core.memory.consolidator import MemoryConsolidator


def test_working_memory_hierarchy():
    wm = WorkingMemory(max_observations=5)
    g1 = wm.push_goal("Master Objective")
    assert wm.get_current_goal().id == g1.id
    assert wm.get_current_goal().status == GoalStatus.ACTIVE

    g2 = wm.push_goal("Subtask 1", parent_id=g1.id)
    assert wm.get_current_goal().id == g2.id

    wm.complete_current_goal(success=True)
    assert g2.status == GoalStatus.COMPLETED
    assert wm.get_current_goal().id == g1.id


def test_working_memory_scratchpad_and_buffer():
    wm = WorkingMemory(max_observations=3)
    wm.set_fact("status", "running")
    assert wm.get_fact("status") == "running"

    for i in range(5):
        wm.add_observation(f"Obs {i}")
    assert len(wm.get_observations()) == 3
    assert wm.get_observations() == ["Obs 2", "Obs 3", "Obs 4"]


def test_semantic_embedder():
    embedder = SemanticEmbedder(dim=64)
    v1 = embedder.embed("Autonomous agent planning with tree search")
    v2 = embedder.embed("MCTS planning for intelligent agents")
    v3 = embedder.embed("Photosynthesis in cellular chloroplasts")

    sim_related = float(np.dot(v1, v2))
    sim_unrelated = float(np.dot(v1, v3))

    assert np.isclose(np.linalg.norm(v1), 1.0)
    assert sim_related > sim_unrelated


def test_episodic_store_retrieval():
    store = EpisodicStore(embedding_dim=64)
    store.record_episode(
        goal="Solve ARC grid rotation",
        context="Input grid 2x2",
        action_sequence=["apply_rot90()", "verify()"],
        outcome="Success",
        reward=1.0,
    )
    store.record_episode(
        goal="Query enzyme kinetics",
        context="In vitro assay",
        action_sequence=["query_db()"],
        outcome="Found 4 records",
        reward=0.8,
    )

    results = store.search("Rotate spatial matrix 90 degrees", top_k=1)
    assert len(results) == 1
    assert "rotation" in results[0][0].goal.lower()


def test_semantic_graph_forward_chain():
    graph = SemanticGraph()
    graph.add_fact("alpha", "causes", "beta")
    graph.add_fact("beta", "causes", "gamma")

    # Transitivity rule: IF (?a causes ?b) AND (?b causes ?c) THEN (?a directly_or_indirectly_causes ?c)
    graph.add_rule(
        name="transitive_causality",
        antecedents=[("?a", "causes", "?b"), ("?b", "causes", "?c")],
        consequent=("?a", "indirectly_causes", "?c"),
    )

    inferred = graph.forward_chain()
    assert len(inferred) >= 1
    inferred_facts = graph.query(subject="alpha", relation="indirectly_causes", object_="gamma")
    assert len(inferred_facts) == 1


def test_memory_consolidator():
    episodic = EpisodicStore()
    semantic = SemanticGraph()
    consolidator = MemoryConsolidator(min_reward_threshold=0.5)

    episodic.record_episode(
        goal="Invert spatial matrix",
        context="Task 1",
        action_sequence=["apply_flip_horizontal()", "verify_solution()"],
        outcome="Solved",
        reward=1.0,
        reflection="Horizontal symmetry is conserved",
    )
    episodic.record_episode(
        goal="Invert spatial matrix",
        context="Task 2",
        action_sequence=["apply_flip_horizontal()", "verify_solution()"],
        outcome="Solved",
        reward=0.9,
    )

    report = consolidator.consolidate(episodic, semantic)
    assert report.episodes_examined == 2
    assert report.facts_distilled > 0
    assert len(semantic.get_all_facts()) > 0
