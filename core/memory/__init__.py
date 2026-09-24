"""Memory Subsystem of Micro-AGI.

Exposes:
- WorkingMemory: Active goal stack & attention-limited context.
- EpisodicStore: Fast-learning vector-indexed experiential memory.
- SemanticGraph: Slow-learning neocortical knowledge graph with forward-chaining rules.
- MemoryConsolidator: Offline CLS distillation from episodes to invariant schemas.
"""

from core.memory.working_memory import WorkingMemory, Goal, GoalStatus
from core.memory.episodic_store import EpisodicStore, Episode, SemanticEmbedder
from core.memory.semantic_graph import SemanticGraph, Triple, InferenceRule
from core.memory.consolidator import MemoryConsolidator, ConsolidationReport

__all__ = [
    "WorkingMemory",
    "Goal",
    "GoalStatus",
    "EpisodicStore",
    "Episode",
    "SemanticEmbedder",
    "SemanticGraph",
    "Triple",
    "InferenceRule",
    "MemoryConsolidator",
    "ConsolidationReport",
]
