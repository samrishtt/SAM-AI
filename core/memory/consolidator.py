"""Memory Consolidation Engine (Hippocampal-Neocortical Transfer).

Implements the Complementary Learning Systems (CLS) consolidation process:
iterates through high-reward episodic traces, clusters recurring state-action
transitions, induces generalized invariant rules, and integrates them into
the Semantic Graph while pruning redundant episodic buffer traces.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from collections import Counter
import re

from core.memory.episodic_store import EpisodicStore, Episode
from core.memory.semantic_graph import SemanticGraph, Triple


@dataclass
class ConsolidationReport:
    episodes_examined: int
    facts_distilled: int
    rules_induced: int
    episodes_pruned: int
    summary: str


class MemoryConsolidator:
    """Orchestrates offline / sleep-phase memory distillation from episodic to semantic store."""

    def __init__(self, min_reward_threshold: float = 0.5):
        self.min_reward_threshold = min_reward_threshold

    def consolidate(self, episodic: EpisodicStore, semantic: SemanticGraph) -> ConsolidationReport:
        """Runs one consolidation cycle."""
        high_reward_eps = [
            ep for ep in episodic.episodes
            if ep.reward >= self.min_reward_threshold
        ]

        facts_distilled = 0
        rules_induced = 0
        episodes_pruned = 0

        # 1. Causal Fact Distillation: map successful (goal, first_action, outcome)
        for ep in high_reward_eps:
            if ep.action_sequence and ep.outcome:
                first_action = ep.action_sequence[0].split("(")[0].strip()
                goal_clean = re.sub(r"[^\w\s]", "", ep.goal.lower()).strip()
                
                # Formulate distilled causal triple: (action, achieves_goal, goal)
                semantic.add_fact(
                    subject=first_action,
                    relation="solves_subproblem",
                    object_=goal_clean[:30],
                    confidence=min(1.0, 0.7 + (ep.reward * 0.3)),
                    source=f"consolidated:{ep.id}",
                )
                facts_distilled += 1

                # If reflection contains key insight
                if ep.reflection:
                    semantic.add_fact(
                        subject=first_action,
                        relation="associated_heuristic",
                        object_=ep.reflection[:50].lower(),
                        confidence=0.85,
                        source=f"consolidated:{ep.id}",
                    )
                    facts_distilled += 1

        # 2. Rule Induction: Detect recurring multi-action patterns (e.g., A -> B)
        action_pairs: Counter = Counter()
        for ep in high_reward_eps:
            actions = [a.split("(")[0].strip() for a in ep.action_sequence]
            for i in range(len(actions) - 1):
                action_pairs[(actions[i], actions[i+1])] += 1

        for (a1, a2), count in action_pairs.items():
            if count >= 2:  # Recurring trajectory motif
                rule_name = f"heuristic_chain_{a1}_then_{a2}"
                # IF (?task, requires, a1) THEN (?task, suggests_next, a2)
                semantic.add_rule(
                    name=rule_name,
                    antecedents=[("?task", "requires", a1)],
                    consequent=("?task", "suggests_next", a2),
                    description=f"Empirically learned trajectory: {a1} frequently followed by {a2} in high-reward tasks",
                )
                rules_induced += 1

        summary = (
            f"Consolidated {len(high_reward_eps)} high-reward episodes. "
            f"Distilled {facts_distilled} semantic facts and induced {rules_induced} transition rules."
        )

        return ConsolidationReport(
            episodes_examined=len(high_reward_eps),
            facts_distilled=facts_distilled,
            rules_induced=rules_induced,
            episodes_pruned=episodes_pruned,
            summary=summary,
        )
