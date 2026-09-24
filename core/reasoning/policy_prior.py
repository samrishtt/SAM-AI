"""System 1: Intuitive Policy Prior Generator.

Provides fast, low-latency candidate action/thought proposals P(a | s)
conditioned on working memory, semantic knowledge rules, and episodic associations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.memory.working_memory import WorkingMemory
from core.memory.semantic_graph import SemanticGraph
from core.memory.episodic_store import EpisodicStore


@dataclass
class ActionCandidate:
    action: str
    prior_prob: float
    rationale: str
    metadata: Dict = None


class PolicyPrior:
    """System 1 intuitive generator proposing candidate reasoning steps and priors."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def propose_candidates(
        self,
        goal: str,
        working_memory: WorkingMemory,
        semantic_graph: Optional[SemanticGraph] = None,
        episodic_store: Optional[EpisodicStore] = None,
        allowed_actions: Optional[List[str]] = None,
        max_candidates: int = 5,
    ) -> List[ActionCandidate]:
        """Generates candidate reasoning steps with normalized prior probabilities P(a | s)."""
        candidates: List[ActionCandidate] = []
        raw_scores: List[float] = []

        # 1. Check if allowed explicit actions provided
        if allowed_actions:
            for act in allowed_actions:
                # Base heuristic scoring
                score = 1.0
                rationale = "Available domain action"
                
                # Boost if action matches active sub-goal keywords
                for word in goal.lower().split():
                    if len(word) > 3 and word in act.lower():
                        score += 1.5
                        rationale += f" (matches goal keyword '{word}')"

                # Penalize already executed actions in working memory
                for obs in working_memory.get_observations():
                    if act in obs or act.split("(")[0] in obs:
                        score -= 1.8
                        rationale += " (already executed)"

                # Boost if episodic memory recalled success with similar action
                if episodic_store and len(episodic_store) > 0:
                    similar_eps = episodic_store.search(goal, top_k=2, min_reward=0.5)
                    for ep, sim in similar_eps:
                        if any(act.split("(")[0] in a for a in ep.action_sequence):
                            score += 2.0 * sim
                            rationale += f" (empirically validated in past episode {ep.id})"

                # Boost if semantic graph contains rule
                if semantic_graph:
                    verb = act.split("(")[0].strip()
                    facts = semantic_graph.query(subject=verb, relation="solves_subproblem")
                    if facts:
                        score += 1.8
                        rationale += " (reinforced by semantic schema)"

                candidates.append(ActionCandidate(action=act, prior_prob=0.0, rationale=rationale))
                raw_scores.append(score)
        else:
            # Default cognitive meta-actions: decompose, hypothesize, verify, synthesize
            default_pool = [
                ("decompose_problem(goal)", 2.0, "Hierarchical task breakdown"),
                ("formulate_hypothesis(domain='latent')", 1.8, "Generate explanatory candidate"),
                ("test_hypothesis_in_sandbox()", 1.5, "Empirical sandbox verification"),
                ("retrieve_analogous_patterns()", 1.2, "Query episodic/semantic store"),
                ("synthesize_final_solution()", 1.0, "Consolidate verified steps"),
            ]
            for act, score, rat in default_pool:
                candidates.append(ActionCandidate(action=act, prior_prob=0.0, rationale=rat))
                raw_scores.append(score)

        # Apply softmax over scores with temperature to generate probability distribution P(a | s)
        scores_arr = np.array(raw_scores, dtype=np.float32) / max(0.1, self.temperature)
        exp_scores = np.exp(scores_arr - np.max(scores_arr))
        probs = exp_scores / np.sum(exp_scores)

        for i, cand in enumerate(candidates):
            cand.prior_prob = float(probs[i])

        # Sort descending by prior probability
        candidates.sort(key=lambda c: c.prior_prob, reverse=True)
        return candidates[:max_candidates]
