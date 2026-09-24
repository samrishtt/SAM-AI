"""Symbolic Deduction Benchmark Suite.

Evaluates multi-step formal logical reasoning, transitive inference chains,
and counterfactual constraint satisfaction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DeductionProblem:
    id: str
    title: str
    premises: List[str]
    query: str
    ground_truth: str
    allowed_reasoning_steps: List[str]


def get_standard_deduction_problems() -> List[DeductionProblem]:
    """Returns curated multi-step deductive logic problems."""
    return [
        DeductionProblem(
            id="logic_001_transitive_causality",
            title="Multi-hop Causal Chain",
            premises=[
                "Enzyme_Alpha activates Protein_Beta",
                "Protein_Beta phosphorylates Kinase_Gamma",
                "Kinase_Gamma induces Transcription_Delta",
            ],
            query="Does Enzyme_Alpha causally upregulate Transcription_Delta?",
            ground_truth="conclude_positive_causation(Enzyme_Alpha, Transcription_Delta)",
            allowed_reasoning_steps=[
                "query_direct_relation(Enzyme_Alpha, Kinase_Gamma)",
                "deduce_transitive_link(Enzyme_Alpha, Protein_Beta, Kinase_Gamma)",
                "deduce_transitive_link(Protein_Beta, Kinase_Gamma, Transcription_Delta)",
                "conclude_positive_causation(Enzyme_Alpha, Transcription_Delta)",
                "conclude_negative_causation()",
            ],
        ),
        DeductionProblem(
            id="logic_002_counterfactual_inhibition",
            title="Negative Constraint Propagation",
            premises=[
                "Agent_A blocks Receptor_R",
                "Receptor_R is required for Signal_S",
                "Signal_S triggers Apoptosis",
            ],
            query="What is the net effect of Agent_A on Apoptosis?",
            ground_truth="conclude_inhibitory_effect(Agent_A, Apoptosis)",
            allowed_reasoning_steps=[
                "deduce_blocking_effect(Agent_A, Receptor_R)",
                "propagate_signal_disruption(Receptor_R, Signal_S)",
                "conclude_inhibitory_effect(Agent_A, Apoptosis)",
                "conclude_stimulatory_effect()",
            ],
        ),
        DeductionProblem(
            id="logic_003_set_inclusion",
            title="Categorical Syllogism",
            premises=[
                "All Elements of Class_X belong to Class_Y",
                "All Elements of Class_Y belong to Class_Z",
                "Entity_Omega belongs to Class_X",
            ],
            query="Does Entity_Omega belong to Class_Z?",
            ground_truth="conclude_membership(Entity_Omega, Class_Z)",
            allowed_reasoning_steps=[
                "deduce_membership(Entity_Omega, Class_Y)",
                "deduce_transitive_inclusion(Class_X, Class_Y, Class_Z)",
                "conclude_membership(Entity_Omega, Class_Z)",
                "conclude_non_membership()",
            ],
        ),
    ]


class DeductionEvaluator:
    """Evaluates multi-step logical deduction accuracy."""

    def evaluate(self, agent_solution: str, problem: DeductionProblem) -> bool:
        if not agent_solution:
            return False
        clean_sol = agent_solution.strip().lower()
        clean_gt = problem.ground_truth.strip().lower()
        return clean_gt in clean_sol or problem.ground_truth.split("(")[0].lower() in clean_sol
