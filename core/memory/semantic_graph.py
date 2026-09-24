"""Semantic Memory Subsystem (Neocortical Knowledge Graph).

Maintains relational triples, invariant domain schemas, and a forward-chaining
neuro-symbolic inference engine for logical deduction and rule induction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object_: str
    confidence: float = 1.0
    source: str = "axiom"

    def __repr__(self) -> str:
        return f"({self.subject} --[{self.relation}]--> {self.object_} [conf: {self.confidence:.2f}])"


@dataclass
class InferenceRule:
    """Horn-clause style deductive rule: IF antecedents THEN consequent."""
    name: str
    antecedents: List[Tuple[str, str, str]]  # List of (subject_var, relation, object_var)
    consequent: Tuple[str, str, str]         # (subject_var, relation, object_var)
    description: str = ""


class SemanticGraph:
    """Relational knowledge graph with deductive inference capabilities."""

    def __init__(self):
        self._triples: Set[Triple] = set()
        self._rules: List[InferenceRule] = []

    def add_fact(self, subject: str, relation: str, object_: str, confidence: float = 1.0, source: str = "observed") -> Triple:
        """Adds a verified relational triple to the graph."""
        t = Triple(subject=subject.strip().lower(), relation=relation.strip().lower(), object_=object_.strip().lower(), confidence=confidence, source=source)
        # Remove any existing triple with lower confidence
        existing = [item for item in self._triples if item.subject == t.subject and item.relation == t.relation and item.object_ == t.object_]
        for ex in existing:
            self._triples.remove(ex)
        self._triples.add(t)
        return t

    def add_rule(self, name: str, antecedents: List[Tuple[str, str, str]], consequent: Tuple[str, str, str], description: str = "") -> InferenceRule:
        """Registers a symbolic deduction rule."""
        rule = InferenceRule(name=name, antecedents=antecedents, consequent=consequent, description=description)
        self._rules.append(rule)
        return rule

    def query(self, subject: Optional[str] = None, relation: Optional[str] = None, object_: Optional[str] = None) -> List[Triple]:
        """Queries the graph with pattern matching (None acts as wildcard)."""
        subj_match = subject.strip().lower() if subject else None
        rel_match = relation.strip().lower() if relation else None
        obj_match = object_.strip().lower() if object_ else None

        results = []
        for t in self._triples:
            if subj_match and t.subject != subj_match:
                continue
            if rel_match and t.relation != rel_match:
                continue
            if obj_match and t.object_ != obj_match:
                continue
            results.append(t)
        return results

    def forward_chain(self, max_iterations: int = 5) -> List[Triple]:
        """Executes forward-chaining inference applying all rules until fixed-point."""
        new_inferred: List[Triple] = []

        for _ in range(max_iterations):
            iteration_added = False

            for rule in self._rules:
                matches = self._match_antecedents(rule.antecedents)
                for binding in matches:
                    # Instantiate consequent
                    c_subj = self._substitute(rule.consequent[0], binding)
                    c_rel = self._substitute(rule.consequent[1], binding)
                    c_obj = self._substitute(rule.consequent[2], binding)

                    # Check if already present
                    existing = self.query(subject=c_subj, relation=c_rel, object_=c_obj)
                    if not existing:
                        inferred_triple = self.add_fact(c_subj, c_rel, c_obj, confidence=0.9, source=f"rule:{rule.name}")
                        new_inferred.append(inferred_triple)
                        iteration_added = True

            if not iteration_added:
                break

        return new_inferred

    def _match_antecedents(self, antecedents: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Finds variable bindings that satisfy all antecedent clauses."""
        if not antecedents:
            return [{}]

        results = [{}]
        for clause in antecedents:
            new_results = []
            c_subj, c_rel, c_obj = clause

            for binding in results:
                # Substitute bound variables
                s = self._substitute(c_subj, binding) if not c_subj.startswith("?") else None
                r = self._substitute(c_rel, binding) if not c_rel.startswith("?") else None
                o = self._substitute(c_obj, binding) if not c_obj.startswith("?") else None

                candidates = self.query(subject=s, relation=r, object_=o)
                for cand in candidates:
                    next_binding = dict(binding)
                    valid = True

                    if c_subj.startswith("?"):
                        if c_subj in next_binding and next_binding[c_subj] != cand.subject:
                            valid = False
                        next_binding[c_subj] = cand.subject

                    if c_rel.startswith("?"):
                        if c_rel in next_binding and next_binding[c_rel] != cand.relation:
                            valid = False
                        next_binding[c_rel] = cand.relation

                    if c_obj.startswith("?"):
                        if c_obj in next_binding and next_binding[c_obj] != cand.object_:
                            valid = False
                        next_binding[c_obj] = cand.object_

                    if valid:
                        new_results.append(next_binding)

            results = new_results
            if not results:
                break

        return results

    def _substitute(self, term: str, binding: Dict[str, str]) -> str:
        return binding.get(term, term)

    def get_all_facts(self) -> List[Triple]:
        return list(self._triples)

    def summary(self) -> str:
        return f"SemanticGraph(facts={len(self._triples)}, rules={len(self._rules)})"
