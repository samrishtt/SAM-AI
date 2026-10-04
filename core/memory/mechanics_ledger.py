"""
Persistent Mechanics & Causal Rule Ledger for Interactive Long-Horizon Environments.
Fixes Flaw 3 (Context Eviction Amnesia).

Tracks discovered game dynamics, hazard rules, entity roles, and verified subroutines
across levels and episodes. Injects compact distilled priors into prompt contexts (<250 tokens),
preventing repetitive fatal mistakes and preserving game physics knowledge.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class CausalRule:
    """Represents an observed or verified causal rule: Condition -> Effect."""
    condition: str
    effect: str
    confidence: float = 1.0
    observations: int = 1
    is_lethal: bool = False
    is_goal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityRole:
    """Identifies the functional role of a color or visual component."""
    entity_id: str  # e.g. "Color_1", "Shape_Cross"
    role: str       # "Player", "Hazard", "Goal", "Pushable", "Key", "LockedDoor", "Wall"
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeSubroutine:
    """Stores reusable DSL or Python functions verified in previous levels."""
    name: str
    signature: str
    description: str
    code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MechanicsLedger:
    """
    Persistent store for verified environment physics, game rules, and entity roles.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.rules: Dict[str, CausalRule] = {}
        self.entities: Dict[str, EntityRole] = {}
        self.subroutines: Dict[str, CodeSubroutine] = {}

        if self.persistence_path and os.path.exists(self.persistence_path):
            self.load_from_disk()

    def record_rule(
        self,
        condition: str,
        effect: str,
        confidence: float = 1.0,
        is_lethal: bool = False,
        is_goal: bool = False
    ) -> CausalRule:
        """Adds or updates an observed causal rule."""
        key = f"{condition} -> {effect}".lower().strip()
        if key in self.rules:
            rule = self.rules[key]
            rule.observations += 1
            rule.confidence = min(1.0, rule.confidence + 0.1)
            rule.is_lethal = rule.is_lethal or is_lethal
            rule.is_goal = rule.is_goal or is_goal
        else:
            rule = CausalRule(
                condition=condition,
                effect=effect,
                confidence=confidence,
                observations=1,
                is_lethal=is_lethal,
                is_goal=is_goal
            )
            self.rules[key] = rule

        if self.persistence_path:
            self.save_to_disk()
        return rule

    def register_entity(
        self,
        entity_id: str,
        role: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> EntityRole:
        """Registers semantic role for a color or visual component."""
        ent = EntityRole(
            entity_id=str(entity_id),
            role=role,
            properties=properties or {},
            confidence=confidence
        )
        self.entities[str(entity_id)] = ent
        if self.persistence_path:
            self.save_to_disk()
        return ent

    def register_subroutine(
        self,
        name: str,
        signature: str,
        description: str,
        code: str
    ) -> CodeSubroutine:
        """Registers a reusable Python subroutine."""
        sub = CodeSubroutine(
            name=name,
            signature=signature,
            description=description,
            code=code
        )
        self.subroutines[name] = sub
        if self.persistence_path:
            self.save_to_disk()
        return sub

    def format_prompt_injection(self) -> str:
        """
        Produces a compact, ultra-dense prompt injection block (<200 tokens)
        designed to prevent context window amnesia.
        """
        lines = ["<mechanics_ledger>"]

        if self.entities:
            lines.append("[ENTITY ROLES]")
            for ent_id, ent in self.entities.items():
                lines.append(f"- {ent_id}: {ent.role}")

        if self.rules:
            lines.append("[CAUSAL RULES]")
            for rule in self.rules.values():
                tag = " [LETHAL]" if rule.is_lethal else (" [GOAL]" if rule.is_goal else "")
                lines.append(f"- IF {rule.condition} THEN {rule.effect}{tag} (conf={rule.confidence:.2f})")

        if self.subroutines:
            lines.append("[AVAILABLE SUBROUTINES]")
            for name, sub in self.subroutines.items():
                lines.append(f"- {sub.signature}: {sub.description}")

        lines.append("</mechanics_ledger>")
        return "\n".join(lines)

    def save_to_disk(self) -> None:
        """Serializes current ledger to disk."""
        if not self.persistence_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)
        data = {
            "rules": [r.to_dict() for r in self.rules.values()],
            "entities": [e.to_dict() for e in self.entities.values()],
            "subroutines": [s.to_dict() for s in self.subroutines.values()]
        }
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self) -> None:
        """Loads ledger from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        with open(self.persistence_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.rules = {}
        for r_data in data.get("rules", []):
            rule = CausalRule(**r_data)
            key = f"{rule.condition} -> {rule.effect}".lower().strip()
            self.rules[key] = rule

        self.entities = {}
        for e_data in data.get("entities", []):
            ent = EntityRole(**e_data)
            self.entities[ent.entity_id] = ent

        self.subroutines = {}
        for s_data in data.get("subroutines", []):
            sub = CodeSubroutine(**s_data)
            self.subroutines[sub.name] = sub
