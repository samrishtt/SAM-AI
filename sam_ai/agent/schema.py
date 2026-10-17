#!/usr/bin/env python3
"""
SAM-AI Action Schema & Validation Module
========================================
Defines strict schemas, parsers, and validation rules for ARC-AGI-3 actions.
Rejects malformed or illegal actions before environment execution.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class AllowedAction(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"  # Up
    ACTION2 = "ACTION2"  # Down
    ACTION3 = "ACTION3"  # Left
    ACTION4 = "ACTION4"  # Right
    ACTION5 = "ACTION5"  # Interact / Action
    ACTION6 = "ACTION6"  # Coordinate click
    ACTION7 = "ACTION7"  # Undo


@dataclass
class ParsedAction:
    """Structured representation of a parsed agent action."""
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    plan: str = ""
    raw_text: str = ""
    is_valid: bool = False
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "data": self.data,
            "thought": self.thought,
            "plan": self.plan,
            "is_valid": self.is_valid,
            "validation_error": self.validation_error,
        }


class ActionParser:
    """Strict parser and validator for model outputs in ARC-3."""

    @staticmethod
    def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from text, handling markdown fences and loose json."""
        if not text or not text.strip():
            return None

        # 1. Try markdown code block
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # 2. Try outermost curly braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 3. Direct parse
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    @classmethod
    def parse_and_validate(
        cls,
        model_text: str,
        available_actions: Optional[List[str]] = None,
        game_state: str = "NOT_FINISHED",
    ) -> ParsedAction:
        """
        Parses model text against strict schema and validates game-state invariants.
        Returns a ParsedAction with is_valid=True only if all checks pass.
        """
        raw_json = cls.extract_json_block(model_text)
        if raw_json is None:
            return ParsedAction(
                action="UNKNOWN",
                raw_text=model_text,
                is_valid=False,
                validation_error="Malformed response: no valid JSON block found in output.",
            )

        thought = str(raw_json.get("thought", ""))
        plan = str(raw_json.get("plan", ""))
        action_name = str(raw_json.get("action", "")).strip().upper()
        data = raw_json.get("data", {})
        if not isinstance(data, dict):
            data = {}

        # 1. Validate action string against AllowedAction enum
        valid_action_names = {a.value for a in AllowedAction}
        if action_name not in valid_action_names:
            return ParsedAction(
                action=action_name,
                data=data,
                thought=thought,
                plan=plan,
                raw_text=model_text,
                is_valid=False,
                validation_error=f"Illegal action '{action_name}'. Must be one of: {sorted(valid_action_names)}",
            )

        # 2. Check game_state: In GAME_OVER, only RESET is valid
        if game_state.upper() == "GAME_OVER" and action_name != "RESET":
            return ParsedAction(
                action=action_name,
                data=data,
                thought=thought,
                plan=plan,
                raw_text=model_text,
                is_valid=False,
                validation_error="Illegal action: Game is in GAME_OVER state; only RESET is allowed.",
            )

        # 3. Check environment availability if supplied
        if available_actions is not None and len(available_actions) > 0:
            avail_upper = {str(a).strip().upper() for a in available_actions}
            if action_name not in avail_upper and action_name != "RESET":
                return ParsedAction(
                    action=action_name,
                    data=data,
                    thought=thought,
                    plan=plan,
                    raw_text=model_text,
                    is_valid=False,
                    validation_error=f"Action '{action_name}' is not in available actions: {sorted(avail_upper)}",
                )

        # 4. Check ACTION6 coordinate requirements
        if action_name == "ACTION6":
            if "x" not in data or "y" not in data:
                return ParsedAction(
                    action=action_name,
                    data=data,
                    thought=thought,
                    plan=plan,
                    raw_text=model_text,
                    is_valid=False,
                    validation_error="ACTION6 requires 'data' with integer 'x' and 'y' coordinates.",
                )
            try:
                x = int(data["x"])
                y = int(data["y"])
                if not (0 <= x <= 63 and 0 <= y <= 63):
                    return ParsedAction(
                        action=action_name,
                        data=data,
                        thought=thought,
                        plan=plan,
                        raw_text=model_text,
                        is_valid=False,
                        validation_error=f"Coordinates out of bounds: ({x}, {y}) must be in range [0, 63].",
                    )
                data = {"x": x, "y": y}
            except (ValueError, TypeError) as e:
                return ParsedAction(
                    action=action_name,
                    data=data,
                    thought=thought,
                    plan=plan,
                    raw_text=model_text,
                    is_valid=False,
                    validation_error=f"Coordinates must be integers, got: {data}",
                )

        return ParsedAction(
            action=action_name,
            data=data,
            thought=thought,
            plan=plan,
            raw_text=model_text,
            is_valid=True,
            validation_error=None,
        )


# Alias for backward compatibility
ActionSchemaParser = ActionParser

