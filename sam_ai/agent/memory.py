#!/usr/bin/env python3
"""
SAM-AI Task-Specific Failure Memory & Trajectory Tracker
=========================================================
Tracks recent transitions, evaluates outcomes, and maintains a limited
failure memory for the active puzzle to prevent repeating known dead-ends.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time

from sam_ai.substrate.interfaces import (
    FailureCategory,
    FailureRecord,
    VerificationResult,
    VerificationStatus,
)


@dataclass
class TransitionRecord:
    """Record of a single environment interaction step."""
    step_idx: int
    action: str
    data: Dict[str, Any]
    thought: str
    plan: str
    state_before: str
    state_after: str
    levels_completed: int
    changed_pixels: int
    is_failure: bool = False
    failure_category: Optional[FailureCategory] = None
    failure_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_summary_line(self) -> str:
        res = f"Step {self.step_idx}: {self.action}"
        if self.data:
            res += f"({self.data})"
        res += f" -> State: {self.state_after}, Changed: {self.changed_pixels} px"
        if self.is_failure:
            res += f" [FAIL: {self.failure_reason}]"
        return res


class TaskFailureMemory:
    """
    Limited task-specific failure memory for the active environment.
    Registers dead-ends (no-op wall collisions, hazard deaths, schema rejects)
    and surfaces concise warnings in subsequent model prompts.
    """

    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        self.transitions: List[TransitionRecord] = []
        self.no_op_actions: Dict[str, int] = {}  # action -> consecutive count
        self.fatal_actions: List[Tuple[str, Dict[str, Any]]] = []
        self.discovered_rules: List[str] = []

    def record_transition(
        self,
        step_idx: int,
        action: str,
        data: Dict[str, Any],
        thought: str,
        plan: str,
        state_before: str,
        state_after: str,
        levels_completed: int,
        changed_pixels: int,
    ) -> TransitionRecord:
        """Evaluates outcome of step and registers failures if detected."""
        is_failure = False
        cat = None
        reason = None

        if state_after.upper() == "GAME_OVER":
            is_failure = True
            cat = FailureCategory.EXECUTION
            reason = f"Triggered GAME_OVER terminal hazard."
            self.fatal_actions.append((action, data))
        elif changed_pixels == 0 and action != "RESET":
            is_failure = True
            cat = FailureCategory.PERCEPTION
            cnt = self.no_op_actions.get(action, 0) + 1
            self.no_op_actions[action] = cnt
            reason = f"Action had zero visual effect ({cnt}x no-op). Potential obstacle or ineffective interaction."
        else:
            # Successful state transition resets consecutive no-op count for this action
            if action in self.no_op_actions:
                self.no_op_actions[action] = 0

        record = TransitionRecord(
            step_idx=step_idx,
            action=action,
            data=data,
            thought=thought,
            plan=plan,
            state_before=state_before,
            state_after=state_after,
            levels_completed=levels_completed,
            changed_pixels=changed_pixels,
            is_failure=is_failure,
            failure_category=cat,
            failure_reason=reason,
        )
        self.transitions.append(record)
        return record

    def get_recent_history_text(self, count: int = 5) -> str:
        """Returns the last N step outcomes."""
        recent = self.transitions[-count:] if self.transitions else []
        if not recent:
            return "No previous steps in this session."
        return "\n".join(t.to_summary_line() for t in recent)

    def get_active_warnings(self) -> List[str]:
        """Returns concise warnings about recently failed actions to avoid repetition."""
        warnings = []
        for act, cnt in self.no_op_actions.items():
            if cnt >= 2:
                warnings.append(f"AVOID repeating '{act}': produced 0 changes for the last {cnt} attempts (blocked).")
        if self.fatal_actions:
            last_fatal = self.fatal_actions[-1]
            warnings.append(f"CAUTION: '{last_fatal[0]}' recently caused GAME_OVER. Avoid in similar configurations.")
        return warnings

    def to_substrate_failure_record(
        self,
        task_id: str,
        model_name: str,
        transition: TransitionRecord,
    ) -> FailureRecord:
        """Converts an agent transition failure into a standardized substrate FailureRecord."""
        return FailureRecord(
            task_id=task_id,
            domain="arc3",
            model_revision=model_name,
            strategy="sam_ai_arc3_agent",
            candidate=f"{transition.action}({transition.data})",
            output=transition.to_summary_line(),
            verification_result=VerificationResult(
                status=VerificationStatus.FAIL,
                confidence=0.0,
                details=transition.failure_reason or "Transition failure",
            ),
            failure_category=transition.failure_category or FailureCategory.EXECUTION,
            failure_details=transition.failure_reason or "Execution error",
            timestamp=transition.timestamp,
        )
