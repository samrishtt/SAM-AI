"""
SAM-AI Autonomous Agent Package
===============================
Provides the turn-based interactive ARC-3 agent, action schemas,
observation converters, and task failure memory.
"""

from sam_ai.agent.schema import ActionParser, ParsedAction, AllowedAction
from sam_ai.agent.representation import ObservationConverter, StructuredObservation
from sam_ai.agent.memory import TaskFailureMemory, TransitionRecord
from sam_ai.agent.arc3_agent import SAMARC3Agent, AgentStepResult, SessionSummary

__all__ = [
    "ActionParser",
    "ParsedAction",
    "AllowedAction",
    "ObservationConverter",
    "StructuredObservation",
    "TaskFailureMemory",
    "TransitionRecord",
    "SAMARC3Agent",
    "AgentStepResult",
    "SessionSummary",
]
