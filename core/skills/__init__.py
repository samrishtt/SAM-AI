"""Contract-based skills library for Micro-AGI."""

from core.skills.web_skills import (
    ContractSkill,
    SkillExecutionResult,
    WebSearchSkill,
    ReadWebDocumentSkill,
    FactExtractionSkill,
)

__all__ = [
    "ContractSkill",
    "SkillExecutionResult",
    "WebSearchSkill",
    "ReadWebDocumentSkill",
    "FactExtractionSkill",
]
