"""Contract-Based Web Skill Library.

Implements the ContractSkill and PANDO architectural paradigm:
each autonomous web operation is encapsulated with explicit preconditions,
postconditions, and automated self-repair fallback mechanisms.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.environment.web_environment import WebEnvironment, WebPage


@dataclass
class SkillExecutionResult:
    success: bool
    data: Any
    error: Optional[str] = None
    repaired: bool = False
    repair_action: Optional[str] = None


class ContractSkill:
    """Base class for contract-governed autonomous agent actions."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def check_preconditions(self, **kwargs) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def execute_core(self, **kwargs) -> Any:
        raise NotImplementedError

    def check_postconditions(self, result: Any, **kwargs) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def self_repair(self, error_msg: str, **kwargs) -> SkillExecutionResult:
        """Fallback routine triggered when postconditions or execution fails."""
        return SkillExecutionResult(success=False, data=None, error=error_msg)

    def run(self, **kwargs) -> SkillExecutionResult:
        """Executes skill under contract guarantees."""
        # 1. Precondition verification
        ok, pre_err = self.check_preconditions(**kwargs)
        if not ok:
            return SkillExecutionResult(success=False, data=None, error=f"Precondition failed: {pre_err}")

        # 2. Core execution
        try:
            raw_res = self.execute_core(**kwargs)
        except Exception as e:
            return self.self_repair(str(e), **kwargs)

        # 3. Postcondition verification
        post_ok, post_err = self.check_postconditions(raw_res, **kwargs)
        if not post_ok:
            return self.self_repair(post_err or "Postcondition failure", **kwargs)

        return SkillExecutionResult(success=True, data=raw_res)


class WebSearchSkill(ContractSkill):
    """Searches the open web with automated query refinement fallback."""

    def __init__(self, env: WebEnvironment):
        super().__init__(
            name="web_search",
            description="Searches live web encyclopedias and sources for relevant pages.",
        )
        self.env = env

    def check_preconditions(self, **kwargs) -> Tuple[bool, Optional[str]]:
        query = kwargs.get("query", "").strip()
        if not query:
            return False, "Query string cannot be empty."
        return True, None

    def execute_core(self, **kwargs) -> List[Dict[str, str]]:
        query = kwargs["query"]
        limit = kwargs.get("limit", 4)
        return self.env.search_wikipedia(query=query, limit=limit)

    def check_postconditions(self, result: Any, **kwargs) -> Tuple[bool, Optional[str]]:
        if not isinstance(result, list) or len(result) == 0:
            return False, "Search returned empty results."
        if all(not r.get("url") for r in result):
            return False, "No valid URLs resolved in search results."
        return True, None

    def self_repair(self, error_msg: str, **kwargs) -> SkillExecutionResult:
        # Self-repair: strip stop-words and simplify query
        query = kwargs.get("query", "")
        simplified = " ".join([w for w in query.split() if len(w) > 3][:3])
        if simplified and simplified != query:
            try:
                res = self.env.search_wikipedia(query=simplified, limit=3)
                if res and any(r.get("url") for r in res):
                    return SkillExecutionResult(
                        success=True,
                        data=res,
                        repaired=True,
                        repair_action=f"Simplified query from '{query}' to '{simplified}'",
                    )
            except Exception:
                pass
        return SkillExecutionResult(success=False, data=[], error=error_msg)


class ReadWebDocumentSkill(ContractSkill):
    """Fetches and distills live web pages with anti-injection defenses."""

    def __init__(self, env: WebEnvironment):
        super().__init__(
            name="read_web_document",
            description="Navigates to a URL, sanitizes content, and extracts structured text.",
        )
        self.env = env

    def check_preconditions(self, **kwargs) -> Tuple[bool, Optional[str]]:
        url = kwargs.get("url", "").strip()
        if not url:
            return False, "URL cannot be empty."
        return True, None

    def execute_core(self, **kwargs) -> WebPage:
        url = kwargs["url"]
        max_chars = kwargs.get("max_chars", 4000)
        return self.env.navigate(url=url, max_chars=max_chars)

    def check_postconditions(self, result: Any, **kwargs) -> Tuple[bool, Optional[str]]:
        if not isinstance(result, WebPage):
            return False, "Invalid return type; expected WebPage."
        if result.status_code >= 400:
            return False, f"HTTP Error status {result.status_code}."
        if len(result.text_content.strip()) < 30:
            return False, "Page returned insufficient text content."
        return True, None

    def self_repair(self, error_msg: str, **kwargs) -> SkillExecutionResult:
        url = kwargs.get("url", "")
        # If http failed, try https or vice versa
        alt_url = url.replace("http://", "https://") if url.startswith("http://") else url
        if alt_url != url:
            try:
                page = self.env.navigate(alt_url)
                if page.status_code == 200:
                    return SkillExecutionResult(
                        success=True,
                        data=page,
                        repaired=True,
                        repair_action=f"Repaired URL scheme to {alt_url}",
                    )
            except Exception:
                pass
        return SkillExecutionResult(success=False, data=None, error=error_msg)


class FactExtractionSkill(ContractSkill):
    """Distills declarative claims and scientific assertions from raw text."""

    def __init__(self):
        super().__init__(
            name="extract_facts",
            description="Extracts key entity relations and factual assertions from text.",
        )

    def check_preconditions(self, **kwargs) -> Tuple[bool, Optional[str]]:
        text = kwargs.get("text", "").strip()
        if len(text) < 20:
            return False, "Input text is too short for fact extraction."
        return True, None

    def execute_core(self, **kwargs) -> List[str]:
        text = kwargs["text"]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter for sentences containing informative definitional or relational terms
        informative = []
        keywords = ["is a", "is an", "refers to", "developed", "demonstrated", "achieved", "consists of", "known as"]
        for s in sentences:
            s_clean = s.strip()
            if any(k in s_clean.lower() for k in keywords) and 25 < len(s_clean) < 300:
                informative.append(s_clean)
        return informative[:5] if informative else [sentences[0]]

    def check_postconditions(self, result: Any, **kwargs) -> Tuple[bool, Optional[str]]:
        if not isinstance(result, list) or len(result) == 0:
            return False, "No factual claims could be extracted."
        return True, None
