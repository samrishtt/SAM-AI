"""Autonomous Computer & Browser Operator.

Coordinates high-level digital workflows across browsers, APIs, file structures,
and sandboxed execution runtimes, implementing the Operator paradigm with
contract-governed action guarantees.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from core.environment.web_environment import WebEnvironment, WebPage
from core.skills.web_skills import (
    WebSearchSkill,
    ReadWebDocumentSkill,
    FactExtractionSkill,
    SkillExecutionResult,
)
from core.execution.ast_sandbox import PythonASTSandbox, ExecutionResult
from core.security.adversarial_shield import AdversarialShield, ThreatAssessment


@dataclass
class WorkflowActionTrace:
    step_id: int
    action_type: str
    target: str
    status: str
    duration_ms: float
    output_snippet: str


@dataclass
class OperatorWorkflowResult:
    goal: str
    success: bool
    final_output: Any
    steps_executed: int
    traces: List[WorkflowActionTrace]
    threats_blocked: int
    elapsed_time_sec: float


class ComputerOperator:
    """Autonomous agent operating across web environments and computational sandboxes."""

    def __init__(self):
        self.env = WebEnvironment()
        self.sandbox = PythonASTSandbox()
        self.shield = AdversarialShield()

        # Contract skills
        self.search_skill = WebSearchSkill(self.env)
        self.read_skill = ReadWebDocumentSkill(self.env)
        self.fact_skill = FactExtractionSkill()

    def execute_workflow(
        self,
        goal: str,
        workflow_steps: List[Dict[str, Any]],
    ) -> OperatorWorkflowResult:
        """Executes a multi-step digital workflow with contract guarantees and IPI protection."""
        start_time = time.perf_counter()
        traces: List[WorkflowActionTrace] = []
        threats_count = 0
        execution_context: Dict[str, Any] = {}
        success = True
        final_output = None

        for idx, step in enumerate(workflow_steps, 1):
            s_start = time.perf_counter()
            action_type = step.get("type", "").lower()
            target = step.get("target", "")

            # 1. Action: Web Search
            if action_type == "web_search":
                res: SkillExecutionResult = self.search_skill.run(query=target)
                status = "SUCCESS" if res.success else "FAILED"
                output = str(res.data[:2] if res.data else res.error)
                execution_context["search_results"] = res.data

            # 2. Action: Read Web Page
            elif action_type == "read_web_page":
                res: SkillExecutionResult = self.read_skill.run(url=target)
                if res.success:
                    page: WebPage = res.data
                    # Run through adversarial shield
                    threat = self.shield.inspect_and_sanitize(page.text_content)
                    if not threat.is_safe:
                        threats_count += len(threat.detected_threats)
                    page.text_content = threat.sanitized_content
                    execution_context["current_page"] = page
                    status = "SUCCESS"
                    output = page.summary(100)
                else:
                    status = "FAILED"
                    output = str(res.error)

            # 3. Action: Run Python Code in AST Sandbox
            elif action_type == "run_python_code":
                code = step.get("code", "")
                res_code: ExecutionResult = self.sandbox.execute(code, initial_vars=execution_context)
                status = "SUCCESS" if res_code.success else "FAILED"
                output = f"Return: {res_code.return_value} | Out: {res_code.output.strip()}"
                execution_context["computation_result"] = res_code.return_value
                final_output = res_code.return_value

            # 4. Action: Extract Facts
            elif action_type == "extract_facts":
                text = step.get("text", "") or (execution_context.get("current_page").text_content if "current_page" in execution_context else "")
                res_fact = self.fact_skill.run(text=text)
                status = "SUCCESS" if res_fact.success else "FAILED"
                output = f"Extracted {len(res_fact.data or [])} facts."
                execution_context["facts"] = res_fact.data
                final_output = res_fact.data

            else:
                status = "UNKNOWN_ACTION"
                output = f"Unsupported action: {action_type}"
                success = False

            elapsed_ms = (time.perf_counter() - s_start) * 1000.0
            traces.append(
                WorkflowActionTrace(
                    step_id=idx,
                    action_type=action_type,
                    target=target,
                    status=status,
                    duration_ms=elapsed_ms,
                    output_snippet=output,
                )
            )

            if status == "FAILED":
                success = False
                break

        total_elapsed = time.perf_counter() - start_time
        return OperatorWorkflowResult(
            goal=goal,
            success=success,
            final_output=final_output or execution_context,
            steps_executed=len(traces),
            traces=traces,
            threats_blocked=threats_count,
            elapsed_time_sec=total_elapsed,
        )
