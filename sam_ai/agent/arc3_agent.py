#!/usr/bin/env python3
"""
SAM-AI ARC-AGI-3 Agent Core Implementation
==========================================
Integrates perception, state representation, model querying, strict action schema
parsing, pre-execution verification, and task-specific failure memory into an
autonomous turn-based reasoning agent.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

from sam_ai.substrate.interfaces import (
    ModelRunner,
    ModelOutput,
    VerificationStatus,
)
from sam_ai.agent.schema import ActionParser, ParsedAction, AllowedAction
from sam_ai.agent.representation import ObservationConverter, StructuredObservation
from sam_ai.agent.memory import TaskFailureMemory, TransitionRecord
from sam_ai.agent.telemetry import CallTelemetry


@dataclass
class AgentStepResult:
    """Telemetry and outcome of a single agent step."""
    step_idx: int
    parsed_action: ParsedAction
    executed_action: str
    executed_data: Dict[str, Any]
    verification_status: VerificationStatus
    verification_error: Optional[str]
    model_output: ModelOutput
    transition: Optional[TransitionRecord] = None
    latency_ms: float = 0.0
    telemetry: Optional[CallTelemetry] = None


@dataclass
class SessionSummary:
    """Summary of an entire environment session."""
    game_id: str
    model_name: str
    is_real_model: bool
    total_steps: int
    levels_completed: int
    final_state: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_latency_ms: float
    verification_failures: int
    steps_history: List[AgentStepResult] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "model_name": self.model_name,
            "is_real_model": self.is_real_model,
            "total_steps": self.total_steps,
            "levels_completed": self.levels_completed,
            "final_state": self.final_state,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "verification_failures": self.verification_failures,
            "success": self.success,
        }


class SAMARC3Agent:
    """
    SAM-AI Turn-Based Agent for ARC-AGI-3.
    Decoupled from underlying foundation model; operates via configurable ModelRunner.
    """

    SYSTEM_PROMPT = (
        "You are SAM-AI, an advanced inductive reasoning agent solving an ARC-AGI-3 interactive puzzle.\n"
        "Rules:\n"
        "1. Discover the latent mechanics of the puzzle grid by testing hypotheses carefully.\n"
        "2. Avoid redundant or illegal moves. Never repeat moves that produced zero visual change.\n"
        "3. If the game is in GAME_OVER, the ONLY legal action is RESET.\n"
        "4. Output your response as a strict JSON object with fields:\n"
        '   {"thought": "<analysis>", "plan": "<short plan>", "action": "<ACTION_NAME>", "data": {<optional>}}\n'
        "5. Valid action names are: RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7.\n"
        "6. For ACTION6, data MUST contain {'x': int, 'y': int} with 0 <= x, y <= 63."
    )

    def __init__(
        self,
        model_runner: ModelRunner,
        enable_verification_guard: bool = True,
        enable_failure_memory: bool = True,
        max_action_retries: int = 1,
        is_real_model: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: Optional[int] = 42,
        max_tokens: int = 3072,
        use_system_prompt: bool = True,
        stop_on_parse_failure: bool = False,
    ):
        self.model = model_runner
        self.enable_verification_guard = enable_verification_guard
        self.enable_failure_memory = enable_failure_memory
        self.max_action_retries = max_action_retries
        self.is_real_model = is_real_model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.use_system_prompt = use_system_prompt
        self.stop_on_parse_failure = stop_on_parse_failure
        self.memory = TaskFailureMemory()

    def build_step_prompt(self, obs: StructuredObservation) -> str:
        """Constructs prompt containing current observation, recent history, and failure warnings."""
        sections = [obs.to_prompt_text(include_delta=True)]

        if self.enable_failure_memory:
            # Add recent trajectory
            history_text = self.memory.get_recent_history_text(count=4)
            sections.append(f"\n### Recent Action History:\n{history_text}")

            # Add active failure warnings
            warnings = self.memory.get_active_warnings()
            if warnings:
                warn_text = "\n".join(f"- ⚠️ {w}" for w in warnings)
                sections.append(f"\n### Critical Failure Warnings:\n{warn_text}")

        sections.append(
            "\n### Your Turn:\n"
            "State your thought, immediate plan, and exact action JSON object:"
        )
        return "\n".join(sections)

    def select_action(
        self,
        obs: StructuredObservation,
        step_idx: int,
    ) -> AgentStepResult:
        """
        Executes one turn of deliberation:
        1. Checks state safety.
        2. Queries model.
        3. Parses output and validates schema.
        4. Rejects illegal actions if verification guard is enabled.
        """
        start_time = time.perf_counter()

        # Safety rule: If GAME_OVER, immediately enforce RESET if guard is enabled
        if self.enable_verification_guard and obs.state.upper() == "GAME_OVER":
            parsed = ParsedAction(
                action="RESET",
                thought="Safety guard triggered: state is GAME_OVER. Disposing mandatory RESET.",
                plan="Reset level to recover.",
                is_valid=True,
            )
            telemetry = CallTelemetry.create(
                prompt_tokens=0,
                completion_tokens=0,
                model_text='{"action": "RESET"}',
                parse_valid=True,
                latency_ms=0.0,
                seed=self.seed,
            )
            return AgentStepResult(
                step_idx=step_idx,
                parsed_action=parsed,
                executed_action="RESET",
                executed_data={},
                verification_status=VerificationStatus.PASS,
                verification_error=None,
                model_output=ModelOutput(text='{"action": "RESET"}', latency_ms=0.0),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                telemetry=telemetry,
            )

        step_prompt_text = self.build_step_prompt(obs)
        if self.use_system_prompt:
            prompt = step_prompt_text
            sys_prompt = self.SYSTEM_PROMPT
        else:
            # P1c: No system prompt: put instructions in the user turn
            prompt = f"{self.SYSTEM_PROMPT}\n\n{step_prompt_text}"
            sys_prompt = None

        model_out = self.model.generate(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        parsed = ActionParser.parse_and_validate(
            model_text=model_out.text,
            available_actions=obs.available_actions,
            game_state=obs.state,
        )

        v_status = VerificationStatus.PASS if parsed.is_valid else VerificationStatus.FAIL
        v_error = parsed.validation_error

        # If invalid and stop_on_parse_failure is False and retries are allowed, attempt one corrective query
        if not parsed.is_valid and not self.stop_on_parse_failure and self.max_action_retries > 0:
            correction_prompt = (
                f"{prompt}\n\n"
                f"[REJECTION NOTICE]: Your previous action was rejected with error:\n"
                f"'{parsed.validation_error}'\n"
                f"Please correct the error and output a valid JSON action."
            )
            model_out_retry = self.model.generate(
                prompt=correction_prompt,
                system_prompt=sys_prompt,
                temperature=0.1,
                max_tokens=self.max_tokens,
            )
            retry_parsed = ActionParser.parse_and_validate(
                model_text=model_out_retry.text,
                available_actions=obs.available_actions,
                game_state=obs.state,
            )
            if retry_parsed.is_valid:
                parsed = retry_parsed
                model_out = model_out_retry
                v_status = VerificationStatus.PASS
                v_error = None

        # Execute decision
        if not parsed.is_valid:
            if self.stop_on_parse_failure:
                # P1c: If an action cannot be parsed, record parse_failed and stop the episode.
                executed_action = "parse_failed"
                executed_data = {}
            elif self.enable_verification_guard:
                # Rejection guard: Fallback to RESET or safe action
                executed_action = "RESET"
                executed_data = {}
            else:
                executed_action = "RESET"
                executed_data = {}
        else:
            executed_action = parsed.action if parsed.action != "UNKNOWN" else "RESET"
            executed_data = parsed.data

        latency = (time.perf_counter() - start_time) * 1000.0

        telemetry = CallTelemetry.create(
            prompt_tokens=model_out.prompt_tokens,
            completion_tokens=model_out.completion_tokens,
            model_text=model_out.text,
            parse_valid=parsed.is_valid,
            latency_ms=model_out.latency_ms or latency,
            seed=self.seed,
            prompt_ends_with_think=getattr(model_out, "prompt_ends_with_think", False),
            think_prefix_added=getattr(model_out, "think_prefix_added", False),
        )

        return AgentStepResult(
            step_idx=step_idx,
            parsed_action=parsed,
            executed_action=executed_action,
            executed_data=executed_data,
            verification_status=v_status,
            verification_error=v_error,
            model_output=model_out,
            latency_ms=latency,
            telemetry=telemetry,
        )

    def run_session(
        self,
        env: Any,
        game_id: str = "unknown",
        max_steps: int = 50,
        step_callback: Optional[Callable[[AgentStepResult], None]] = None,
    ) -> SessionSummary:
        """
        Executes an end-to-end interactive session with an environment wrapper.
        Compatible with official arc_agi.base.EnvironmentWrapper and compliant mock wrappers.
        """
        session_start = time.perf_counter()
        raw_obs = env.reset()
        prev_grid: Optional[List[List[int]]] = None

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0
        verification_failures = 0
        history: List[AgentStepResult] = []

        for step_idx in range(1, max_steps + 1):
            struct_obs = ObservationConverter.convert(
                obs_data=raw_obs,
                game_id=game_id,
                prev_grid=prev_grid,
                total_levels=getattr(env, "info", None).num_levels if hasattr(env, "info") and hasattr(env.info, "num_levels") else 1,
            )

            # Check if game is already won
            if struct_obs.state.upper() == "WIN":
                break

            # 1. Deliberate and select action
            step_res = self.select_action(struct_obs, step_idx=step_idx)
            total_prompt_tokens += step_res.model_output.prompt_tokens
            total_completion_tokens += step_res.model_output.completion_tokens
            total_latency_ms += step_res.latency_ms

            if step_res.verification_status != VerificationStatus.PASS:
                verification_failures += 1

            # P1c: If an action cannot be parsed, stop the episode without fallback
            if step_res.executed_action == "parse_failed":
                history.append(step_res)
                if step_callback:
                    step_callback(step_res)
                break

            # 2. Execute permitted action through environment
            state_before = struct_obs.state
            act_name = step_res.executed_action

            # Convert action string to environment action (GameAction enum) if needed
            env_action = act_name
            try:
                import arcengine.enums
                if hasattr(arcengine.enums, "GameAction") and act_name in arcengine.enums.GameAction.__members__:
                    env_action = arcengine.enums.GameAction[act_name]
            except Exception:
                pass

            try:
                # Official EnvironmentWrapper.step takes (action: GameAction, data=..., reasoning=...)
                if hasattr(env, "step"):
                    try:
                        raw_next_obs = env.step(env_action, data=step_res.executed_data)
                    except Exception:
                        # Fallback for mock environments that only accept string
                        raw_next_obs = env.step(act_name, data=step_res.executed_data)
                else:
                    raw_next_obs = None
            except Exception as e:
                # Capture execution error
                step_res.verification_error = f"Environment step error: {e}"
                raw_next_obs = None

            # 3. Post-execution transition evaluation
            next_grid, _, _ = ObservationConverter.extract_grid(raw_next_obs) if raw_next_obs else (struct_obs.grid, 0, 0)
            changed_pixels = 0
            if prev_grid is not None and len(prev_grid) == len(next_grid) and len(prev_grid[0]) == len(next_grid[0]):
                for r in range(len(next_grid)):
                    for c in range(len(next_grid[0])):
                        if next_grid[r][c] != prev_grid[r][c]:
                            changed_pixels += 1

            state_after = getattr(raw_next_obs, "state", "NOT_FINISHED")
            state_after_str = state_after.name if hasattr(state_after, "name") else str(state_after)
            levels_comp = getattr(raw_next_obs, "levels_completed", struct_obs.levels_completed)

            if self.enable_failure_memory:
                trans = self.memory.record_transition(
                    step_idx=step_idx,
                    action=step_res.executed_action,
                    data=step_res.executed_data,
                    thought=step_res.parsed_action.thought,
                    plan=step_res.parsed_action.plan,
                    state_before=state_before,
                    state_after=state_after_str,
                    levels_completed=int(levels_comp or 0),
                    changed_pixels=changed_pixels,
                )
                step_res.transition = trans

            history.append(step_res)
            if step_callback:
                step_callback(step_res)

            prev_grid = next_grid
            raw_obs = raw_next_obs

            # Stop condition
            if state_after_str.upper() == "WIN":
                break

        final_state = struct_obs.state if raw_obs is None else getattr(raw_obs, "state", struct_obs.state)
        final_state_str = final_state.name if hasattr(final_state, "name") else str(final_state)
        levels_completed = struct_obs.levels_completed if raw_obs is None else getattr(raw_obs, "levels_completed", struct_obs.levels_completed)

        return SessionSummary(
            game_id=game_id,
            model_name=self.model.model_name,
            is_real_model=self.is_real_model,
            total_steps=len(history),
            levels_completed=int(levels_completed or 0),
            final_state=final_state_str,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_latency_ms=total_latency_ms,
            verification_failures=verification_failures,
            steps_history=history,
            success=(final_state_str.upper() == "WIN"),
        )
