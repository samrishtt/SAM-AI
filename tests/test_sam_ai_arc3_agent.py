#!/usr/bin/env python3
"""
SAM-AI ARC-AGI-3 Agent Test Suite (TEST_ONLY)
==============================================
Validates:
1. Strict schema parsing and action extraction.
2. Rejection of malformed JSON, out-of-bounds coordinates, and illegal actions.
3. Safety guard enforcement during GAME_OVER.
4. Observation conversion, ASCII rendering, and frame delta diffing.
5. Task failure memory and repetitive no-op warning generation.
6. End-to-end interactive session loop on a compliant mock environment.
7. Clean failure handling on absent / unreachable models.

NOTE: All tests in this file run with deterministic fake runners and are strictly labeled TEST_ONLY.
"""

import unittest
from typing import Dict, Any, List, Optional
import numpy as np

from sam_ai.substrate.interfaces import VerificationStatus, ModelOutput
from sam_ai.substrate.models import DeterministicModelRunner, OpenAICompatibleModelRunner
from sam_ai.agent.schema import ActionParser, ParsedAction, AllowedAction
from sam_ai.agent.representation import ObservationConverter, StructuredObservation
from sam_ai.agent.memory import TaskFailureMemory, TransitionRecord
from sam_ai.agent.arc3_agent import SAMARC3Agent, SessionSummary


class MockArcEnvironment:
    """Compliant mock environment for deterministic offline plumbing tests."""

    def __init__(self, target_coord: tuple = (2, 2)):
        self.target_coord = target_coord
        self.player_pos = [0, 0]
        self.grid = np.zeros((5, 5), dtype=int)
        self.grid[self.target_coord[0], self.target_coord[1]] = 3  # Goal (Green)
        self.grid[self.player_pos[0], self.player_pos[1]] = 1      # Player (Blue)
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.num_levels = 1
        self.step_count = 0

    def reset(self):
        self.player_pos = [0, 0]
        self.grid = np.zeros((5, 5), dtype=int)
        self.grid[self.target_coord[0], self.target_coord[1]] = 3
        self.grid[self.player_pos[0], self.player_pos[1]] = 1
        self.state = "NOT_FINISHED"
        self.step_count = 0
        return self._make_obs()

    def _make_obs(self):
        class MockObs:
            def __init__(self, grid, state, levels_completed):
                self.frame = [grid.copy()]
                self.state = state
                self.levels_completed = levels_completed
                self.available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        return MockObs(self.grid, self.state, self.levels_completed)

    def step(self, action: str, data: Optional[Dict[str, Any]] = None):
        self.step_count += 1
        act = action.upper()

        if self.state == "GAME_OVER" and act != "RESET":
            raise RuntimeError("HTTP 400 Bad Request: only RESET allowed in GAME_OVER")

        if act == "RESET":
            return self.reset()

        # Clear old player pos
        self.grid[self.player_pos[0], self.player_pos[1]] = 0

        # Moves: ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right
        if act == "ACTION1":  # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif act == "ACTION2":  # Down
            self.player_pos[0] = min(4, self.player_pos[0] + 1)
        elif act == "ACTION3":  # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif act == "ACTION4":  # Right
            self.player_pos[1] = min(4, self.player_pos[1] + 1)
        elif act == "ACTION6" and data:
            # Teleport/click
            x, y = data.get("x", 0), data.get("y", 0)
            if 0 <= y < 5 and 0 <= x < 5:
                self.player_pos = [y, x]

        # Put player at new pos
        self.grid[self.player_pos[0], self.player_pos[1]] = 1

        # Check win condition
        if self.player_pos[0] == self.target_coord[0] and self.player_pos[1] == self.target_coord[1]:
            self.state = "WIN"
            self.levels_completed = 1

        # Trap condition (visiting 4, 4 triggers GAME_OVER)
        if self.player_pos == [4, 4]:
            self.state = "GAME_OVER"

        return self._make_obs()


class TestActionSchemaAndValidation(unittest.TestCase):
    """TEST_ONLY: Tests for strict action parsing and validation."""

    def test_valid_json_action(self):
        text = '{"thought": "Moving right", "plan": "Reach goal", "action": "ACTION4", "data": {}}'
        parsed = ActionParser.parse_and_validate(text)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.action, "ACTION4")
        self.assertIsNone(parsed.validation_error)

    def test_markdown_fence_action(self):
        text = "Here is my move:\n```json\n{\n  \"thought\": \"Click button\",\n  \"action\": \"ACTION6\",\n  \"data\": {\"x\": 15, \"y\": 32}\n}\n```"
        parsed = ActionParser.parse_and_validate(text)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.action, "ACTION6")
        self.assertEqual(parsed.data, {"x": 15, "y": 32})

    def test_malformed_json_rejection(self):
        text = "I think we should move down: ACTION2"
        parsed = ActionParser.parse_and_validate(text)
        self.assertFalse(parsed.is_valid)
        self.assertIn("no valid JSON block", parsed.validation_error)

    def test_illegal_action_name_rejection(self):
        text = '{"thought": "Jump over wall", "action": "JUMP"}'
        parsed = ActionParser.parse_and_validate(text)
        self.assertFalse(parsed.is_valid)
        self.assertIn("Illegal action 'JUMP'", parsed.validation_error)

    def test_action6_out_of_bounds_rejection(self):
        text = '{"action": "ACTION6", "data": {"x": 120, "y": 5}}'
        parsed = ActionParser.parse_and_validate(text)
        self.assertFalse(parsed.is_valid)
        self.assertIn("out of bounds", parsed.validation_error)

    def test_game_over_non_reset_rejection(self):
        text = '{"thought": "Try moving", "action": "ACTION1"}'
        parsed = ActionParser.parse_and_validate(text, game_state="GAME_OVER")
        self.assertFalse(parsed.is_valid)
        self.assertIn("only RESET is allowed", parsed.validation_error)

    def test_game_over_reset_allowed(self):
        text = '{"thought": "Restarting", "action": "RESET"}'
        parsed = ActionParser.parse_and_validate(text, game_state="GAME_OVER")
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.action, "RESET")


class TestObservationRepresentation(unittest.TestCase):
    """TEST_ONLY: Tests for observation representation and spatial diffing."""

    def test_observation_conversion(self):
        raw_grid = np.array([[0, 1], [3, 0]])
        class DummyObs:
            frame = [raw_grid]
            state = "NOT_FINISHED"
            levels_completed = 0
            available_actions = ["ACTION1", "ACTION2"]

        struct = ObservationConverter.convert(DummyObs(), game_id="test_game")
        self.assertEqual(struct.height, 2)
        self.assertEqual(struct.width, 2)
        self.assertEqual(struct.color_counts, {0: 2, 1: 1, 3: 1})
        self.assertEqual(struct.ascii_grid, ".B\nG.")
        self.assertIn("Available Actions: ['ACTION1', 'ACTION2']", struct.to_prompt_text())

    def test_spatial_diff_detection(self):
        prev = [[0, 1], [0, 0]]
        curr = [[0, 0], [0, 1]]
        class DummyObs:
            frame = [np.array(curr)]
            state = "NOT_FINISHED"

        struct = ObservationConverter.convert(DummyObs(), prev_grid=prev)
        self.assertEqual(struct.changed_pixels, 2)
        self.assertEqual(set(struct.changed_coords), {(0, 1), (1, 1)})


class TestTaskFailureMemory(unittest.TestCase):
    """TEST_ONLY: Tests for task failure memory and no-op warnings."""

    def test_noop_warning_accumulation(self):
        mem = TaskFailureMemory()
        # Record 2 consecutive no-ops for ACTION1
        mem.record_transition(
            step_idx=1, action="ACTION1", data={}, thought="", plan="",
            state_before="NOT_FINISHED", state_after="NOT_FINISHED",
            levels_completed=0, changed_pixels=0
        )
        self.assertEqual(len(mem.get_active_warnings()), 0)

        mem.record_transition(
            step_idx=2, action="ACTION1", data={}, thought="", plan="",
            state_before="NOT_FINISHED", state_after="NOT_FINISHED",
            levels_completed=0, changed_pixels=0
        )
        warnings = mem.get_active_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("AVOID repeating 'ACTION1'", warnings[0])

    def test_game_over_hazard_warning(self):
        mem = TaskFailureMemory()
        mem.record_transition(
            step_idx=1, action="ACTION2", data={}, thought="", plan="",
            state_before="NOT_FINISHED", state_after="GAME_OVER",
            levels_completed=0, changed_pixels=1
        )
        warnings = mem.get_active_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("recently caused GAME_OVER", warnings[0])


class TestAgentExecutionAndMockSession(unittest.TestCase):
    """TEST_ONLY: Tests for agent execution loop and safety fallbacks."""

    def test_verification_guard_fallback_on_malformed_output(self):
        # Fake runner always emits malformed gibberish
        fake_runner = DeterministicModelRunner(
            name="fake-gibberish-runner",
            default_response="Thinking out loud: maybe I should press button 1..."
        )
        agent = SAMARC3Agent(
            model_runner=fake_runner,
            enable_verification_guard=True,
            is_real_model=False,
        )
        obs = StructuredObservation(
            game_id="test", state="NOT_FINISHED", levels_completed=0,
            total_levels=1, height=2, width=2, grid=[[0, 0], [0, 0]],
            ascii_grid="..\n..", color_counts={0: 4}, available_actions=["ACTION1", "RESET"]
        )

        step_res = agent.select_action(obs, step_idx=1)
        self.assertEqual(step_res.verification_status, VerificationStatus.FAIL)
        # Verification guard caught failure and safely enforced fallback RESET
        self.assertEqual(step_res.executed_action, "RESET")

    def test_end_to_end_mock_game_solution(self):
        # Program deterministic model to solve the mock game:
        # Start at (0,0), Goal at (2,2).
        # Optimal path: ACTION2, ACTION2, ACTION4, ACTION4.
        responses = {
            "Step 1": '{"action": "ACTION2", "thought": "Move down 1", "plan": "Reach row 2"}',
            "Step 2": '{"action": "ACTION2", "thought": "Move down 2", "plan": "Reach row 2"}',
            "Step 3": '{"action": "ACTION4", "thought": "Move right 1", "plan": "Reach col 2"}',
            "Step 4": '{"action": "ACTION4", "thought": "Move right 2", "plan": "Reach goal"}',
        }
        call_count = [0]
        def generator(prompt):
            call_count[0] += 1
            step_key = f"Step {min(call_count[0], 4)}"
            return responses[step_key]

        fake_solver = DeterministicModelRunner(
            name="fake-optimal-solver",
            generator_fn=generator
        )

        agent = SAMARC3Agent(
            model_runner=fake_solver,
            enable_verification_guard=True,
            enable_failure_memory=True,
            is_real_model=False,
        )

        env = MockArcEnvironment(target_coord=(2, 2))
        summary = agent.run_session(env, game_id="mock_puzzle_01", max_steps=10)

        self.assertTrue(summary.success)
        self.assertEqual(summary.final_state, "WIN")
        self.assertEqual(summary.levels_completed, 1)
        self.assertEqual(summary.total_steps, 4)
        self.assertEqual(summary.verification_failures, 0)
        self.assertFalse(summary.is_real_model)


class TestAbsentModelHandling(unittest.TestCase):
    """TEST_ONLY: Tests that absent / unreachable models report errors cleanly without crashing."""

    def test_unreachable_openai_endpoint_reports_error(self):
        # Non-existent local port
        offline_runner = OpenAICompatibleModelRunner(
            base_url="http://127.0.0.1:54321/v1",
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            timeout_seconds=0.5
        )
        out = offline_runner.generate("Test prompt")
        self.assertEqual(out.text, "")
        self.assertIsNotNone(out.raw_response)
        self.assertIn("error", out.raw_response)

    def test_absent_huggingface_model_raises_cleanly(self):
        from sam_ai.substrate.models import HuggingFaceModelRunner
        # Test non-existent local directory path to verify clean error handling
        runner = HuggingFaceModelRunner(
            model_name_or_path="./non_existent_dir_12345",
            device_map="cpu",
            local_files_only=True,
        )
        with self.assertRaises(Exception):
            runner.load_model()


if __name__ == "__main__":
    unittest.main()

