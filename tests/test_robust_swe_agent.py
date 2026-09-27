"""Unit tests for Robust SWE Agent Scaffold."""

import tempfile
import pytest
from pathlib import Path
from sam_ai.agents.robust_swe_agent import RobustSWEAgent


def test_dangerous_command_blocking():
    """Verifies that catastrophic commands are blocked by the safety layer."""
    agent = RobustSWEAgent(repo_dir=".")
    dangerous_cmd = "rm -rf /"
    res = agent.execute_safe_command(dangerous_cmd)
    assert res.exit_code == -1
    assert "SecurityError" in res.stderr


def test_precision_edit_ast_guard():
    """Verifies that syntax-breaking edits are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target_file = tmp_path / "app.py"
        target_file.write_text("def valid_function():\n    return 42\n", encoding="utf-8")

        agent = RobustSWEAgent(repo_dir=str(tmp_path))

        # Attempt to insert syntax error
        bad_edit = agent.apply_precision_edit(
            relative_path="app.py",
            old_content="return 42",
            new_content="return (invalid syntax here %%%",
        )

        assert not bad_edit["success"]
        assert "SyntaxError" in bad_edit["error"]
        # Original content must be intact
        assert target_file.read_text(encoding="utf-8") == "def valid_function():\n    return 42\n"


def test_successful_precision_edit():
    """Verifies clean localized replacement."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target_file = tmp_path / "math_lib.py"
        target_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        agent = RobustSWEAgent(repo_dir=str(tmp_path))
        res = agent.apply_precision_edit(
            relative_path="math_lib.py",
            old_content="return a - b",
            new_content="return a + b",
        )

        assert res["success"]
        assert "return a + b" in target_file.read_text(encoding="utf-8")
        assert "math_lib.py" in agent.modified_files


def test_multi_turn_interactive_repair_loop_with_rollback():
    """
    Verifies Pillar 1: Multi-Turn Interactive Repair with error reflection and atomic rollback.
    """
    import subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Setup test repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "sam@ai.dev"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Sam"], cwd=tmp_path, check=True)

        calc_file = tmp_path / "calc.py"
        calc_file.write_text("def compute(x):\n    return x * 2  # Bug: should be x ** 2\n", encoding="utf-8")

        test_file = tmp_path / "test_calc.py"
        test_file.write_text("from calc import compute\nassert compute(3) == 9, 'Expected 9'\n", encoding="utf-8")

        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)

        agent = RobustSWEAgent(repo_dir=str(tmp_path))

        # Generator function that fails on turn 1, but succeeds on turn 2 after receiving error feedback
        attempts = 0
        def mock_generator(issue, error_context):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # Turn 1: Flawed attempt that fails
                return ("calc.py", "return x * 2", "return x + 5")
            else:
                # Turn 2: Correct fix based on error feedback
                return ("calc.py", "return x * 2", "return x ** 2")

        result = agent.run_interactive_repair_loop(
            issue_statement="Fix compute function so compute(3) == 9",
            test_command="python test_calc.py",
            edit_generator=mock_generator,
            max_turns=3,
        )

        assert result["resolved"] is True
        assert result["turns_taken"] == 2
        assert "calc.py" in result["git_diff"]
        assert "+    return x ** 2" in result["git_diff"]

