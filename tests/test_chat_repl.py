"""Tests for InteractiveAgiREPL conversational intelligence, task execution, and tools."""

import os
import pytest
from core.chat.repl import InteractiveAgiREPL


@pytest.fixture
def repl(tmp_path):
    return InteractiveAgiREPL(str(tmp_path))


def test_conversational_greetings_and_status(repl):
    # Greeting
    res_hi = repl.process_command("hello!")
    assert "Hello! I am **Micro-AGI**" in res_hi

    # Status
    res_status = repl.process_command("how r u?")
    assert "running at peak cognitive efficiency" in res_status
    assert "Neocortical Knowledge Graph" in res_status

    # Identity
    res_id = repl.process_command("who are you?")
    assert "Micro-AGI" in res_id
    assert "Dual-Process Cognition" in res_id


def test_file_creation_task(repl, tmp_path):
    # Create Python file
    cmd = "create file math_utils.py with content def add(a, b):\n    return a + b\nprint(add(10, 20))\n"
    res = repl.process_command(cmd)
    assert "File Created Successfully: `math_utils.py`" in res
    assert os.path.exists(tmp_path / "math_utils.py")
    assert "AST Sandbox Verification" in res

    # View created file
    view_res = repl.process_command("view file math_utils.py")
    assert "File Contents: `math_utils.py`" in view_res
    assert "def add(a, b):" in view_res


def test_service_integrations(repl, tmp_path):
    # Instagram
    res_insta = repl.process_command("could u connect with my instagram?")
    assert "Connecting with Instagram" in res_insta
    assert "Meta Graph API" in res_insta
    assert os.path.exists(tmp_path / "instagram_connector.py")

    # Discord
    res_discord = repl.process_command("connect to discord")
    assert "Connecting with Discord" in res_discord
    assert os.path.exists(tmp_path / "discord_connector.py")


def test_mcp_integration_guidance(repl):
    res_mcp = repl.process_command("could u connect to any mcp?")
    assert "Connecting to Model Context Protocol (MCP) Servers" in res_mcp
    assert "MCPClient" in res_mcp


def test_math_calculation(repl):
    res_math = repl.process_command("calculate 125 * 8 - 50")
    assert "Mathematical Computation" in res_math
    assert "950" in res_math


def test_algorithm_synthesis(repl):
    res_algo = repl.process_command("write a python quicksort")
    assert "Verified Code Synthesis: Quicksort" in res_algo
    assert "AST Sandbox Output" in res_algo


def test_open_domain_knowledge(repl):
    res_photo = repl.process_command("explain photosynthesis")
    assert "Photosynthesis" in res_photo
    assert "RuBisCO" in res_photo or "Calvin Cycle" in res_photo

    res_general = repl.process_command("how does an operating system kernel work?")
    assert "Analytical Assessment" in res_general
    # Assert no robotic simulation boilerplate
    assert "counterfactual world states" not in res_general
