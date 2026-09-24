"""Unit tests for AST-verified Python sandbox and ToolRegistry."""

import pytest
from core.execution.ast_sandbox import PythonASTSandbox, ExecutionResult
from core.execution.tool_registry import ToolRegistry


def test_sandbox_safe_computation():
    sandbox = PythonASTSandbox()
    code = """
import numpy as np
a = np.array([1, 2, 3, 4])
result = int(np.sum(a))
print(f"Computed sum: {result}")
"""
    res: ExecutionResult = sandbox.execute(code)
    assert res.success
    assert res.return_value == 10
    assert "Computed sum: 10" in res.output
    assert res.error is None


def test_sandbox_blocks_os_import():
    sandbox = PythonASTSandbox()
    code = """
import os
os.system("echo hacked")
"""
    res: ExecutionResult = sandbox.execute(code)
    assert not res.success
    assert res.ast_violation is not None
    assert "Blocked import of unauthorized module: 'os'" in res.ast_violation


def test_sandbox_blocks_dunder_traversal():
    sandbox = PythonASTSandbox()
    code = """
x = ().__class__.__bases__[0].__subclasses__()
"""
    res: ExecutionResult = sandbox.execute(code)
    assert not res.success
    assert res.ast_violation is not None
    assert "Blocked access to private dunder attribute" in res.ast_violation


def test_tool_registry_grid_transform():
    registry = ToolRegistry()
    grid = [[1, 2], [3, 4]]
    res = registry.execute_tool("grid_transform", grid=grid, operation="flip_horizontal")
    assert res["success"]
    assert res["result"] == [[2, 1], [4, 3]]
