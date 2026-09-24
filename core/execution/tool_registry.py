"""Tool Registry and Action Dispatcher.

Provides an extensible, standardized tool registration interface enabling
autonomous agency, sandbox execution, and domain-specific operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json

from core.execution.ast_sandbox import PythonASTSandbox, ExecutionResult


@dataclass
class ToolParameter:
    name: str
    param_type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable[..., Any]


class ToolRegistry:
    """Manages available tools and dispatches calls with argument validation."""

    def __init__(self, sandbox: Optional[PythonASTSandbox] = None):
        self.sandbox = sandbox or PythonASTSandbox()
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name.lower()] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name.lower())

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Dispatches execution to registered tool handler."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' not found."}

        try:
            result = tool.handler(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}

    def _register_default_tools(self) -> None:
        # Tool: python_exec
        def _exec_python(code: str, initial_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            res: ExecutionResult = self.sandbox.execute(code, initial_vars=initial_vars)
            return {
                "success": res.success,
                "output": res.output,
                "return_value": res.return_value,
                "execution_time_ms": res.execution_time_ms,
                "error": res.error,
                "ast_violation": res.ast_violation,
            }

        self.register_tool(
            Tool(
                name="python_exec",
                description="Executes safe Python code inside the AST sandbox. Assign return values to variable 'result'.",
                parameters=[
                    ToolParameter(name="code", param_type="str", description="Python code block to execute."),
                ],
                handler=_exec_python,
            )
        )

        # Tool: grid_transform
        def _grid_transform(grid: List[List[int]], operation: str, **kwargs) -> List[List[int]]:
            import numpy as np
            arr = np.array(grid)
            op = operation.lower()
            if op == "rot90":
                arr = np.rot90(arr, k=kwargs.get("k", -1))
            elif op == "rot180":
                arr = np.rot90(arr, k=2)
            elif op == "rot270":
                arr = np.rot90(arr, k=1)
            elif op == "flip_horizontal":
                arr = np.fliplr(arr)
            elif op == "flip_vertical":
                arr = np.flipud(arr)
            elif op == "recolor":
                from_c = kwargs.get("from_color")
                to_c = kwargs.get("to_color")
                if from_c is not None and to_c is not None:
                    arr[arr == from_c] = to_c
            return arr.tolist()

        self.register_tool(
            Tool(
                name="grid_transform",
                description="Applies spatial geometric operations on 2D discrete grids (rotation, reflection, recolor).",
                parameters=[
                    ToolParameter(name="grid", param_type="list[list[int]]", description="2D grid array."),
                    ToolParameter(name="operation", param_type="str", description="rot90, rot180, flip_horizontal, flip_vertical, recolor."),
                ],
                handler=_grid_transform,
            )
        )
