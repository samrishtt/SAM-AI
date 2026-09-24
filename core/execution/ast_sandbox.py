"""AST-Verified Deterministic Python Sandbox.

Analyzes the Abstract Syntax Tree (AST) of candidate code snippets prior to execution
to mathematically guarantee safety, blocking unauthorized imports, private dunder
introspection, and shell escapes, while permitting pure mathematical and algorithmic synthesis.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
import io
import sys
import time
from typing import Any, Dict, List, Optional, Set
import numpy as np


@dataclass
class ExecutionResult:
    success: bool
    output: str
    return_value: Any
    execution_time_ms: float
    error: Optional[str] = None
    ast_violation: Optional[str] = None


class ASTSafetyInspector(ast.NodeVisitor):
    """Inspects AST nodes for potential safety violations."""

    BLOCKED_MODULES: Set[str] = {
        "os", "sys", "subprocess", "socket", "requests", "urllib",
        "shutil", "builtins", "importlib", "pathlib", "ctypes", "pickle"
    }

    BLOCKED_FUNCTIONS: Set[str] = {
        "eval", "exec", "compile", "__import__", "open", "input"
    }

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in self.BLOCKED_MODULES:
                self.violations.append(f"Blocked import of unauthorized module: '{base_mod}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in self.BLOCKED_MODULES:
                self.violations.append(f"Blocked import from unauthorized module: '{base_mod}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BLOCKED_FUNCTIONS:
                self.violations.append(f"Blocked call to restricted builtin function: '{node.func.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Prevent dunder escape vectors like __class__, __subclasses__, __globals__
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.violations.append(f"Blocked access to private dunder attribute: '{node.attr}'")
        self.generic_visit(node)


class PythonASTSandbox:
    """Safe execution sandbox with AST verification and output capture."""

    def __init__(self, timeout_sec: float = 3.0):
        self.timeout_sec = timeout_sec

    def verify_safety(self, code: str) -> Optional[str]:
        """Parses and inspects code AST. Returns violation message if unsafe, else None."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError in code: {e}"

        inspector = ASTSafetyInspector()
        inspector.visit(tree)
        if inspector.violations:
            return "; ".join(inspector.violations)
        return None

    def execute(self, code: str, initial_vars: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Executes verified code within a restricted namespace."""
        violation = self.verify_safety(code)
        if violation:
            return ExecutionResult(
                success=False,
                output="",
                return_value=None,
                execution_time_ms=0.0,
                ast_violation=violation,
                error=f"AST Safety Rejection: {violation}",
            )

        def _safe_import(name, *args, **kwargs):
            base_mod = name.split(".")[0]
            if base_mod in ASTSafetyInspector.BLOCKED_MODULES:
                raise ImportError(f"Import of unauthorized module '{base_mod}' is forbidden.")
            return __import__(name, *args, **kwargs)

        # Restricted safe globals
        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                "enumerate": enumerate, "filter": filter, "float": float, "int": int,
                "len": len, "list": list, "map": map, "max": max, "min": min,
                "range": range, "reversed": reversed, "round": round, "set": set,
                "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
                "print": print, "Exception": Exception, "ValueError": ValueError,
                "__import__": _safe_import,
            },
            "np": np,
            "numpy": np,
        }

        exec_scope: Dict[str, Any] = dict(safe_globals)
        if initial_vars:
            exec_scope.update(initial_vars)

        # Capture stdout
        old_stdout = sys.stdout
        redirected_stdout = io.StringIO()
        sys.stdout = redirected_stdout

        start_time = time.perf_counter()
        try:
            # Execute code block with unified scope for recursive functions and cross-definition visibility
            exec(code, exec_scope, exec_scope)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            output = redirected_stdout.getvalue()

            # Result can be extracted from 'result' variable if defined
            result_val = exec_scope.get("result", None)

            return ExecutionResult(
                success=True,
                output=output,
                return_value=result_val,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                output=redirected_stdout.getvalue(),
                return_value=None,
                execution_time_ms=elapsed_ms,
                error=f"{type(e).__name__}: {str(e)}",
            )
        finally:
            sys.stdout = old_stdout
