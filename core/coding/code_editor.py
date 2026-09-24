"""Precision Code Editor & Diff Engine (Claude Code / Codex Engine).

Provides atomic file viewing, contiguous search-and-replace, unified diff
generation, and AST pre-commit syntax verification.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
import difflib
import os
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EditResult:
    success: bool
    path: str
    diff: str
    error: Optional[str] = None
    lines_changed: int = 0


class CodeEditor:
    """Atomic code editing and diff inspection engine."""

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.workspace_root, path))

    def view_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reads file slice with 1-indexed line numbers."""
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"success": False, "error": f"File '{path}' does not exist."}

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            s = max(1, start_line or 1)
            e = min(total_lines, end_line or total_lines)

            slice_lines = lines[s - 1 : e]
            formatted = "".join(f"{s + idx:4d} | {line}" for idx, line in enumerate(slice_lines))

            return {
                "success": True,
                "path": full_path,
                "total_lines": total_lines,
                "start_line": s,
                "end_line": e,
                "content": formatted,
            }
        except Exception as ex:
            return {"success": False, "error": f"{type(ex).__name__}: {str(ex)}"}

    def replace_content(
        self,
        path: str,
        target_content: str,
        replacement_content: str,
        allow_multiple: bool = False,
    ) -> EditResult:
        """Executes exact contiguous string replacement with unified diff generation."""
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return EditResult(success=False, path=path, diff="", error=f"File '{path}' does not exist.")

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                original_text = f.read()

            count = original_text.count(target_content)
            if count == 0:
                return EditResult(
                    success=False,
                    path=path,
                    diff="",
                    error=f"Target content not found in '{path}'. Ensure exact match including whitespace.",
                )
            if count > 1 and not allow_multiple:
                return EditResult(
                    success=False,
                    path=path,
                    diff="",
                    error=f"Target content appears {count} times in '{path}'. Set allow_multiple=True or specify a unique chunk.",
                )

            new_text = original_text.replace(target_content, replacement_content, 1 if not allow_multiple else -1)

            # Pre-commit AST verification for Python files
            if full_path.endswith(".py"):
                try:
                    ast.parse(new_text)
                except SyntaxError as syn_err:
                    return EditResult(
                        success=False,
                        path=path,
                        diff="",
                        error=f"SyntaxError in replacement code: {syn_err}",
                    )

            # Generate unified diff
            diff = "".join(
                difflib.unified_diff(
                    original_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )

            # Write updated content
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_text)

            lines_changed = abs(len(new_text.splitlines()) - len(original_text.splitlines()))
            return EditResult(success=True, path=path, diff=diff, lines_changed=lines_changed)

        except Exception as e:
            return EditResult(success=False, path=path, diff="", error=f"{type(e).__name__}: {str(e)}")

    def write_new_file(self, path: str, content: str, overwrite: bool = False) -> EditResult:
        """Creates a new file with AST pre-validation."""
        full_path = self._resolve_path(path)
        if os.path.exists(full_path) and not overwrite:
            return EditResult(success=False, path=path, diff="", error=f"File '{path}' already exists.")

        if full_path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as syn_err:
                return EditResult(success=False, path=path, diff="", error=f"SyntaxError in code: {syn_err}")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return EditResult(
            success=True,
            path=path,
            diff=f"+++ {path} (new file, {len(content.splitlines())} lines)",
            lines_changed=len(content.splitlines()),
        )
