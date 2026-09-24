"""Unit tests for Claude Code / Codex engine (RepoMap, CodeEditor, REPL)."""

import pytest
import os
from core.coding.repo_map import RepoMap
from core.coding.code_editor import CodeEditor, EditResult
from core.chat.repl import InteractiveAgiREPL


def test_repo_map_indexing(tmp_path):
    # Create sample Python file
    sample_code = """
class CognitiveAgent:
    def plan(self, goal):
        pass

def top_level_helper(x, y):
    return x + y
"""
    f_path = tmp_path / "agent.py"
    f_path.write_text(sample_code, encoding="utf-8")

    repo_map = RepoMap(str(tmp_path))
    index = repo_map.scan_repository()

    assert "agent.py" in index
    assert any(c.name == "CognitiveAgent" for c in index["agent.py"].classes)
    assert any("top_level_helper" in fn.name for fn in index["agent.py"].functions)
    rendered = repo_map.render_map()
    assert "agent.py" in rendered


def test_code_editor_atomic_replace(tmp_path):
    editor = CodeEditor(str(tmp_path))
    file_path = "module.py"
    original = "def solve():\n    return 41\n"
    create_res = editor.write_new_file(file_path, original)
    assert create_res.success

    # View file
    view_res = editor.view_file(file_path)
    assert view_res["success"]
    assert "return 41" in view_res["content"]

    # Search & Replace
    edit_res: EditResult = editor.replace_content(
        path=file_path,
        target_content="return 41",
        replacement_content="return 42",
    )
    assert edit_res.success
    assert "--- a/module.py" in edit_res.diff
    assert "+    return 42" in edit_res.diff

    # Intercept invalid syntax
    bad_edit = editor.replace_content(
        path=file_path,
        target_content="return 42",
        replacement_content="return (((invalid",
    )
    assert not bad_edit.success
    assert "SyntaxError" in str(bad_edit.error)


def test_interactive_repl_commands(tmp_path):
    repl = InteractiveAgiREPL(str(tmp_path))

    # Test /mcp create
    mcp_out = repl.process_command("/mcp create AnalyticsServer")
    assert "[MCP GENERATED]" in mcp_out
    assert os.path.exists(os.path.join(str(tmp_path), "mcp_analyticsserver_server.py"))

    # Test /memory
    mem_out = repl.process_command("/memory")
    assert "[KNOWLEDGE GRAPH]" in mem_out
