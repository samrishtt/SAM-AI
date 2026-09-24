"""Repository Map & Symbol Indexer (Claude Code / Codex Engine).

Extracts abstract syntax tree (AST) symbol definitions, function signatures,
class hierarchies, and cross-file import dependencies across a codebase,
providing dense, token-efficient global context.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Set


@dataclass
class SymbolDefinition:
    name: str
    kind: str  # "class", "function", "method"
    line_number: int
    docstring: Optional[str] = None
    args: List[str] = field(default_factory=list)


@dataclass
class FileIndex:
    relative_path: str
    classes: List[SymbolDefinition] = field(default_factory=list)
    functions: List[SymbolDefinition] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    loc: int = 0


class RepoMapExtractor(ast.NodeVisitor):
    """Inspects AST nodes of a Python file to extract symbols."""

    def __init__(self):
        self.classes: List[SymbolDefinition] = []
        self.functions: List[SymbolDefinition] = []
        self.imports: List[str] = []
        self._current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        doc = ast.get_docstring(node)
        first_line = doc.split("\n")[0] if doc else None
        sym = SymbolDefinition(name=node.name, kind="class", line_number=node.lineno, docstring=first_line)
        self.classes.append(sym)

        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        doc = ast.get_docstring(node)
        first_line = doc.split("\n")[0] if doc else None
        args = [arg.arg for arg in node.args.args]
        kind = "method" if self._current_class else "function"
        full_name = f"{self._current_class}.{node.name}" if self._current_class else node.name
        sym = SymbolDefinition(name=full_name, kind=kind, line_number=node.lineno, docstring=first_line, args=args)
        self.functions.append(sym)
        self.generic_visit(node)


class RepoMap:
    """Indexes and renders concise structural maps of a software repository."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.index: Dict[str, FileIndex] = {}

    def scan_repository(self, max_files: int = 150) -> Dict[str, FileIndex]:
        """Scans and indexes all Python source files in the repository."""
        self.index.clear()
        file_count = 0

        for root, dirs, files in os.walk(self.root_dir):
            # Ignore hidden or build dirs
            dirs[:] = [d for d in dirs if not d.startswith((".", "__")) and d not in {"venv", "node_modules", "dist", "build"}]

            for f in files:
                if f.endswith(".py"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.root_dir).replace("\\", "/")
                    file_idx = self._index_file(full_path, rel_path)
                    if file_idx:
                        self.index[rel_path] = file_idx
                        file_count += 1
                        if file_count >= max_files:
                            break
            if file_count >= max_files:
                break

        return self.index

    def _index_file(self, full_path: str, rel_path: str) -> Optional[FileIndex]:
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            tree = ast.parse(content)
            visitor = RepoMapExtractor()
            visitor.visit(tree)

            loc = len(content.splitlines())
            return FileIndex(
                relative_path=rel_path,
                classes=visitor.classes,
                functions=visitor.functions,
                imports=list(set(visitor.imports)),
                loc=loc,
            )
        except Exception:
            return None

    def render_map(self) -> str:
        """Renders compact structural representation of the indexed repository."""
        lines = [f"# Repository Map: {os.path.basename(self.root_dir)} ({len(self.index)} files)", ""]

        for rel_path, f_idx in sorted(self.index.items()):
            lines.append(f"File: {rel_path} ({f_idx.loc} LOC)")
            if f_idx.classes:
                c_str = ", ".join(c.name for c in f_idx.classes[:5])
                lines.append(f"  Classes: {c_str}")
            if f_idx.functions:
                top_fns = [f"{fn.name}({', '.join(fn.args[:2])})" for fn in f_idx.functions[:6]]
                lines.append(f"  Functions: {', '.join(top_fns)}")
            lines.append("")

        return "\n".join(lines).strip()
