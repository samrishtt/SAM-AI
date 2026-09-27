"""Hierarchical Repository Mapper & AST Symbol Indexer for 100k+ Line Codebases.

Allows SAM-AI to navigate massive repositories (e.g., Django, SymPy, Astropy)
in milliseconds without overflowing LLM context windows:
1. Extracts AST symbol tables (classes, functions, methods, docstrings, signatures).
2. Builds inverted keyword/symbol index for sub-millisecond retrieval.
3. Resolves call-graphs and localized context windows surrounding candidate bug sites.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class SymbolEntry:
    """Represents a code symbol (class, function, or method) extracted via AST."""
    name: str
    kind: str  # "class", "function", "method"
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""
    signature: str = ""
    parent_class: Optional[str] = None
    called_functions: Set[str] = field(default_factory=set)


@dataclass
class RepoIndex:
    """Structured index of an entire repository."""
    root_dir: str
    total_files_indexed: int
    total_symbols_indexed: int
    symbols: Dict[str, List[SymbolEntry]] = field(default_factory=dict)
    file_map: Dict[str, List[SymbolEntry]] = field(default_factory=dict)
    inverted_index: Dict[str, Set[str]] = field(default_factory=dict)  # token -> set of symbol names


class HierarchicalRepoMapper:
    """Fast, zero-GPU AST symbol indexer and repository navigator."""

    EXCLUDED_DIRS = {
        ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
        "build", "dist", "node_modules", ".tox", "egg-info", "htmlcov"
    }

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).resolve()

    def index_repository(self, max_files: int = 2000) -> RepoIndex:
        """Indexes all Python files in the repository using native AST."""
        symbols: Dict[str, List[SymbolEntry]] = {}
        file_map: Dict[str, List[SymbolEntry]] = {}
        inverted_index: Dict[str, Set[str]] = {}

        total_files = 0
        total_symbols = 0

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS]

            for file in files:
                if not file.endswith(".py"):
                    continue

                total_files += 1
                if total_files > max_files:
                    break

                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(self.root_dir)).replace("\\", "/")

                file_symbols = self._parse_file(full_path, rel_path)
                if file_symbols:
                    file_map[rel_path] = file_symbols
                    for sym in file_symbols:
                        total_symbols += 1
                        symbols.setdefault(sym.name, []).append(sym)

                        # Tokenize symbol name and docstring for inverted index
                        tokens = self._tokenize(sym.name + " " + sym.docstring)
                        for token in tokens:
                            inverted_index.setdefault(token, set()).add(sym.name)

        return RepoIndex(
            root_dir=str(self.root_dir),
            total_files_indexed=total_files,
            total_symbols_indexed=total_symbols,
            symbols=symbols,
            file_map=file_map,
            inverted_index=inverted_index,
        )

    def _parse_file(self, full_path: Path, rel_path: str) -> List[SymbolEntry]:
        """Parses a single Python file into SymbolEntries via AST."""
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(full_path))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            return []

        symbols: List[SymbolEntry] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                symbols.append(SymbolEntry(
                    name=node.name,
                    kind="class",
                    file_path=rel_path,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    docstring=doc,
                    signature=f"class {node.name}"
                ))
                # Methods inside class
                for sub in ast.iter_child_nodes(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_doc = ast.get_docstring(sub) or ""
                        args = [a.arg for a in sub.args.args]
                        sig = f"def {sub.name}({', '.join(args)})"
                        called = self._extract_called_functions(sub)
                        symbols.append(SymbolEntry(
                            name=f"{node.name}.{sub.name}",
                            kind="method",
                            file_path=rel_path,
                            line_start=sub.lineno,
                            line_end=getattr(sub, "end_lineno", sub.lineno),
                            docstring=sub_doc,
                            signature=sig,
                            parent_class=node.name,
                            called_functions=called,
                        ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                args = [a.arg for a in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"
                called = self._extract_called_functions(node)
                symbols.append(SymbolEntry(
                    name=node.name,
                    kind="function",
                    file_path=rel_path,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    docstring=doc,
                    signature=sig,
                    called_functions=called,
                ))

        return symbols

    def _extract_called_functions(self, func_node: ast.AST) -> Set[str]:
        """Extracts function/method call names within a function body."""
        calls = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return calls

    def _tokenize(self, text: str) -> Set[str]:
        """Splits camelCase and snake_case into lower-case search tokens."""
        tokens = set()
        # Split snake_case and non-alphanumeric
        words = re.split(r"[^a-zA-Z0-9_]+", text)
        for w in words:
            # Split camelCase
            sub_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", w).split()
            for sw in sub_words:
                cleaned = sw.strip().lower()
                if len(cleaned) > 2:
                    tokens.add(cleaned)
        return tokens

    def search_symbols(self, index: RepoIndex, query: str, top_k: int = 5) -> List[SymbolEntry]:
        """Searches symbols matching a query string (e.g. from an issue description)."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        # Exact name match gets highest score
        for sym_name in index.symbols:
            if sym_name.lower() in query.lower():
                scores[sym_name] = scores.get(sym_name, 0.0) + 10.0

        # Token overlap matching
        for token in query_tokens:
            if token in index.inverted_index:
                for sym_name in index.inverted_index[token]:
                    scores[sym_name] = scores.get(sym_name, 0.0) + 1.0

        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for sym_name, _ in sorted_symbols:
            for entry in index.symbols.get(sym_name, []):
                results.append(entry)
        return results[:top_k]

    def get_localized_context(self, symbol: SymbolEntry, context_padding: int = 20) -> str:
        """Retrieves exact lines of code surrounding a symbol for LLM prompt insertion."""
        full_path = self.root_dir / symbol.file_path
        if not full_path.exists():
            return f"# File not found: {symbol.file_path}"

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        start = max(1, symbol.line_start - context_padding)
        end = min(len(lines), symbol.line_end + context_padding)

        snippet_lines = []
        for i in range(start - 1, end):
            line_no = i + 1
            marker = "--> " if symbol.line_start <= line_no <= symbol.line_end else "    "
            snippet_lines.append(f"{marker}{line_no:4d} | {lines[i].rstrip()}")

        header = f"# === Localized Context: {symbol.file_path} ({symbol.kind}: {symbol.name}) ==="
        return header + "\n" + "\n".join(snippet_lines)
