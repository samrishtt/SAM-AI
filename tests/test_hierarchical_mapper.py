import tempfile
from pathlib import Path
from sam_ai.agents.hierarchical_mapper import HierarchicalRepoMapper, SymbolEntry


def test_hierarchical_indexing_and_search():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_root = Path(temp_dir)

        # Create simulated repo structure
        sub_dir = repo_root / "package" / "core"
        sub_dir.mkdir(parents=True, exist_ok=True)

        module_code = '''"""Core computational module."""

def calculate_determinant(matrix):
    """Computes determinant of square matrix."""
    return 42

class MatrixOptimizer:
    """Optimizes linear systems."""

    def __init__(self, tolerance=1e-5):
        self.tol = tolerance

    def solve_system(self, a, b):
        """Solves ax = b using decomposition."""
        det = calculate_determinant(a)
        return [det]
'''
        (sub_dir / "solver.py").write_text(module_code, encoding="utf-8")

        mapper = HierarchicalRepoMapper(repo_root)
        index = mapper.index_repository()

        assert index.total_files_indexed == 1
        assert index.total_symbols_indexed == 4  # function, class, 2 methods

        # Verify exact symbol lookup
        assert "calculate_determinant" in index.symbols
        sym = index.symbols["calculate_determinant"][0]
        assert sym.kind == "function"
        assert sym.file_path == "package/core/solver.py"
        assert sym.docstring == "Computes determinant of square matrix."

        # Verify method indexing with class prefix
        assert "MatrixOptimizer.solve_system" in index.symbols
        method_sym = index.symbols["MatrixOptimizer.solve_system"][0]
        assert method_sym.kind == "method"
        assert "calculate_determinant" in method_sym.called_functions

        # Test natural language search from issue description
        results = mapper.search_symbols(index, "Issue in MatrixOptimizer while computing determinant", top_k=2)
        assert len(results) > 0
        names = [r.name for r in results]
        assert "MatrixOptimizer" in names or "calculate_determinant" in names

        # Test localized context generation
        context = mapper.get_localized_context(method_sym, context_padding=5)
        assert "=== Localized Context: package/core/solver.py" in context
        assert "--> " in context  # marks the target method
        assert "def solve_system" in context
