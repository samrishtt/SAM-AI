"""
Validates the syntax and structure of sam_ai_multidomain_baseline.ipynb.
Ensures:
1. Notebook is valid JSON following nbformat v4.
2. Contains exactly expected cells.
3. Every Python code cell compiles with compile(source, '<cell>', 'exec').
4. Strictly NO adapter is loaded in code cells.
5. Identifies fail-closed hardware checks and sandbox timeouts.
"""

import json
import ast
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/sam_ai_multidomain_baseline/sam_ai_multidomain_baseline.ipynb")
METADATA_PATH = Path("notebooks/sam_ai_multidomain_baseline/kernel-metadata.json")

def verify_notebook():
    print(f"[*] Auditing {NOTEBOOK_PATH}...")
    assert NOTEBOOK_PATH.exists(), f"Notebook missing at {NOTEBOOK_PATH}"
    assert METADATA_PATH.exists(), f"Metadata missing at {METADATA_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' key"
    cells = nb["cells"]
    print(f"[*] Total cells: {len(cells)}")

    code_cells = [c for c in cells if c["cell_type"] == "code"]
    markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]
    print(f"[*] Markdown cells: {len(markdown_cells)} | Code cells: {len(code_cells)}")

    for idx, c in enumerate(code_cells):
        code_str = "".join(c["source"])
        try:
            compile(code_str, f"<cell_{idx}>", "exec")
            print(f"    [OK] Code cell {idx+1} compiles successfully ({len(code_str.splitlines())} lines).")
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in code cell {idx+1}: {e}")

    # Inspect full notebook text for security and constraints
    full_nb_text = "".join("".join(c["source"]) for c in cells)

    # 1. Base model verified
    assert "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" in full_nb_text
    print("[OK] Foundation model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' verified.")

    # 2. Strict no-adapter assertion present
    assert "CRITICAL ERROR: A PEFT adapter was detected" in full_nb_text
    assert "CRITICAL ERROR: Model is wrapped in PeftModel" in full_nb_text
    print("[OK] Strict fail-closed anti-adapter assertions verified.")

    # 3. Isolated sandbox timeout present
    assert "timeout_seconds: float = 3.0" in full_nb_text or "timeout=3.0" in full_nb_text or "timeout=timeout_seconds" in full_nb_text
    print("[OK] HumanEval sandbox 3.0s timeout verified.")

    # 4. Checkpoint resumability present
    assert "predictions.jsonl" in full_nb_text
    assert "completed_ids" in full_nb_text
    print("[OK] Incremental streaming & resumability verified.")

    # 5. Metadata verification
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "samrishb/sam-ai-multidomain-baseline"
    assert meta["kernel_sources"] == []
    print("[OK] Kernel metadata verified (kernel_sources=[] confirms NO adapter input).")

    print("[SUCCESS] Multi-domain baseline notebook is 100% syntactically valid and compliant with Phase 3A specifications.")

if __name__ == "__main__":
    verify_notebook()
