"""Unit tests for path discovery parity between scripts and notebook, and fake-tree resolution."""

import json
from pathlib import Path
import pytest

from scripts.kaggle_path_discovery import discover_inputs


def test_first_code_cell_parity():
    """Verify that the smoke test notebook's first code cell exactly matches scripts/kaggle_path_discovery.py."""
    nb_path = Path("notebooks/sam_ai_real_model_smoke_test/sam_ai_real_model_smoke_test.ipynb")
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the first code cell
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 1, "Notebook has no code cells"
    first_code = "".join(code_cells[0]["source"])

    py_path = Path("scripts/kaggle_path_discovery.py")
    expected_text = py_path.read_text(encoding="utf-8")

    assert first_code == expected_text, (
        "Notebook first code cell does not match scripts/kaggle_path_discovery.py"
    )


def test_both_call_discover_inputs():
    """Verify that both the notebook and scripts/run_real_model_smoke_test.py call discover_inputs()."""
    nb_path = Path("notebooks/sam_ai_real_model_smoke_test/sam_ai_real_model_smoke_test.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    assert "discover_inputs()" in nb_code, "Notebook does not call discover_inputs()"

    script_path = Path("scripts/run_real_model_smoke_test.py")
    script_text = script_path.read_text(encoding="utf-8")
    assert "discover_inputs(" in script_text, (
        "scripts/run_real_model_smoke_test.py does not call discover_inputs()"
    )


def test_discover_inputs_fake_tree(tmp_path):
    """Verify discover_inputs against a simulated /kaggle/input directory tree."""
    fake_input = tmp_path / "fake_input"
    fake_working = tmp_path / "fake_working"

    # 1. sam_ai package in dataset
    pkg_dir = fake_input / "datasets" / "samrishb" / "sam-ai-source-bundle" / "sam_ai"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("# sam_ai\n", encoding="utf-8")

    # 2. environment_files in competition dir
    env_dir = (
        fake_input
        / "competitions"
        / "arc-prize-2026-arc-agi-3"
        / "environment_files"
        / "game_alpha"
    )
    env_dir.mkdir(parents=True)
    meta = {"game_id": "game_alpha", "class_name": "GameAlpha"}
    (env_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (env_dir / "gamealpha.py").write_text("# game code\n", encoding="utf-8")

    # 3. base model directory with qwen2 config and safetensors
    model_dir = (
        fake_input
        / "models"
        / "gowillgo"
        / "unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit"
        / "transformers"
        / "v1"
        / "1"
    )
    model_dir.mkdir(parents=True)
    cfg = {"model_type": "qwen2", "vocab_size": 152064}
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "model.safetensors").write_text("DUMMY_WEIGHTS", encoding="utf-8")

    res = discover_inputs(input_dir=fake_input, working_dir=fake_working)

    assert res["sam_ai_parent"] == pkg_dir.parent
    assert res["base_model"] == model_dir
    assert res["environment_files"] == env_dir.parent


def test_discover_inputs_missing_raises_and_prints_layout(tmp_path):
    """Verify that missing inputs raise FileNotFoundError containing the formatted layout."""
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_inputs(input_dir=empty_input, working_dir=tmp_path / "working")

    err = str(exc_info.value)
    assert "Failed to discover required Kaggle input paths" in err
    assert "=== INPUT LAYOUT" in err
