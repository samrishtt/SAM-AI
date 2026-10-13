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
    """Verify that both notebook and scripts/run_real_model_smoke_test.py call discover_inputs() and use standard keys."""
    nb_path = Path("notebooks/sam_ai_real_model_smoke_test/sam_ai_real_model_smoke_test.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Exclude cell 1 (which embeds kaggle_path_discovery.py definition)
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    nb_remaining_code = "\n".join(
        "".join(c["source"]) for c in code_cells[1:]
    )
    assert "discover_inputs()" in nb_remaining_code, "Notebook does not call discover_inputs()"
    assert "paths['sam_ai_parent']" in nb_remaining_code or 'paths["sam_ai_parent"]' in nb_remaining_code
    assert "paths['environment_files']" in nb_remaining_code or 'paths["environment_files"]' in nb_remaining_code
    assert "paths['base_model_dir']" in nb_remaining_code or 'paths["base_model_dir"]' in nb_remaining_code
    assert "/kaggle/input" not in nb_remaining_code, "Notebook contains hardcoded /kaggle/input outside Cell 1"

    script_path = Path("scripts/run_real_model_smoke_test.py")
    script_text = script_path.read_text(encoding="utf-8")
    assert "discover_inputs(" in script_text, (
        "scripts/run_real_model_smoke_test.py does not call discover_inputs()"
    )
    assert 'discovered_paths["sam_ai_parent"]' in script_text or "discovered_paths['sam_ai_parent']" in script_text or 'paths["sam_ai_parent"]' in script_text
    assert 'discovered_paths.get("environment_files")' in script_text or 'paths["environment_files"]' in script_text
    assert 'discovered_paths.get("base_model_dir")' in script_text or 'paths["base_model_dir"]' in script_text
    assert '"/kaggle/input' not in script_text and "'/kaggle/input" not in script_text, (
        "scripts/run_real_model_smoke_test.py contains hardcoded /kaggle/input string literals"
    )


def test_discover_inputs_fake_tree(tmp_path):
    """Verify discover_inputs against a simulated /kaggle/input directory tree."""
    fake_input = tmp_path / "fake_input"

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
    cfg = {
        "model_type": "qwen2",
        "num_hidden_layers": 48,
        "quantization_config": {"load_in_4bit": True, "quant_method": "bitsandbytes"}
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "model.safetensors").write_text("DUMMY_WEIGHTS", encoding="utf-8")

    res = discover_inputs(root=str(fake_input), verbose=False)

    assert Path(res["sam_ai_parent"]).resolve() == pkg_dir.parent.resolve()
    assert Path(res["base_model_dir"]).resolve() == model_dir.resolve()
    assert Path(res["environment_files"]).resolve() == env_dir.parent.resolve()
    assert res["base_model_config"]["num_hidden_layers"] == 48


def test_discover_inputs_missing_raises_and_prints_layout(tmp_path):
    """Verify that missing inputs raise RuntimeError containing the formatted layout."""
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()

    with pytest.raises(RuntimeError) as exc_info:
        discover_inputs(root=str(empty_input), verbose=False)

    err = str(exc_info.value)
    assert "Missing inputs" in err
    assert "Layout seen:" in err
