import json
import tempfile
from pathlib import Path
from sam_ai.training.self_improvement_flywheel import (
    SelfImprovementFlywheel,
    TaskSynthesizer,
    DeterministicTaskVerifier,
    SynthesizedTask,
)


def test_task_synthesis_and_verification():
    synthesizer = TaskSynthesizer(seed=42)

    # 1. Test Math Vieta Challenge
    vieta_task = synthesizer.generate_math_vieta_challenge()
    assert vieta_task.domain == "math"
    assert "<think>" in vieta_task.prompt
    assert DeterministicTaskVerifier.verify(vieta_task) is True

    # 2. Test Math Modular Challenge
    mod_task = synthesizer.generate_math_modular_challenge()
    assert mod_task.domain == "math"
    assert DeterministicTaskVerifier.verify(mod_task) is True

    # 3. Test Coding Challenge
    code_task = synthesizer.generate_code_challenge()
    assert code_task.domain == "code"
    assert DeterministicTaskVerifier.verify(code_task) is True

    # 4. Test ARC Spatial Challenge
    arc_task = synthesizer.generate_arc_challenge()
    assert arc_task.domain == "arc"
    assert DeterministicTaskVerifier.verify(arc_task) is True


def test_verifier_rejects_broken_code():
    bad_task = SynthesizedTask(
        task_id="broken_task",
        domain="math",
        prompt="Solve 1+1",
        ground_truth="2",
        verification_code="assert 1 + 1 == 3", # Flawed ground truth
    )
    assert DeterministicTaskVerifier.verify(bad_task) is False

    syntax_err_task = SynthesizedTask(
        task_id="syntax_err",
        domain="code",
        prompt="Write code",
        ground_truth="error",
        verification_code="def broken(: return 1", # Syntax error
    )
    assert DeterministicTaskVerifier.verify(syntax_err_task) is False


def test_flywheel_dataset_export():
    flywheel = SelfImprovementFlywheel(seed=100)
    with tempfile.TemporaryDirectory() as temp_dir:
        out_file = Path(temp_dir) / "synthetic_grpo_dataset.jsonl"
        exported_count = flywheel.export_grpo_training_dataset(out_file, count=12)

        assert exported_count == 12
        assert out_file.exists()

        records = []
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 12
        domains = {r["domain"] for r in records}
        assert "math" in domains
        assert "code" in domains or "arc" in domains
        for r in records:
            assert "prompt" in r
            assert "ground_truth" in r
            assert "task_id" in r
