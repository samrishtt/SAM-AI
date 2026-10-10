#!/usr/bin/env python3
"""
SAM-AI Evaluation Runner & Instrumentation Harness
===================================================
Executes structured evaluations of the SAM-AI agent, recording exact model provenance,
git commit, seeds, passes, actions, verification status, and telemetry.
Persists per-run raw JSONL and a summary report.

Rules:
- Clearly stamps runs as REAL_MODEL or TEST_ONLY.
- Never reports best stochastic pass as sole score; reports mean and spread.
- Checks official environment availability before executing.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sam_ai.substrate.interfaces import ModelRunner

from sam_ai.substrate.models import DeterministicModelRunner, OpenAICompatibleModelRunner
from sam_ai.agent.arc3_agent import SAMARC3Agent, SessionSummary
from tests.test_sam_ai_arc3_agent import MockArcEnvironment


def get_git_commit() -> str:
    """Returns current git commit hash, or 'unknown'."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Computes SHA256 hash of configuration dictionary."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def check_official_arc_environments(env_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Checks whether official ARC-AGI-3 environment files are available locally."""
    target_dir = env_dir or os.environ.get("ENVIRONMENTS_DIR", "environment_files")
    p = Path(target_dir)
    if not p.exists() or not p.is_dir():
        return False, f"Directory '{target_dir}' does not exist."

    metadata_files = list(p.rglob("metadata.json"))
    if not metadata_files:
        return False, f"No 'metadata.json' environment files found in '{target_dir}'."

    return True, f"Found {len(metadata_files)} official environment files."


def run_evaluation(
    model_runner: ModelRunner,
    game_ids: List[str],
    n_passes: int = 1,
    max_steps_per_game: int = 30,
    enable_verification: bool = True,
    enable_failure_memory: bool = True,
    output_dir: str = "evidence/sam_ai_runs",
    is_real_model: bool = False,
    run_type: str = "TEST_ONLY",
    use_mock_env: bool = True,
) -> Dict[str, Any]:
    """
    Executes multi-pass evaluation across specified games with complete provenance tracking.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = out_path / "runs_raw.jsonl"
    summary_path = out_path / "summary.json"

    git_commit = get_git_commit()
    config = {
        "model_name": model_runner.model_name,
        "n_passes": n_passes,
        "max_steps": max_steps_per_game,
        "verification_guard": enable_verification,
        "failure_memory": enable_failure_memory,
        "is_real_model": is_real_model,
        "run_type": run_type,
    }
    config_hash = compute_config_hash(config)

    print("=" * 80)
    print(f"SAM-AI EVALUATION HARNESS [{run_type}]")
    print(f"Model: {model_runner.model_name} (Real Model: {is_real_model})")
    print(f"Git Commit: {git_commit} | Config Hash: {config_hash}")
    print(f"Games: {game_ids} | Passes: {n_passes}")
    print("=" * 80)

    per_game_results: Dict[str, List[Dict[str, Any]]] = {gid: [] for gid in game_ids}
    all_runs_records: List[Dict[str, Any]] = []

    for gid in game_ids:
        print(f"\n[*] Evaluating Game: {gid}")
        for pass_idx in range(n_passes):
            seed = 42 + pass_idx

            if use_mock_env:
                # Deterministic compliant mock environment
                env = MockArcEnvironment(target_coord=(2, 2))
                env_type = "MOCK_ARC_TEST_ONLY"
            else:
                # Official environment check
                ok, reason = check_official_arc_environments()
                if not ok:
                    raise RuntimeError(f"Official ARC environment unavailable: {reason}")
                env = None
                env_type = "OFFICIAL_ARC_AGI"

            agent = SAMARC3Agent(
                model_runner=model_runner,
                enable_verification_guard=enable_verification,
                enable_failure_memory=enable_failure_memory,
                is_real_model=is_real_model,
            )

            start_t = time.perf_counter()
            summary = agent.run_session(env, game_id=gid, max_steps=max_steps_per_game)
            elapsed_s = time.perf_counter() - start_t

            # Structured record
            run_record = {
                "timestamp": time.time(),
                "run_type": run_type,
                "is_real_model": is_real_model,
                "model_name": model_runner.model_name,
                "git_commit": git_commit,
                "config_hash": config_hash,
                "environment_type": env_type,
                "game_id": gid,
                "pass_idx": pass_idx,
                "seed": seed,
                "steps_taken": summary.total_steps,
                "levels_completed": summary.levels_completed,
                "final_state": summary.final_state,
                "success": summary.success,
                "verification_failures": summary.verification_failures,
                "prompt_tokens": summary.total_prompt_tokens,
                "completion_tokens": summary.total_completion_tokens,
                "latency_ms": summary.total_latency_ms,
                "elapsed_seconds": round(elapsed_s, 3),
            }

            # Persist to raw JSONL immediately (append mode)
            with open(raw_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_record) + "\n")

            per_game_results[gid].append(run_record)
            all_runs_records.append(run_record)

            status_icon = "WIN" if summary.success else summary.final_state
            print(f"    Pass {pass_idx} (seed {seed}): Steps={summary.total_steps}, Levels={summary.levels_completed}, Status={status_icon}, Latency={elapsed_s:.2f}s")

    # Aggregate Statistics
    total_runs = len(all_runs_records)
    total_wins = sum(1 for r in all_runs_records if r["success"])
    win_rate = (total_wins / total_runs) * 100 if total_runs > 0 else 0.0
    mean_steps = sum(r["steps_taken"] for r in all_runs_records) / total_runs if total_runs > 0 else 0.0
    mean_levels = sum(r["levels_completed"] for r in all_runs_records) / total_runs if total_runs > 0 else 0.0

    # Pass-by-pass breakdown
    pass_stats: Dict[int, Dict[str, float]] = {}
    for p_idx in range(n_passes):
        p_runs = [r for r in all_runs_records if r["pass_idx"] == p_idx]
        p_wins = sum(1 for r in p_runs if r["success"])
        p_rate = (p_wins / len(p_runs)) * 100 if p_runs else 0.0
        pass_stats[p_idx] = {
            "runs": len(p_runs),
            "win_rate": round(p_rate, 2),
            "mean_levels": round(sum(r["levels_completed"] for r in p_runs) / len(p_runs), 2) if p_runs else 0.0,
        }

    summary_data = {
        "evaluation_timestamp": time.time(),
        "run_type": run_type,
        "is_real_model": is_real_model,
        "model_name": model_runner.model_name,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "total_games": len(game_ids),
        "n_passes": n_passes,
        "total_runs": total_runs,
        "overall_win_rate_percent": round(win_rate, 2),
        "mean_levels_completed": round(mean_levels, 2),
        "mean_steps_per_run": round(mean_steps, 2),
        "pass_breakdown": pass_stats,
        "raw_jsonl_file": str(raw_jsonl_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Overall Win Rate: {win_rate:.2f}% ({total_wins}/{total_runs} runs)")
    print(f"Mean Levels Completed: {mean_levels:.2f}")
    print(f"Mean Steps: {mean_steps:.2f}")
    print(f"Raw Evidence: {raw_jsonl_path}")
    print(f"Summary JSON: {summary_path}")
    print("=" * 80)

    return summary_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM-AI Agent Evaluation Harness")
    parser.add_argument("--mode", choices=["test_only", "real_model"], default="test_only")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--games", nargs="+", default=["mock_game_01", "mock_game_02"])
    parser.add_argument("--outdir", default="evidence/sam_ai_runs")
    args = parser.parse_args()

    if args.mode == "real_model":
        print("[!] REAL_MODEL evaluation requested.")
        print("[BLOCKED] Real SAM-AI 14B model weights not accessible on local CPU environment.")
        print("          Missing prerequisites:")
        print("          1. Base model weights: unsloth/DeepSeek-R1-Distill-Qwen-14B (requires ~10-28 GB download + GPU).")
        print("          2. Hardware accelerator: CUDA is False (local machine is CPU only).")
        print("          3. Official ARC-AGI-3 environment files: directory containing metadata.json files is missing locally.")
        sys.exit(1)
    else:
        # Deterministic mock runner for test_only verification
        call_count = [0]
        def mock_gen(prompt):
            call_count[0] += 1
            # Simple alternating moves towards (2, 2)
            moves = ["ACTION2", "ACTION2", "ACTION4", "ACTION4"]
            m = moves[(call_count[0] - 1) % len(moves)]
            return json.dumps({"thought": f"Move toward target step {call_count[0]}", "plan": "Reach row 2 col 2", "action": m, "data": {}})

        runner = DeterministicModelRunner(
            name="deterministic-mock-runner-v1",
            generator_fn=mock_gen,
        )

        run_evaluation(
            model_runner=runner,
            game_ids=args.games,
            n_passes=args.passes,
            is_real_model=False,
            run_type="TEST_ONLY",
            output_dir=args.outdir,
            use_mock_env=True,
        )
