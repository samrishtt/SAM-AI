"""Official SWE-bench Verified Leaderboard Submission Generator for SAM-AI.

Generates the exact predictions.jsonl file required by the official SWE-bench
evaluation harness (princeton-nlp/SWE-bench) and leaderboard submission protocol.
Does not self-grade; formats predictions for independent verification by SWE-bench.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import urllib.request


@dataclass
class SWEBenchPrediction:
    """Official prediction schema required by swebench.harness.run_evaluation."""
    instance_id: str
    model_patch: str
    model_name_or_path: str = "sam-ai"


class SWEBenchLeaderboardRunner:
    """
    Executes SAM-AI across official SWE-bench instances and compiles the
    official submission artifact for external grading.
    """

    HF_VERIFIED_DATASET_URL = (
        "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test"
    )

    def __init__(self, output_dir: str = "predictions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.output_dir / "swebench_verified_all_preds.jsonl"

    def fetch_verified_instances(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetches official SWE-bench Verified instances from Hugging Face Datasets API.
        """
        url = f"{self.HF_VERIFIED_DATASET_URL}&offset=0&length={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "SAM-AI-Leaderboard-Runner/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = [item["row"] for item in data.get("rows", [])]
                return rows
        except Exception as e:
            print(f"[Warning] Failed to fetch online instances: {e}. Falling back to sample structure.")
            return []

    def format_prediction(self, instance_id: str, patch_content: str, model_id: str = "sam-ai") -> Dict[str, Any]:
        """Formats a single prediction entry strictly matching SWE-bench evaluation specs."""
        return {
            "instance_id": instance_id,
            "model_patch": patch_content.strip(),
            "model_name_or_path": model_id,
        }

    def write_predictions_jsonl(self, predictions: List[Dict[str, Any]]) -> Path:
        """Writes the official all_preds.jsonl file for external harness execution."""
        with open(self.predictions_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        return self.predictions_file

    def generate_evaluation_command(self, predictions_path: Path) -> str:
        """Returns the official bash command to execute independent verification via Docker."""
        return (
            f"python -m swebench.harness.run_evaluation \\\n"
            f"    --dataset_name princeton-nlp/SWE-bench_Verified \\\n"
            f"    --predictions_path {predictions_path.resolve()} \\\n"
            f"    --run_id sam_ai_verified_eval \\\n"
            f"    --max_workers 4"
        )
