"""
SAM-AI Intelligence Foundation: Frozen Evaluation & Promotion Gates
===================================================================
Enforces deterministic comparison between candidate models and baselines
on frozen evaluation suites. Requires zero critical regressions outside
tolerances, validated artifact status, and explicit operator authorization
before promotion. Supports deterministic, reversible rollbacks.

Operational Invariant:
- Unverified or unvalidated artifacts cannot pass the promotion gate.
- Comparisons strictly enforce metric-level tolerance margins.
- Every promotion and rollback appends an immutable audit record.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

from sam_ai.foundation.artifacts import (
    ModelArtifactManifest,
    ArtifactKind,
    ValidationStatus
)


@dataclass
class FrozenEvaluationSuite:
    """Immutable evaluation benchmark definition with regression tolerances."""
    suite_id: str
    name: str
    version: str
    metric_thresholds: Dict[str, float] = field(default_factory=dict)
    tolerance_margins: Dict[str, float] = field(default_factory=dict)
    benchmark_dataset_ids: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrozenEvaluationSuite:
        return cls(**data)


@dataclass
class MetricComparison:
    """Comparison of a single evaluation metric between baseline and candidate."""
    metric_name: str
    baseline_score: float
    candidate_score: float
    delta: float
    rel_diff_pct: float
    tolerance_margin: float
    status: str  # "IMPROVED", "NEUTRAL", "REGRESSED_ACCEPTABLE", "REGRESSED_CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineVsCandidateComparison:
    """Full comparative evaluation report between candidate and baseline."""
    baseline_artifact_id: str
    candidate_artifact_id: str
    eval_suite_id: str
    metrics: Dict[str, MetricComparison]
    overall_improved: bool
    critical_regressions: List[str]
    recommendation: str  # "ACCEPT", "REJECT", "MANUAL_REVIEW_REQUIRED"
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = {k: v.to_dict() if isinstance(v, MetricComparison) else v for k, v in self.metrics.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineVsCandidateComparison:
        payload = dict(data)
        if "metrics" in payload and isinstance(payload["metrics"], dict):
            metrics_map = {}
            for k, v in payload["metrics"].items():
                if isinstance(v, dict):
                    metrics_map[k] = MetricComparison(**v)
                elif isinstance(v, MetricComparison):
                    metrics_map[k] = v
            payload["metrics"] = metrics_map
        return cls(**payload)


def compare_candidate_to_baseline(
    baseline_scores: Dict[str, float],
    candidate_scores: Dict[str, float],
    suite: FrozenEvaluationSuite,
    baseline_id: str,
    candidate_id: str
) -> BaselineVsCandidateComparison:
    """
    Deterministically compares candidate scores against baseline scores
    using the thresholds and tolerance margins of a FrozenEvaluationSuite.
    """
    metrics: Dict[str, MetricComparison] = {}
    critical_regressions: List[str] = []
    has_improvements = False

    all_metric_keys = set(baseline_scores.keys()).union(candidate_scores.keys())

    for metric in all_metric_keys:
        b_val = baseline_scores.get(metric, 0.0)
        c_val = candidate_scores.get(metric, 0.0)
        delta = c_val - b_val
        rel_diff = (delta / b_val * 100.0) if b_val != 0.0 else (100.0 if delta > 0 else 0.0)
        tolerance = suite.tolerance_margins.get(metric, 0.0)

        # Check threshold
        min_threshold = suite.metric_thresholds.get(metric)
        below_threshold = (min_threshold is not None and c_val < min_threshold)

        if delta > 0:
            status = "IMPROVED"
            has_improvements = True
        elif delta == 0:
            status = "NEUTRAL"
        else:
            # Negative delta
            if abs(delta) <= tolerance:
                status = "REGRESSED_ACCEPTABLE"
            else:
                status = "REGRESSED_CRITICAL"
                critical_regressions.append(
                    f"Metric '{metric}' regressed by {abs(delta):.4f} (allowed tolerance: {tolerance:.4f})"
                )

        if below_threshold and status != "REGRESSED_CRITICAL":
            critical_regressions.append(
                f"Metric '{metric}' candidate score ({c_val:.4f}) is below minimum threshold ({min_threshold:.4f})"
            )
            status = "REGRESSED_CRITICAL"

        metrics[metric] = MetricComparison(
            metric_name=metric,
            baseline_score=b_val,
            candidate_score=c_val,
            delta=delta,
            rel_diff_pct=rel_diff,
            tolerance_margin=tolerance,
            status=status
        )

    # Determine recommendation
    if critical_regressions:
        recommendation = "REJECT"
        overall_improved = False
    elif has_improvements:
        recommendation = "ACCEPT"
        overall_improved = True
    else:
        recommendation = "MANUAL_REVIEW_REQUIRED"
        overall_improved = False

    return BaselineVsCandidateComparison(
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
        eval_suite_id=suite.suite_id,
        metrics=metrics,
        overall_improved=overall_improved,
        critical_regressions=critical_regressions,
        recommendation=recommendation
    )


@dataclass
class PromotionDecision:
    """Audit record for a model promotion or rollback."""
    candidate_artifact_id: str
    previous_promoted_id: Optional[str]
    action: str  # "PROMOTED", "REJECTED", "ROLLED_BACK"
    operator: str
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    comparison_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromotionDecision:
        return cls(**data)


class PromotionGate:
    """
    Promotion Gatekeeper.
    Manages active model pointers, enforces validation criteria,
    and logs immutable promotion and rollback decisions.
    """

    def __init__(self, state_dir: Union[str, Path]):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.state_dir / "active_model.json"
        self.history_file = self.state_dir / "promotion_history.jsonl"

    def get_active_model_id(self) -> Optional[str]:
        """Returns currently promoted model artifact ID, if any."""
        if not self.active_file.exists():
            return None
        try:
            with open(self.active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("active_artifact_id")
        except Exception:
            return None

    def get_history(self) -> List[PromotionDecision]:
        """Returns all promotion and rollback records in chronological order."""
        records: List[PromotionDecision] = []
        if not self.history_file.exists():
            return records
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(PromotionDecision.from_dict(json.loads(line)))
                    except Exception:
                        pass
        return records

    def promote(
        self,
        candidate_manifest: ModelArtifactManifest,
        comparison: BaselineVsCandidateComparison,
        operator: str,
        rationale: str,
        force_override: bool = False
    ) -> PromotionDecision:
        """
        Promotes a candidate artifact to active status after strict validation.
        """
        # 1. Validation Status Check
        if candidate_manifest.validation_status != ValidationStatus.VALIDATED:
            raise ValueError(
                f"Candidate artifact '{candidate_manifest.artifact_id}' has validation status "
                f"'{candidate_manifest.validation_status}'. Only VALIDATED artifacts may be promoted."
            )

        # 2. Training/Adaptation Compliance Check
        is_valid, reasons = candidate_manifest.validate_for_training()
        if not is_valid:
            raise ValueError(
                f"Candidate artifact failed integrity check: {'; '.join(reasons)}"
            )

        # 3. Comparison Check
        if comparison.recommendation != "ACCEPT" and not force_override:
            raise ValueError(
                f"Candidate artifact comparison recommendation is '{comparison.recommendation}'. "
                f"Critical regressions: {comparison.critical_regressions}. Cannot promote without passing eval."
            )

        previous_id = self.get_active_model_id()
        decision = PromotionDecision(
            candidate_artifact_id=candidate_manifest.artifact_id,
            previous_promoted_id=previous_id,
            action="PROMOTED",
            operator=operator,
            rationale=rationale,
            comparison_summary={
                "eval_suite_id": comparison.eval_suite_id,
                "overall_improved": comparison.overall_improved,
                "metrics": {k: v.to_dict() for k, v in comparison.metrics.items()}
            }
        )

        # Persist Active State
        with open(self.active_file, "w", encoding="utf-8") as f:
            json.dump({
                "active_artifact_id": candidate_manifest.artifact_id,
                "promoted_at": decision.timestamp,
                "promoted_by": operator,
                "model_id": candidate_manifest.model_id,
                "kind": candidate_manifest.kind.value
            }, f, indent=2)

        # Append to Audit Log
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

        return decision

    def rollback(
        self,
        target_artifact_id: str,
        operator: str,
        reason: str
    ) -> PromotionDecision:
        """
        Reverts active model pointer to a previous validated artifact.
        """
        current_id = self.get_active_model_id()
        if not current_id:
            raise ValueError("No active model currently promoted to rollback from.")

        decision = PromotionDecision(
            candidate_artifact_id=target_artifact_id,
            previous_promoted_id=current_id,
            action="ROLLED_BACK",
            operator=operator,
            rationale=reason,
            comparison_summary={"reason": reason, "restored_target_id": target_artifact_id}
        )

        with open(self.active_file, "w", encoding="utf-8") as f:
            json.dump({
                "active_artifact_id": target_artifact_id,
                "promoted_at": decision.timestamp,
                "promoted_by": operator,
                "note": f"Rolled back from {current_id}"
            }, f, indent=2)

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

        return decision
