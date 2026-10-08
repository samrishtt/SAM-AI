#!/usr/bin/env python3
"""
SAM-AI Domain-Neutral Failure Memory (Phase 4)
==============================================
Records, classifies, and aggregates task failures across domains to discover
systemic weaknesses (e.g. perception, representation, reasoning, search, planning,
memory, tool use, verification, execution, state tracking).

Rule: Do not call this 'learning' until later system behavior measurably changes
because of the stored failures.
"""

import json
import os
import time
from typing import Dict, List, Optional
from collections import Counter

from sam_ai.substrate.interfaces import FailureMemory, FailureRecord, FailureCategory, VerificationResult, VerificationStatus


class StructuredFailureMemory(FailureMemory):
    """Persistent structured failure memory store."""

    def __init__(self, storage_path: str = "eval_reports/failure_memory.jsonl"):
        self.storage_path = storage_path
        self._records: List[FailureRecord] = []
        os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else ".", exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    rec = FailureRecord(
                        task_id=data["task_id"],
                        domain=data["domain"],
                        model_revision=data.get("model_revision", "unknown"),
                        strategy=data.get("strategy", "unknown"),
                        candidate=data.get("candidate"),
                        output=data.get("output"),
                        verification_result=VerificationResult(
                            status=VerificationStatus(data["verification_status"]),
                            confidence=data.get("confidence", 0.0),
                            details=data.get("verification_details", ""),
                        ),
                        failure_category=FailureCategory(data.get("failure_category", "unknown")),
                        failure_details=data.get("failure_details", ""),
                        latency_ms=data.get("latency_ms", 0.0),
                        compute_cost=data.get("compute_cost", 0.0),
                        timestamp=data.get("timestamp", time.time()),
                    )
                    self._records.append(rec)
        except Exception:
            pass

    def record_failure(self, failure: FailureRecord) -> None:
        self._records.append(failure)
        # Append to jsonl
        entry = {
            "task_id": failure.task_id,
            "domain": failure.domain,
            "model_revision": failure.model_revision,
            "strategy": failure.strategy,
            "candidate": str(failure.candidate)[:500],
            "output": str(failure.output)[:500],
            "verification_status": failure.verification_result.status.value,
            "confidence": failure.verification_result.confidence,
            "verification_details": failure.verification_result.details,
            "failure_category": failure.failure_category.value,
            "failure_details": failure.failure_details,
            "latency_ms": failure.latency_ms,
            "compute_cost": failure.compute_cost,
            "timestamp": failure.timestamp,
        }
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_failures(
        self,
        domain: Optional[str] = None,
        category: Optional[FailureCategory] = None,
    ) -> List[FailureRecord]:
        results = self._records
        if domain:
            results = [r for r in results if r.domain == domain]
        if category:
            results = [r for r in results if r.failure_category == category]
        return results

    def distribution(self, domain: Optional[str] = None) -> Dict[str, int]:
        records = [r for r in self._records if domain is None or r.domain == domain]
        counts = Counter(r.failure_category.value for r in records)
        return dict(counts)

    def total_failures(self, domain: Optional[str] = None) -> int:
        if domain:
            return sum(1 for r in self._records if r.domain == domain)
        return len(self._records)
