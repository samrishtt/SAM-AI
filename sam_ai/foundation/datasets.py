"""
SAM-AI Intelligence Foundation: Dataset Provenance & Manifest Protocol
=====================================================================
Machine-readable specifications for training, validation, and evaluation
datasets. Enforces strict provenance, verifiable SHA-256 checksums,
and licensing compliance.

Operational Invariant:
- Unverified datasets or datasets with UNKNOWN licenses fail closed.
- Synthetic datasets must explicitly identify their generator/teacher.
- Unit test fixtures must explicitly declare is_test_fixture=True.
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

from sam_ai.foundation.artifacts import ValidationStatus, compute_file_sha256


@dataclass
class SplitInfo:
    """Metadata and integrity hash for an individual dataset split."""
    name: str
    num_examples: int
    file_path: Optional[str] = None
    sha256_hash: Optional[str] = None
    format: str = "jsonl"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplitInfo:
        return cls(**data)


@dataclass
class DatasetManifest:
    """
    Standard machine-readable manifest for training and evaluation datasets.
    """
    dataset_id: str
    source: str
    version: str
    license: str = "UNKNOWN"
    provenance: Dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False
    generator_id: Optional[str] = None
    filtering_criteria: List[str] = field(default_factory=list)
    schema_definition: Dict[str, str] = field(default_factory=dict)
    splits: Dict[str, SplitInfo] = field(default_factory=dict)
    is_test_fixture: bool = False
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["validation_status"] = (
            self.validation_status.value
            if isinstance(self.validation_status, ValidationStatus)
            else str(self.validation_status)
        )
        d["splits"] = {k: v.to_dict() if isinstance(v, SplitInfo) else v for k, v in self.splits.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetManifest:
        payload = dict(data)
        if "validation_status" in payload and isinstance(payload["validation_status"], str):
            payload["validation_status"] = ValidationStatus(payload["validation_status"])
        if "splits" in payload and isinstance(payload["splits"], dict):
            splits_map = {}
            for k, v in payload["splits"].items():
                if isinstance(v, dict):
                    splits_map[k] = SplitInfo.from_dict(v)
                elif isinstance(v, SplitInfo):
                    splits_map[k] = v
            payload["splits"] = splits_map
        return cls(**payload)


def validate_dataset_manifest(
    manifest: DatasetManifest,
    base_dir: Optional[Union[str, Path]] = None
) -> Tuple[bool, List[str]]:
    """
    Validates dataset manifest compliance and physical integrity.
    Fails closed if licensing, provenance, or file hashes do not match.
    """
    reasons: List[str] = []

    # 1. License Check
    if manifest.license in ("UNKNOWN", "", None):
        reasons.append("Dataset license is UNKNOWN. Rejecting training use without verified license.")

    # 2. Synthetic Provenance Check
    if manifest.is_synthetic and not manifest.generator_id:
        reasons.append("Synthetic dataset must declare generator_id / teacher origin.")

    # 3. Production Non-Fixture Checks
    if not manifest.is_test_fixture:
        if not manifest.provenance or not any(k in manifest.provenance for k in ("author", "source_url", "organization")):
            reasons.append("Production dataset must declare verified provenance (author, source_url, or organization).")
        if not manifest.splits:
            reasons.append("Production dataset must contain at least one split.")
        total_examples = sum(s.num_examples for s in manifest.splits.values())
        if total_examples <= 0:
            reasons.append("Production dataset must have at least one example across its splits.")

    # 4. Disk & SHA-256 Integrity Verification
    base_path = Path(base_dir) if base_dir is not None else Path.cwd()
    for split_name, split in manifest.splits.items():
        if split.file_path:
                fp = Path(split.file_path)
                target_file = fp if fp.is_absolute() else base_path / fp
                if not target_file.exists():
                    reasons.append(f"Split '{split_name}' file not found at: {target_file}")
                elif split.sha256_hash:
                    try:
                        computed_hash = compute_file_sha256(target_file)
                        if computed_hash.lower() != split.sha256_hash.lower():
                            reasons.append(
                                f"Split '{split_name}' SHA-256 mismatch: expected {split.sha256_hash}, got {computed_hash}"
                            )
                    except Exception as e:
                        reasons.append(f"Failed to hash split '{split_name}': {e}")

    is_valid = len(reasons) == 0
    return is_valid, reasons


class DatasetRegistry:
    """
    Offline local filesystem registry for DatasetManifest records.
    Persists deterministic JSON records under evidence/ or local storage.
    """

    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, DatasetManifest] = {}
        self.reload()

    def reload(self) -> None:
        """Reloads all dataset manifests from disk."""
        self._datasets.clear()
        for manifest_file in self.registry_dir.glob("*.dataset.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    manifest = DatasetManifest.from_dict(data)
                    self._datasets[manifest.dataset_id] = manifest
            except Exception as e:
                print(f"[Warning] Failed to load dataset from {manifest_file}: {e}")

    def register(self, manifest: DatasetManifest, overwrite: bool = False) -> Path:
        """Registers and persists a dataset manifest."""
        if manifest.dataset_id in self._datasets and not overwrite:
            raise ValueError(f"Dataset ID '{manifest.dataset_id}' already registered.")
        
        self._datasets[manifest.dataset_id] = manifest
        file_path = self.registry_dir / f"{manifest.dataset_id}.dataset.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return file_path

    def get(self, dataset_id: str) -> Optional[DatasetManifest]:
        """Retrieves dataset manifest by ID."""
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[DatasetManifest]:
        """Lists all registered dataset manifests."""
        return list(self._datasets.values())
