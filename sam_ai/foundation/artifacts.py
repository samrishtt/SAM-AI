"""
SAM-AI Intelligence Foundation: Model & Artifact Manifest Protocol
==================================================================
Machine-readable specifications for model checkpoints, LoRA adapters,
and candidate artifacts. Enforces immutable lineage, strict provenance,
and explicit validation states.

Operational Invariant:
- Unknown metadata must remain explicitly unknown ("UNKNOWN") and fail
  closed for any operation depending on it.
- Never infer compatibility from a model name alone.
"""

from __future__ import annotations
import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple


class ArtifactKind(str, Enum):
    BASE_MODEL = "base_model"
    ADAPTER = "adapter"
    MERGED_MODEL = "merged_model"
    TEACHER_REFERENCE = "teacher_reference"
    STUDENT_CANDIDATE = "student_candidate"


class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"


class PrecisionType(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT4 = "int4"
    FLOAT32 = "float32"
    UNKNOWN = "UNKNOWN"


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelArtifactManifest:
    """
    Standard machine-readable manifest for foundation models, adapters, and references.
    """
    artifact_id: str
    kind: ArtifactKind
    model_id: str
    source_path: str
    revision: str = "UNKNOWN"
    architecture: str = "UNKNOWN"
    tokenizer_id: str = "UNKNOWN"
    weight_hashes: Dict[str, str] = field(default_factory=dict)
    config_hash: Optional[str] = None
    license: str = "UNKNOWN"
    provenance_record: Dict[str, Any] = field(default_factory=dict)
    precision: PrecisionType = PrecisionType.UNKNOWN
    intended_runtime: str = "transformers"
    tested_library_versions: Dict[str, str] = field(default_factory=dict)
    parent_artifact_ids: List[str] = field(default_factory=list)
    adapter_target_modules: List[str] = field(default_factory=list)
    experiment_id: Optional[str] = None
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value if isinstance(self.kind, ArtifactKind) else str(self.kind)
        d["precision"] = self.precision.value if isinstance(self.precision, PrecisionType) else str(self.precision)
        d["validation_status"] = self.validation_status.value if isinstance(self.validation_status, ValidationStatus) else str(self.validation_status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelArtifactManifest:
        payload = dict(data)
        if "kind" in payload and isinstance(payload["kind"], str):
            payload["kind"] = ArtifactKind(payload["kind"])
        if "precision" in payload and isinstance(payload["precision"], str):
            payload["precision"] = PrecisionType(payload["precision"])
        if "validation_status" in payload and isinstance(payload["validation_status"], str):
            payload["validation_status"] = ValidationStatus(payload["validation_status"])
        return cls(**payload)

    def validate_for_training(self) -> Tuple[bool, List[str]]:
        """
        Validates artifact readiness for training or adaptation.
        Fails closed if essential architectural or licensing metadata is missing.
        """
        reasons = []
        if self.license in ("UNKNOWN", "", None):
            reasons.append("Artifact license is UNKNOWN. Cannot authorize training without verified license.")
        if self.architecture in ("UNKNOWN", "", None):
            reasons.append("Model architecture is UNKNOWN. Cannot compute target projection layers.")
        if self.kind == ArtifactKind.ADAPTER and not self.parent_artifact_ids:
            reasons.append("Adapter artifact must explicitly declare its parent base_model artifact ID.")
        return len(reasons) == 0, reasons


class ArtifactRegistry:
    """
    Offline local filesystem registry for ModelArtifactManifest records.
    Persists deterministic JSON records under evidence/ or local storage.
    """

    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, ModelArtifactManifest] = {}
        self.reload()

    def reload(self) -> None:
        """Reloads all manifests from disk."""
        self._artifacts.clear()
        for manifest_file in self.registry_dir.glob("*.artifact.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    manifest = ModelArtifactManifest.from_dict(data)
                    self._artifacts[manifest.artifact_id] = manifest
            except Exception as e:
                print(f"[Warning] Failed to load artifact from {manifest_file}: {e}")

    def register(self, manifest: ModelArtifactManifest, overwrite: bool = False) -> Path:
        """Registers and persists a model artifact manifest."""
        if manifest.artifact_id in self._artifacts and not overwrite:
            raise ValueError(f"Artifact ID '{manifest.artifact_id}' already registered.")
        
        self._artifacts[manifest.artifact_id] = manifest
        file_path = self.registry_dir / f"{manifest.artifact_id}.artifact.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return file_path

    def get(self, artifact_id: str) -> Optional[ModelArtifactManifest]:
        """Retrieves artifact by ID."""
        return self._artifacts.get(artifact_id)

    def list_artifacts(self, kind: Optional[ArtifactKind] = None) -> List[ModelArtifactManifest]:
        """Lists artifacts, optionally filtered by kind."""
        if kind is None:
            return list(self._artifacts.values())
        return [a for a in self._artifacts.values() if a.kind == kind]
