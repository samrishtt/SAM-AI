"""Telemetry schema and environment introspection utilities for SAM-AI smoke tests."""

import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if available."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


_CACHED_VERSIONS: Optional[Dict[str, str]] = None


def get_package_versions() -> Dict[str, str]:
    """Retrieve runtime versions of key dependencies."""
    global _CACHED_VERSIONS
    if _CACHED_VERSIONS is not None:
        return dict(_CACHED_VERSIONS)
    packages = ["torch", "transformers", "arc_agi", "arcengine", "peft"]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not_installed"
    _CACHED_VERSIONS = versions
    return dict(versions)


@dataclass
class CallTelemetry:
    """Schema for per-call model inference telemetry.

    Required fields:
    - prompt_tokens: int
    - completion_tokens: int
    - has_think_close: bool
    - parse_valid: bool
    - latency_ms: float
    - seed: Optional[int]
    - repo_commit_hash: str
    - package_versions: Dict[str, str]
    - prompt_ends_with_think: bool
    - think_prefix_added: bool
    - decode_mode: str
    - decode: Optional[Dict[str, Any]]
    """
    prompt_tokens: int
    completion_tokens: int
    has_think_close: bool
    parse_valid: bool
    latency_ms: float
    seed: Optional[int]
    repo_commit_hash: str
    package_versions: Dict[str, str]
    prompt_ends_with_think: bool = False
    think_prefix_added: bool = False
    decode_mode: str = "plain"
    decode: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        model_text: str,
        parse_valid: bool,
        latency_ms: float,
        seed: Optional[int] = None,
        commit_hash: Optional[str] = None,
        package_versions: Optional[Dict[str, str]] = None,
        prompt_ends_with_think: bool = False,
        think_prefix_added: bool = False,
        decode_mode: str = "plain",
        decode: Optional[Dict[str, Any]] = None,
    ) -> "CallTelemetry":
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            has_think_close="</think>" in model_text,
            parse_valid=parse_valid,
            latency_ms=round(latency_ms, 2),
            seed=seed,
            repo_commit_hash=commit_hash or get_git_commit_hash(),
            package_versions=package_versions or get_package_versions(),
            prompt_ends_with_think=prompt_ends_with_think,
            think_prefix_added=think_prefix_added,
            decode_mode=decode_mode,
            decode=decode,
        )
