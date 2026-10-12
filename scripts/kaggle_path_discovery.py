"""Kaggle input path discovery module for SAM-AI smoke tests.

Standard-library only. Discovers input paths by content rather than brittle
hard-coded path assumptions. Finds:
- sam_ai_parent: directory containing the sam_ai package directory
- environment_files: directory containing ARC-AGI-3 environment metadata and game files
- base_model: directory containing model config.json and weights
"""

import os
import sys
import json
import tarfile
import zipfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def print_layout(root: Path, max_entries: int = 150) -> str:
    """Scan and format the directory layout for diagnostics."""
    lines = [f"=== INPUT LAYOUT ({root}) ==="]
    if not root.exists():
        lines.append(f"  <Directory does not exist: {root}>")
        text = "\n".join(lines)
        print(text)
        return text

    count = 0
    for current, dirs, files in os.walk(root):
        rel = os.path.relpath(current, root)
        if rel != ".":
            lines.append(f"  [DIR]  {rel}")
            count += 1
        for f in files:
            rel_file = os.path.normpath(os.path.join(rel if rel != "." else "", f))
            lines.append(f"  [FILE] {rel_file}")
            count += 1
            if count >= max_entries:
                lines.append(f"  ... [truncated at {max_entries} entries]")
                break
        if count >= max_entries:
            break

    text = "\n".join(lines)
    print(text)
    return text


def _is_valid_env_dir(d: Path) -> bool:
    """Check if directory contains at least one metadata.json with game files."""
    if not d.is_dir():
        return False
    metas = list(d.rglob("metadata.json"))
    if not metas:
        return False
    for m in metas:
        try:
            with open(m, "r", encoding="utf-8") as f:
                data = json.load(f)
            game_id = data.get("game_id")
            class_name = data.get("class_name") or (game_id[0].upper() + game_id[1:] if game_id else "Game")
            candidates = [
                m.parent / f"{class_name.lower()}.py",
                m.parent / f"{class_name}.py",
                m.parent / f"{game_id}.py",
            ]
            if any(c.is_file() for c in candidates):
                return True
        except Exception:
            continue
    return False


def _is_valid_model_dir(d: Path) -> bool:
    """Check if directory contains a Qwen/DeepSeek config.json and weights."""
    if not d.is_dir():
        return False
    cfg_file = d / "config.json"
    if not cfg_file.is_file():
        return False
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "").lower()
        if model_type not in ("qwen2", "deepseek_v2", "deepseek_v3"):
            return False
        weights = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
        return len(weights) > 0
    except Exception:
        return False


def discover_inputs(
    input_dir: str | Path = "/kaggle/input",
    working_dir: str | Path = "/kaggle/working",
) -> Dict[str, Path]:
    """Discover paths to sam_ai, environment_files, and base_model by content.

    Prints the /kaggle/input layout first. Raises FileNotFoundError with layout
    if any required prerequisite is missing.
    """
    input_root = Path(input_dir).resolve()
    working_root = Path(working_dir).resolve()
    working_root.mkdir(parents=True, exist_ok=True)

    # 1. Print full input layout for diagnostics
    layout_str = print_layout(input_root)

    sam_ai_parent: Optional[Path] = None
    env_dir: Optional[Path] = None
    base_model: Optional[Path] = None

    if input_root.exists():
        # A. Find sam_ai package directory
        for current, dirs, files in os.walk(input_root):
            depth = len(Path(current).relative_to(input_root).parts)
            if depth > 6:
                dirs.clear()
                continue
            if "sam_ai" in dirs:
                candidate = Path(current) / "sam_ai"
                if (candidate / "__init__.py").is_file():
                    sam_ai_parent = Path(current)
                    break

        # A2. Archive fallback if sam_ai directory not found
        if not sam_ai_parent:
            for current, dirs, files in os.walk(input_root):
                for f in files:
                    if ("sam_ai" in f.lower() or "source" in f.lower()) and f.endswith((".tar.gz", ".tgz")):
                        arch_path = Path(current) / f
                        extract_dest = working_root / "sam_ai_extracted"
                        extract_dest.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(arch_path, "r:*") as tar:
                            for m in tar.getmembers():
                                if m.issym() or m.islnk() or ".." in m.name:
                                    raise RuntimeError(f"Unsafe archive member: {m.name}")
                            tar.extractall(extract_dest)
                        for c_root, c_dirs, _ in os.walk(extract_dest):
                            if "sam_ai" in c_dirs and (Path(c_root) / "sam_ai" / "__init__.py").is_file():
                                sam_ai_parent = Path(c_root)
                                break
                        if sam_ai_parent:
                            break
                if sam_ai_parent:
                    break

        # B. Find environment_files
        # Prioritize explicit environment_files directories
        for current, dirs, files in os.walk(input_root):
            depth = len(Path(current).relative_to(input_root).parts)
            if depth > 6:
                dirs.clear()
                continue
            if Path(current).name == "environment_files" and _is_valid_env_dir(Path(current)):
                env_dir = Path(current)
                break
        if not env_dir:
            for current, dirs, files in os.walk(input_root):
                p = Path(current)
                if _is_valid_env_dir(p):
                    env_dir = p
                    break

        # C. Find base_model directory
        for current, dirs, files in os.walk(input_root):
            depth = len(Path(current).relative_to(input_root).parts)
            if depth > 6:
                dirs.clear()
                continue
            p = Path(current)
            if _is_valid_model_dir(p):
                base_model = p
                break

    # Validate findings
    missing = []
    if not sam_ai_parent:
        missing.append("sam_ai package (directory containing sam_ai/__init__.py)")
    if not env_dir:
        missing.append("ARC-3 environment_files (directory with metadata.json and game .py)")
    if not base_model:
        missing.append("Base model directory (directory with config.json and weights)")

    if missing:
        err_msg = (
            f"Failed to discover required Kaggle input paths: {', '.join(missing)}\n\n"
            f"{layout_str}"
        )
        raise FileNotFoundError(err_msg)

    print(f"[OK] Discovered sam_ai_parent:   {sam_ai_parent}")
    print(f"[OK] Discovered environment_dir: {env_dir}")
    print(f"[OK] Discovered base_model:      {base_model}")

    return {
        "sam_ai_parent": sam_ai_parent,
        "environment_files": env_dir,
        "base_model": base_model,
        "env_dir": env_dir,  # alias for backwards compatibility
    }


if __name__ == "__main__":
    try:
        inp = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input"
        res = discover_inputs(input_dir=inp)
        print("Discovery succeeded:", res)
    except Exception as e:
        print("Discovery failed:", e)
        sys.exit(1)
