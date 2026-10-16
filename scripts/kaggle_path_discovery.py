"""Find Kaggle inputs by what they contain, not by a guessed path.

Why: the smoke test halted on a path guess ("sam_ai not found", "environment
data not found"). Kaggle mounts differ by runtime (/kaggle/input/<slug> vs
/kaggle/input/datasets/<owner>/<slug> vs /kaggle/input/competitions/<slug> vs
/kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>), and a
dataset uploaded as .tar.gz may be mounted already extracted. This module
searches for the thing itself, always prints the layout it saw, and raises with
that layout if anything is missing. It never falls back silently.

Usage (first cell of the notebook, and in scripts/run_real_model_smoke_test.py):

    from kaggle_path_discovery import discover_inputs
    paths = discover_inputs()          # prints tree, raises if anything is missing
    sys.path.insert(0, paths["sam_ai_parent"])
    ENV_DIR   = paths["environment_files"]
    MODEL_DIR = paths["base_model_dir"]

Stdlib only. Does not download anything and does not touch the network.
"""
import json
import os
import sys
import tarfile

INPUT_ROOT = "/kaggle/input"
EXTRACT_DIR = "/kaggle/working/sam_ai_src"
MAX_DEPTH = 8


def _walk(root, max_depth=MAX_DEPTH):
    """os.walk that follows symlinks safely and stops at max_depth."""
    seen = set()
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        if dirpath.count(os.sep) - base >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        yield dirpath, dirnames, sorted(filenames)


def tree(root=INPUT_ROOT, max_depth=5, files_per_dir=6, max_lines=300):
    """Human-readable layout, so a failed run still tells us what was mounted."""
    if not os.path.isdir(root):
        return f"{root} does not exist"
    lines, base = [], root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in _walk(root, max_depth):
        depth = dirpath.count(os.sep) - base
        if depth >= max_depth:
            dirnames[:] = []
        lines.append(f"{'  ' * depth}{os.path.basename(dirpath) or dirpath}/")
        shown = filenames[:files_per_dir]
        for f in shown:
            lines.append(f"{'  ' * (depth + 1)}{f}")
        if len(filenames) > len(shown):
            lines.append(f"{'  ' * (depth + 1)}... (+{len(filenames) - len(shown)} files)")
        if len(lines) >= max_lines:
            lines.append("... (tree truncated)")
            break
    return "\n".join(lines)


def _safe_extract(tar_path, dest):
    """Extract a tarball, refusing links, devices and path escapes."""
    os.makedirs(dest, exist_ok=True)
    dest_real = os.path.realpath(dest)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for m in members:
            if not (m.isfile() or m.isdir()):
                raise RuntimeError(f"refusing non-regular tar member: {m.name}")
            target = os.path.realpath(os.path.join(dest_real, m.name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise RuntimeError(f"refusing tar member outside destination: {m.name}")
        tar.extractall(dest_real, members=members)
    return dest_real


def find_sam_ai_parent(root=INPUT_ROOT, extract_dir=EXTRACT_DIR):
    """Directory to put on sys.path so `import sam_ai` works.

    Prefers an already-extracted package (Kaggle usually unpacks archives);
    falls back to extracting sam_ai_source_bundle*.tar.gz.
    """
    for dirpath, dirnames, filenames in _walk(root):
        if os.path.basename(dirpath) == "sam_ai" and "__init__.py" in filenames:
            return os.path.dirname(dirpath)
    for dirpath, _, filenames in _walk(root):
        for f in filenames:
            if f.startswith("sam_ai_source_bundle") and f.endswith((".tar.gz", ".tgz")):
                parent = _safe_extract(os.path.join(dirpath, f), extract_dir)
                if os.path.isfile(os.path.join(parent, "sam_ai", "__init__.py")):
                    return parent
    return None


def find_environment_files(root=INPUT_ROOT):
    """A directory named environment_files that really contains game metadata."""
    for dirpath, dirnames, _ in _walk(root):
        if os.path.basename(dirpath) == "environment_files":
            for sub, _, files in _walk(dirpath, max_depth=4):
                if "metadata.json" in files:
                    return dirpath
    return None


class ModelMismatchError(RuntimeError):
    """Raised when discovered model contradicts the explicitly requested target model."""
    pass


def verify_model_identity(model_dir, config=None, expected_model="qwen3.8"):
    """
    Verifies that the discovered model directory and config match the expected model.
    Fails closed if the expected model is missing, or if a different model
    (such as DeepSeek) is discovered when Qwen was expected.
    """
    if not model_dir or not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    if config is None:
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as fh:
                    config = json.load(fh)
            except Exception as e:
                raise RuntimeError(f"Corrupt or unreadable config.json in '{model_dir}': {e}")
        else:
            config = {}

    expected_lower = (expected_model or "").lower()
    dir_lower = str(model_dir).lower()
    name_or_path = str(config.get("_name_or_path", "")).lower()
    model_type = str(config.get("model_type", "")).lower()

    if any(k in expected_lower for k in ("qwen3.8", "qwen38", "qwen3", "qwen")):
        # Refuse DeepSeek silent substitution
        if "deepseek" in dir_lower or "deepseek" in name_or_path:
            raise ModelMismatchError(
                f"MODEL IDENTITY MISMATCH: Smoke test is explicitly configured for Qwen3.8 ('{expected_model}'), "
                f"but discovered DeepSeek checkpoint in '{model_dir}'. "
                f"Refusing silent model substitution. To evaluate DeepSeek, set target_model='deepseek' explicitly."
            )
        is_qwen = ("qwen" in dir_lower or "qwen" in model_type or "qwen" in name_or_path)
        if not is_qwen:
            raise ModelMismatchError(
                f"MODEL IDENTITY MISMATCH: Expected Qwen model for target '{expected_model}', "
                f"but discovered incompatible model in '{model_dir}' (model_type='{model_type}')."
            )
        return {
            "verified": True,
            "expected_model": expected_model,
            "discovered_dir": model_dir,
            "model_type": model_type,
            "family": "qwen3.8" if ("qwen3.8" in dir_lower or "qwen3.8" in name_or_path) else "qwen",
        }
    elif any(k in expected_lower for k in ("deepseek", "unsloth-deepseek")):
        is_deepseek = ("deepseek" in dir_lower or "deepseek" in name_or_path)
        if not is_deepseek:
            raise ModelMismatchError(
                f"MODEL IDENTITY MISMATCH: Explicitly configured for DeepSeek baseline ('{expected_model}'), "
                f"but discovered non-DeepSeek model in '{model_dir}'."
            )
        return {
            "verified": True,
            "expected_model": expected_model,
            "discovered_dir": model_dir,
            "model_type": model_type,
            "family": "deepseek_r1_distill",
        }
    return {
        "verified": True,
        "expected_model": expected_model,
        "discovered_dir": model_dir,
        "model_type": model_type,
        "family": "generic",
    }


def find_base_model(root=INPUT_ROOT, name_hint=None, target_model=None):
    """A directory with config.json + safetensors whose path mentions the model.

    Returns (dir, summary) where summary records what the config says, so the
    run log shows exactly which weights were found. Never picks a directory
    that does not match the candidate hints (no silent substitution with wrong models).
    Prioritizes primary Qwen3.8-27B, then baseline DeepSeek-14B when unspecified.
    When target_model is specified, searches strictly for that model family.
    """
    if target_model:
        t_lower = target_model.lower()
        if any(k in t_lower for k in ("qwen3.8", "qwen38", "qwen3")):
            hints = ["qwen3.8-27b", "qwen3.8", "qwen3"]
        elif "deepseek" in t_lower:
            hints = ["deepseek-r1-distill-qwen-14b", "unsloth-deepseek", "deepseek"]
        else:
            hints = [target_model]
    elif name_hint:
        hints = [name_hint]
    else:
        # Priority order when unspecified: primary research checkpoint first, then legacy baseline
        hints = ["qwen3.8-27b", "qwen3.8", "deepseek-r1-distill-qwen-14b", "unsloth-deepseek"]

    for hint in hints:
        for dirpath, _, filenames in _walk(root):
            if "config.json" in filenames and any(f.endswith(".safetensors") for f in filenames):
                if hint.lower() not in dirpath.lower():
                    continue
                try:
                    with open(os.path.join(dirpath, "config.json"), encoding="utf-8") as fh:
                        cfg = json.load(fh)
                except (OSError, ValueError):
                    continue
                q = cfg.get("quantization_config") or {}
                return dirpath, {
                    "matched_hint": hint,
                    "model_type": cfg.get("model_type"),
                    "hidden_size": cfg.get("hidden_size"),
                    "num_hidden_layers": cfg.get("num_hidden_layers"),
                    "quant_method": q.get("quant_method"),
                    "load_in_4bit": q.get("load_in_4bit"),
                    "bnb_4bit_quant_type": q.get("bnb_4bit_quant_type"),
                }
    return None, None


def discover_inputs(root=INPUT_ROOT, verbose=True, name_hint=None, target_model=None):
    layout = tree(root)
    if verbose:
        print(f"--- layout of {root} ---\n{layout}\n--- end layout ---")
    sam_parent = find_sam_ai_parent(root)
    env_dir = find_environment_files(root)
    model_dir, model_cfg = find_base_model(root, name_hint=name_hint, target_model=target_model)
    found = {
        "sam_ai_parent": sam_parent,
        "environment_files": env_dir,
        "base_model_dir": model_dir,
        "base_model_config": model_cfg,
    }
    missing = [k for k in ("sam_ai_parent", "environment_files", "base_model_dir") if not found[k]]
    if missing:
        target_info = f" for target '{target_model or name_hint}'" if (target_model or name_hint) else ""
        raise RuntimeError(
            f"Missing inputs: {missing}{target_info}. Nothing was substituted. Layout seen:\n{layout}"
        )
    if verbose:
        print(json.dumps(found, indent=2))
    return found


if __name__ == "__main__":
    try:
        discover_inputs()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
