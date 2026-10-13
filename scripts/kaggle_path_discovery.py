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


def find_base_model(root=INPUT_ROOT, name_hint="deepseek-r1-distill-qwen-14b"):
    """A directory with config.json + safetensors whose path mentions the model.

    Returns (dir, summary) where summary records what the config says, so the
    run log shows exactly which weights were found. Never picks a directory
    that does not match name_hint (no silent substitution).
    """
    for dirpath, _, filenames in _walk(root):
        if "config.json" in filenames and any(f.endswith(".safetensors") for f in filenames):
            if name_hint.lower() not in dirpath.lower():
                continue
            try:
                with open(os.path.join(dirpath, "config.json"), encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except (OSError, ValueError):
                continue
            q = cfg.get("quantization_config") or {}
            return dirpath, {
                "model_type": cfg.get("model_type"),
                "hidden_size": cfg.get("hidden_size"),
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "quant_method": q.get("quant_method"),
                "load_in_4bit": q.get("load_in_4bit"),
                "bnb_4bit_quant_type": q.get("bnb_4bit_quant_type"),
            }
    return None, None


def discover_inputs(root=INPUT_ROOT, verbose=True):
    layout = tree(root)
    if verbose:
        print(f"--- layout of {root} ---\n{layout}\n--- end layout ---")
    sam_parent = find_sam_ai_parent(root)
    env_dir = find_environment_files(root)
    model_dir, model_cfg = find_base_model(root)
    found = {
        "sam_ai_parent": sam_parent,
        "environment_files": env_dir,
        "base_model_dir": model_dir,
        "base_model_config": model_cfg,
    }
    missing = [k for k in ("sam_ai_parent", "environment_files", "base_model_dir") if not found[k]]
    if missing:
        raise RuntimeError(
            f"Missing inputs: {missing}. Nothing was substituted. Layout seen:\n{layout}"
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
