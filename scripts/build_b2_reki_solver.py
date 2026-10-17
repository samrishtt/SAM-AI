#!/usr/bin/env python3
"""
Builder Script for SAM-AI Solver B2 (Option A: Clean Upstream Baseline + Reki Toggles)

Origin: Daniel Franzen Milestone 2 Solution (Apache 2.0)
Architecture:
  - Base: Intel Qwen 3.8B W4A16 AutoRound + Albucino MTP drafter
  - Runtime: SGLang Pennyroyal v2.5.3 (commit d00d88e)
  - Kaggle Shape: NvidiaRtxPro6000
  - Experiment 1: Concurrency Alignment (bm.solver.concurrency = 10, matching SGLang MAXREQ)
  - Modular Reki Toggles: ARC3_CONCURRENCY_MODE, ARC3_FRAME_DIFF, ARC3_REPL_PROBING, etc.
"""

import json
from pathlib import Path

SOURCE_NB = Path("baselines/ARC3_B2_FRANZEN_VERBATIM/arc-agi-3-milestone-2-solution.ipynb")
TARGET_DIR = Path("notebooks/samrish_solver_b2_reki")
TARGET_NB = TARGET_DIR / "arc-agi-3-milestone-2-solution.ipynb"

def build():
    assert SOURCE_NB.exists(), f"Source baseline not found: {SOURCE_NB}"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 0: Header & Attribution
    cell_0_source = [
        "# 🧠 ARC-AGI-3 SAM-AI Solver B2 (Reki Toggle Architecture)\n",
        "\n",
        "**Team:** Samrish B & Vignesh (Parallax AI)  \n",
        "**Baseline Provenance:** Verbatim fork of Daniel Franzen's Milestone 2 Solution (`da-fr/arc-agi-3-solution`, Apache 2.0).  \n",
        "**Upstream Credits:** Harold Bessis, Jeroen Cottaar et al. (Tufa Labs Duck Harness, Apache 2.0); Daniel Franzen (Milestone 2 Winner, Apache 2.0); John Pezzulli (Pennyroyal SGLang v2.5.3).  \n",
        "**Engine:** Intel Qwen 3.8B W4A16 AutoRound + Albucino MTP drafter (Speculative NEXTN) on **NVIDIA RTX Pro 6000**.  \n",
        "\n",
        "### Key Principles:\n",
        "1. **Reki-Style Modular Toggles:** All solver modifications governed by explicit environment variable switches.\n",
        "2. **Experiment 1 (Concurrency Alignment):** Solver concurrency strictly aligned to SGLang engine capacity (`MAXREQ = 10`), eliminating the 120-queue timeout cliff.\n",
        "3. **Decoupled Evaluation:** $N=4$ stochastic passes in offline validation; dedicated single session ($N=1$) for live competition gateway.\n"
    ]
    nb["cells"][0]["source"] = cell_0_source

    # 2. Update Cell 4: Defensive Dynamic Path Resolution
    cell_4_text = "".join(nb["cells"][4]["source"])
    
    # Replace static paths with defensive _resolve_dir
    old_paths_block = (
        "WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'\n"
        "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n"
        "DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'\n"
        "ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'\n"
        "SERVED_MODEL_NAME = 'flashnext'"
    )
    new_paths_block = (
        "def _resolve_dir(candidates):\n"
        "    for c in candidates:\n"
        "        if os.path.exists(c):\n"
        "            return c\n"
        "    return candidates[0]\n\n"
        "WHEELHOUSE_DIR    = _resolve_dir(['/kaggle/input/datasets/dfranzen/pennyroyal-v253', '/kaggle/input/pennyroyal-v253'])\n"
        "MODEL_DIR         = _resolve_dir(['/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1', '/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'])\n"
        "DRAFT_MODEL_DIR   = _resolve_dir(['/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1', '/kaggle/input/albucino-qwen3-8-flash-next-drafter/transformers/default/1'])\n"
        "ORIG_BUNDLE_DIR   = _resolve_dir(['/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy', '/kaggle/input/taaf-kaggle-source-bundle-copy'])\n"
        "SERVED_MODEL_NAME = 'flashnext'"
    )
    assert old_paths_block in cell_4_text, "Failed to match static paths block in Cell 4"
    cell_4_text = cell_4_text.replace(old_paths_block, new_paths_block)
    nb["cells"][4]["source"] = [line + "\n" for line in cell_4_text.splitlines()]

    # 3. Update Cell 16: Reki-Style Ablation Toggles & Experiment 1 Concurrency Alignment
    cell_16_source = [
        "# ==============================================================================\n",
        "# 🧠 SAM-AI B2 ABLATION TOGGLES (Reki-Style Modular Architecture)\n",
        "# ==============================================================================\n",
        "print('[*] Configuring SAM-AI B2 Solver & Reki Ablation Toggles...')\n",
        "\n",
        "# --- 1. Base Exploration & Pass Strategy ---\n",
        "bm.solver.max_actions_per_game = None\n",
        "bm.solver.max_runtime_s_per_game = 48 * 60\n",
        "bm.solver.analyzer_timeout = 900.0\n",
        "bm.game_weights = None\n",
        "\n",
        "# Demonstration excluded games for 10-game offline validation\n",
        "demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', 'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']\n",
        "\n",
        "if TRUE_SUBMISSION:\n",
        "    bm.solver.save_request_logs = False\n",
        "    bm.n_passes = 1\n",
        "    if USE_PRIORITY_SCHEDULING:\n",
        "        bm.solver.max_runtime_s_per_game = 532 * 60\n",
        "else:\n",
        "    bm.n_passes = 4\n",
        "    if USE_PRIORITY_SCHEDULING:\n",
        "        bm.solver.max_runtime_s_per_game = 35 * 60\n",
        "\n",
        "# --- 2. TOGGLE 1: Concurrency Alignment (Experiment 1) ---\n",
        "# Aligns harness concurrency with SGLang MAXREQ capacity to eliminate queue congestion\n",
        "CONCURRENCY_MODE = os.environ.get('ARC3_CONCURRENCY_MODE', 'aligned').strip().lower()\n",
        "if CONCURRENCY_MODE == 'aligned':\n",
        "    # Strictly match SGLang CFG['MAXREQ'] = 10\n",
        "    bm.solver.concurrency = 10\n",
        "elif CONCURRENCY_MODE == 'legacy_high':\n",
        "    # Legacy setting (caused 36 HTTP read timeouts in V7)\n",
        "    bm.solver.concurrency = 120 if TRUE_SUBMISSION else 12\n",
        "else:\n",
        "    try:\n",
        "        bm.solver.concurrency = int(CONCURRENCY_MODE)\n",
        "    except ValueError:\n",
        "        bm.solver.concurrency = 10\n",
        "\n",
        "# Request timeout buffer to prevent premature aborts on complex prefill turns\n",
        "bm.solver.request_timeout = float(os.environ.get('ARC3_REQUEST_TIMEOUT', '180.0'))\n",
        "\n",
        "# --- 3. TOGGLE 2: Visual Frame Differencing (Ablation 2) ---\n",
        "ENABLE_FRAME_DIFF = os.environ.get('ARC3_FRAME_DIFF', '0').strip() in {'1', 'true', 'on'}\n",
        "\n",
        "# --- 4. TOGGLE 3: Pre-Action Sandbox REPL Probing (Ablation 3) ---\n",
        "ENABLE_REPL_PROBING = os.environ.get('ARC3_REPL_PROBING', '0').strip() in {'1', 'true', 'on'}\n",
        "\n",
        "# --- 5. TOGGLE 4: Hysteresis Token Trimming ---\n",
        "HYSTERESIS_TRIM = int(os.environ.get('ARC3_HYSTERESIS_TRIM', '57000'))\n",
        "os.environ['ARC3_HYSTERESIS_TRIM_TOKENS'] = str(HYSTERESIS_TRIM)\n",
        "\n",
        "# --- 6. Summary Banner ---\n",
        "print('=' * 65)\n",
        "print(f'[OK] SAM-AI Solver B2 Active Configuration:')\n",
        "print(f'     Submission Mode   : {TRUE_SUBMISSION}')\n",
        "print(f'     Concurrency Mode  : {CONCURRENCY_MODE} -> {bm.solver.concurrency} streams')\n",
        "print(f'     Request Timeout   : {bm.solver.request_timeout}s')\n",
        "print(f'     Passes per game   : {bm.n_passes}')\n",
        "print(f'     Frame Differencing: {ENABLE_FRAME_DIFF}')\n",
        "print(f'     REPL Probing      : {ENABLE_REPL_PROBING}')\n",
        "print(f'     Hysteresis Retain : {HYSTERESIS_TRIM} tokens')\n",
        "print('=' * 65)\n"
    ]
    nb["cells"][16]["source"] = cell_16_source

    with open(TARGET_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Built SAM-AI Solver B2 notebook at: {TARGET_NB}")

if __name__ == "__main__":
    build()
