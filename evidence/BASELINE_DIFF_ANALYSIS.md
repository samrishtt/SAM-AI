# ARC-AGI-3 Golden Baseline vs. Solver V8 Comparative Analysis

*Generated on 2026-10-08*  
*Baseline: `baselines/ARC3_GOLDEN_BASELINE/arc-agi-3-milestone-2-solution.ipynb` (Score: 29.21, 2026-10-03)*  
*Comparison Target: `notebooks/samrish_solver_v8/arc-agi-3-milestone-2-solution.ipynb` (Scores: 26.04 - 25.76)*  
*Git Tag: `arc3-golden-baseline-29.21`*

---

## 1. Structural & Environmental Diff Summary

| Dimension | Golden Baseline (29.21) | Solver V8 (25.76 - 26.04) | Delta Impact |
| :--- | :--- | :--- | :--- |
| **Model Sources** | `intel-qwen3.8-flash-next-w4a16-autoround` + `albucino-qwen3-8-flash-next-drafter` | Identical model sources | Identical base weights |
| **Hardware** | `NvidiaRtxPro6000` | `NvidiaRtxPro6000` | Identical GPU shape |
| **Dataset Sources** | `pennyroyal-v253`, `taaf-kaggle-source-bundle-copy`, `samrish-skills-bundle` | Skills bundle omitted | Bundle missing in V8 metadata |
| **Path Resolution** | Static `/kaggle/input/datasets/...` | Dynamic `_resolve_dir` | Fixed 6s crash in V8 |
| **System Prompt** | Unmodified upstream prompt | Added 12-line `ARC3_SYSTEM_PROMPT_PREFIX` | Consumes tokens & constrains policy |
| **Harness Meta-Flags** | Zero custom env overrides | 14 custom `os.environ` overrides injected | Heavily alters agent exploration loop |
| **Action Guards** | Default action selection | `EXPOSE_RESET`, `EXPOSE_UNDO`, `STALE_STATE_BLOCK`, `BATCH_NOOP_BLOCK` | Alters action distribution |
| **Context Trimming** | Default harness threshold | `ARC3_HYSTERESIS_TRIM_TOKENS="62000"` | Alters KV cache truncation |

---

## 2. Code Injections in Solver V8 (Cell 7)
The golden baseline contains only core object flood-fill segmentation and passes execution directly to the solver. In contrast, Solver V8 injected over 110 lines of heuristic overrides:
1. **Custom System Prompt**: Overrides model directives with verbose role instructions ("You are SAM-AI ARC-AGI-3 Sovereign Autonomous Agent...").
2. **Action Flags**: Forces environment to expose `RESET` and `UNDO` actions and blocks `NOOP` actions.
3. **Macro Pathfinding**: Injects `sam_ai_v8_bfs_shortest_path` and `sam_ai_shortest_path` into `sandbox_helpers`.
4. **Frame Differentials**: Computes pixel delta arrays across turns.

---

## 3. Stochasticity & Variance Analysis
Historical completed runs on Kaggle ARC-AGI-3:
* 2026-10-01: **27.73**
* 2026-10-03: **29.21** (Peak)
* 2026-10-06: **28.29**
* 2026-10-07: **26.04**
* 2026-10-08: **25.76**

*Mean*: 27.41 | *Standard Deviation*: $\sigma \approx 1.54$  
*Interpretation*:
* The 0.92 drop between 29.21 (Oct 3) and 28.29 (Oct 6) is well within 1 standard deviation ($0.60\sigma$) and consistent with stochastic game exploration / LLM decoding temperature variance.
* The drops to 26.04 and 25.76 represent a $>2.2\sigma$ shift from the peak. This strongly indicates that the cumulative addition of heuristic constraints in V7/V8 degraded net game completion rather than random noise alone.

---

## 4. Ranked Hypotheses for the Score Decline

### **Hypothesis 1 (Highest Confidence): Heuristic Action Penalties & Early Aborts**
* **Mechanism**: Enabling `EXPOSE_UNDO`, `EXPOSE_RESET`, and `ARC3_NOOP_ABORT_THRESHOLD="3"` causes the solver to abort difficult levels prematurely or burn limited action budgets on resets rather than continuing stochastic multi-turn exploration.
* **Verification Test**: Run local ablation isolating `EXPOSE_RESET`, `EXPOSE_UNDO`, and `NOOP_ABORT` against baseline defaults; measure actions spent per game and win rate.

### **Hypothesis 2 (High Confidence): Prompt Contamination & Token Pressure**
* **Mechanism**: Injecting `ARC3_SYSTEM_PROMPT_PREFIX` forces the model into strict rule-following ("Never waste steps in jitter loops", "Trigger an orthogonal exploratory move"), reducing the model's natural emergent spatial reasoning and consuming tokens that trigger earlier `HYSTERESIS_TRIM` truncation.
* **Verification Test**: Remove `ARC3_SYSTEM_PROMPT_PREFIX` while keeping other settings identical, evaluate level completion rate.

### **Hypothesis 3 (Medium Confidence): Hysteresis Trimming Threshold Shift**
* **Mechanism**: Setting `ARC3_HYSTERESIS_TRIM_TOKENS` to 57,000 / 62,000 alters the context eviction boundary for SGLang KV caching, potentially evicting critical early-game state observations.
* **Verification Test**: Revert hysteresis trim tokens to harness default.

### **Hypothesis 4 (Low-Medium Confidence): Run-to-Run Interactive Variance**
* **Mechanism**: Environment level seeds vary per evaluation, and temperature-based LLM sampling introduces ±1.5 point volatility across submissions.
* **Verification Test**: Run 3 identical iterations with identical seeds on local public games to measure exact empirical variance bounds.
