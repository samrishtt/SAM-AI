# Pre-Registered Experiment Specification: EXP-SAM-01-SCAFFOLDING-EVAL

> **STATUS: NOT_RUN — PENDING EXPLICIT USER APPROVAL**  
> **Execution Lock:** ACTIVE. No execution, GPU quota, paid API, or notebook push permitted.  
> **Target System:** SAM-AI Sovereign Agent (`sam_ai.agent.arc3_agent`)  
> **Baseline Policy:** Arm A serves as the internal control baseline. Third-party B2 scores (e.g. 70.89 max-of-4) are strictly excluded as baselines because they evaluate a different model and third-party solver.

---

## 1. Title & Identifier
* **Experiment ID:** `EXP-SAM-01-SCAFFOLDING-EVAL`
* **Title:** Controlled Ablation of SAM-AI Verification Guard and Failure Memory in Interactive ARC-AGI-3
* **Date Pre-Registered:** 2026-10-10
* **Author / Principal:** SAM-AI Engineering & Research Team

---

## 2. Research Hypothesis
**Hypothesis ($H_1$):** In interactive multi-turn grid environments, equipping the foundation model with a pre-execution verification guard (`enable_verification_guard=True`) and structured task-specific failure memory (`enable_failure_memory=True`) will:
1. Reduce the rate of ineffective actions (zero-spatial-delta moves / wall bumps) and illegal actions by at least 30%.
2. Eliminate immediate repetitions of fatal moves (`GAME_OVER` transitions).
3. Improve single-pass ($N=1$) action efficiency (measured as mean environment steps per level solved) compared to an unconstrained ReAct agent using the exact same foundation model and prompt.

**Null Hypothesis ($H_0$):** The verification guard and failure memory either provide no statistically significant improvement in action efficiency ($p \ge 0.05$) or increase prompt context overhead without reducing zero-delta actions.

---

## 3. Experimental Design (A/B Ablation)

Both arms share the **exact same**:
- Foundation model weights, quantization, and temperature ($T = 0.2$, top-p $= 0.95$).
- System prompt instructions and visual ASCII grid representation logic (`sam_ai.agent.representation`).
- Action budget: 50 environment steps maximum per game episode.
- Random seeds and initial game board states.
- Hardware and environment container.

### Arm A: Control Baseline (Minimal ReAct Loop)
* **Configuration:**
  - `enable_verification_guard = False`: Model outputs are parsed strictly with `ActionSchemaParser`. If syntactically valid (e.g. valid JSON with `action_name`), the action is dispatched directly to the environment without semantic checking or retry.
  - `enable_failure_memory = False`: Observations and past actions are recorded in simple linear history; no failure warnings or zero-delta warnings are injected into subsequent prompts.

### Arm B: Treatment (SAM-AI Scaffolded Loop)
* **Configuration:**
  - `enable_verification_guard = True`: Pre-execution validation verifies:
    1. If `is_game_over == True`, the action must be `RESET`. Any other action is rejected with immediate reflection feedback to the model before consuming an environment step.
    2. Coordinate bounds ($0 \le x, y \le 63$) are enforced.
    3. Maximum of 2 verification retries before fallback.
  - `enable_failure_memory = True`: `TaskFailureMemory` records:
    1. Zero-delta moves ($\Delta = \text{Frame}_t \oplus \text{Frame}_{t-1} = 0$).
    2. Fatal moves leading to `GAME_OVER`.
    3. Formats dynamic prompt warnings (e.g. `[BLOCKED ACTION WARNING] Do NOT repeat action X`).

---

## 4. Evaluation Suite & Split

To prevent benchmark overfitting and distinguish memorization from generalization, the evaluation suite is partitioned into:

### Development Suite (5 Games)
* Used for tuning retry prompt templates and schema verification logic:
  - Game IDs: `ls20`, `pw60`, `sc10`, `sp30`, `ft09` (from public ARC-AGI-3 offline suite).

### Held-Out Evaluation Suite (5 Games)
* Kept strictly uninspected during development; used for final evaluation:
  - Game IDs: `da29`, `bb08`, `ch01`, `ae12`, `re04`.

### Seeds & Replications
* **Seeds:** 3 independent seeds (`42`, `137`, `2026`).
* **Runs per game:** 1 pass per seed ($N = 1$).
* **Total Episodes per Arm:** $10 \text{ games} \times 3 \text{ seeds} = 30 \text{ episodes}$.
* **Total Experiment Episodes:** $30 \times 2 \text{ arms} = 60 \text{ episodes}$.

---

## 5. Metrics Specification

### Primary Metric
* **Action Efficiency ($H/A$):** Ratio of total levels completed to total environment actions spent across all 30 episodes:
  $$\text{Efficiency} = \frac{\sum_{i=1}^{30} \text{LevelsCompleted}_i}{\sum_{i=1}^{30} \text{ActionsTaken}_i}$$

### Secondary Diagnostic Metrics
1. **Single-Pass Win Rate:** Percentage of episodes reaching game completion ($N=1$).
2. **Zero-Delta Action Rate:** Percentage of actions where environment grid delta $\Delta = 0$ (e.g. walking into a wall).
3. **Repeated Fatal Action Rate:** Number of times the agent repeats an action that caused `GAME_OVER` within the same game.
4. **Verification Rejection Rate:** Number of model generations rejected by the pre-execution guard before reaching the environment.
5. **Token Overhead & Latency:** Mean prompt token length and mean inference latency (seconds per step).

---

## 6. Pre-Registered Decision Criteria

### Success Criterion (KEEP Component)
* Arm B achieves:
  1. A $\ge 30\%$ relative reduction in Zero-Delta Action Rate compared to Arm A ($p < 0.05$ via two-tailed Mann-Whitney U test).
  2. Zero instances of Repeated Fatal Actions.
  3. No degradation in Single-Pass Win Rate compared to Arm A ($\text{WinRate}_B \ge \text{WinRate}_A$).

### Rejection / Rollback Criterion (REJECT Component)
* Arm B is rejected and rolled back if:
  1. Verification retries cause infinite loops or increase step latency by $> 100\%$ without win rate gain.
  2. Zero-Delta Action Rate reduction is $< 15\%$.
  3. Single-Pass Win Rate on the held-out suite drops compared to Arm A.

---

## 7. Known Sources of Variance & Controls
1. **LLM Output Stochasticity:** Controlled by locking temperature to $T = 0.2$ and evaluating across 3 fixed random seeds.
2. **Environment Initialization:** ARC-AGI-3 environments have internal procedural variations; using identical seeds across Arm A and Arm B ensures pairwise matching.
3. **Prompt Length Drift:** Memory accumulation increases context size; capped at the 10 most recent transitions to prevent attention dilution.

---

## 8. Resource Budget & Prerequisites

### Resource Estimates
* **Total Game Steps:** $\sim 60 \text{ episodes} \times \le 30 \text{ steps} \approx 1,800 \text{ model inferences}$.
* **Estimated Runtime:**
  - On NVIDIA RTX Pro 6000 / L4 (FP8/4-bit 14B): $\sim 1.5 - 2.5 \text{ s/step} \approx 45 - 75 \text{ minutes}$.
* **Storage Footprint:** $< 15 \text{ MB}$ raw JSONL telemetry in `evidence/sam_ai_runs/`.

### Unresolved Prerequisites (Blocking Execution)
1. **Model Weights Access:** Local environment currently has only `config.json` (0 MB weights) and no GPU. Real inference requires running inside an approved GPU environment (e.g. Kaggle kernel with mounted model weights) or setting up an approved remote inference endpoint.
2. **Official ARC-AGI-3 Task Metadata:** Local directory `environment_files/` is absent.
3. **User Approval:** Explicit sign-off required prior to launching any remote or GPU-consuming execution.
