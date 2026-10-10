# Pre-Registered Experiment Specification: EXP-SAM-01-SCAFFOLDING-EVAL (Revised)

> **STATUS: NOT_RUN — PENDING EXPLICIT USER APPROVAL**  
> **Execution Lock:** ACTIVE. No GPU quota, paid API, or notebook push permitted.  
> **Target System:** SAM-AI Sovereign Agent (`sam_ai.agent.arc3_agent`)  
> **Ablation Rigor:** Single-variable isolation. Only the Verification Guard is ablated in this experiment. Failure Memory is deferred to EXP-SAM-02.  
> **Baseline Policy:** Arm A serves as the internal control baseline. Third-party B2 scores (e.g. 70.89 max-of-4) are strictly excluded as baselines because they evaluate a different third-party system.

---

## 1. Title & Identifier
* **Experiment ID:** `EXP-SAM-01-SCAFFOLDING-EVAL`
* **Title:** Single-Variable Ablation of SAM-AI Pre-Execution Verification Guard in Interactive ARC-AGI-3
* **Date Pre-Registered:** 2026-10-10 (Revised for Single-Variable Isolation)
* **Author / Principal:** SAM-AI Engineering & Research Team

---

## 2. Research Hypothesis
**Hypothesis ($H_1$):** In interactive multi-turn grid environments, isolating and enabling the pre-execution verification guard (`enable_verification_guard=True`) while holding the base model, prompt template, action schema parser, and failure memory constant (`enable_failure_memory=False` for both arms) will:
1. Prevent illegal actions (out-of-bounds coordinates on `ACTION6`) and prohibited state actions (non-`RESET` actions when `is_game_over == True`) from reaching the environment.
2. Reduce wasted environment steps by providing immediate structured reflection feedback to the model before consuming game step budget.
3. Improve single-pass ($N=1$) action efficiency (measured as mean levels completed per environment action spent) compared to the unconstrained Arm A control.

**Null Hypothesis ($H_0$):** Pre-execution verification retries provide no statistically measurable benefit in action efficiency and increase prompt latency without reducing wasted actions.

---

## 3. Experimental Design (Sequential Single-Variable Isolation)

### Arm A: Control Baseline (Minimal Agent Loop)
* **Foundation Model:** Pre-quantized 4-bit 14B (`unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`) or genuine SAM-AI checkpoint-25 LoRA adapter.
* **Parser:** Strict [`ActionSchemaParser`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/sam_ai/agent/schema.py).
* **Verification Guard:** `enable_verification_guard = False`. Parsed actions are dispatched directly to the environment without semantic checking or reflection retries.
* **Failure Memory:** `enable_failure_memory = False`. Linear history only; zero-delta warnings are disabled.

### Arm B: Treatment (Verification Guard Only)
* **Foundation Model:** Identical model weights, quantization, and temperature ($T = 0.2$).
* **Parser:** Identical [`ActionSchemaParser`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/sam_ai/agent/schema.py).
* **Verification Guard:** `enable_verification_guard = True`. Pre-execution checks intercept:
  1. Mandatory `RESET` during `is_game_over == True`.
  2. Coordinate boundary enforcement ($0 \le x, y < 64$) on `ACTION6`.
  3. Bounded reflection retry (max 2 attempts) returning diagnostic error messages to the model before stepping the environment.
* **Failure Memory:** `enable_failure_memory = False` (HELD CONSTANT to isolate the verification guard).

### Subsequent Planned Experiment: Arm C (Deferred to EXP-SAM-02)
* **Design:** Tests `enable_failure_memory = True` on top of Arm B.
* **Sequencing:** Will **only** be scheduled and executed after Arm B's isolated performance and latency characteristics are fully analyzed and understood.

---

## 4. Evaluation Suite, Power & Sample Size

### Staged Rollout Protocol
1. **Stage 1 (Plumbing Smoke Test):** 1 public game (`ls20`), max 5 steps, 1 seed ($N=1$). Verifies model loading, inference latency, and log persistence. Must pass with zero unhandled exceptions before Stage 2.
2. **Stage 2 (Full Ablation):** 10 public offline ARC-AGI-3 games (5 Development, 5 Held-out) $\times$ 3 independent seeds (`42`, `137`, `2026`) = 30 episodes per arm ($N=1$).

### Statistical Power & Limitations
* With $N=30$ paired episodes per arm, the evaluation has adequate power ($\ge 80\%$) only to detect large effect sizes ($d \ge 0.75$). For moderate effect sizes ($d \approx 0.4–0.5$), statistical power is approximately $50–65\%$.
* **Reporting Constraint:** Results must report observed effect size and 95% bootstrap confidence intervals. Claims of definitive statistical significance will **not** be made unless the observed difference is large and non-overlapping.

---

## 5. Metrics Specification

### Primary Metric
* **Action Efficiency ($E$):** Total levels solved divided by total environment steps consumed across all 30 episodes:
  $$E = \frac{\sum_{i=1}^{30} \text{LevelsCompleted}_i}{\sum_{i=1}^{30} \text{ActionsTaken}_i}$$

### Secondary Diagnostic Metrics
1. **Illegal / Out-of-Bounds Interceptions:** Count of actions caught and corrected by the verification guard in Arm B vs. failed steps in Arm A.
2. **Game-Over Waste Rate:** Number of steps spent attempting non-`RESET` actions while in `GAME_OVER`.
3. **Single-Pass Win Rate ($N=1$):** Percentage of episodes reaching completion.
4. **Latency Overhead:** Mean step latency difference ($\Delta t = t_B - t_A$) induced by verification retry loops.

---

## 6. Pre-Registered Decision Criteria

### Success Criteria (KEEP Verification Guard)
* Arm B demonstrates:
  1. A reduction of $\ge 25\%$ in wasted (illegal/prohibited) environment actions.
  2. Zero instances of illegal action exceptions reaching the environment runtime.
  3. Mean Action Efficiency $E_B \ge E_A$ without regression in win rate on held-out tasks.

### Rollback / Rejection Criteria (REJECT Verification Guard)
* Rollback Arm B if:
  1. Verification retries cause degenerate repetitive loops, inflating step latency by $> 100\%$.
  2. Model fails to correct its action within 2 retries in $> 20\%$ of intercepted cases.
  3. Action Efficiency does not improve by at least $10\%$.

---

## 7. Execution Budget & Blockers

* **Compute Footprint:**
  - Stage 1 Smoke Test: < 2 minutes GPU runtime.
  - Stage 2 Full Ablation: ~45 minutes GPU runtime on Kaggle NVIDIA L4 / RTX Pro 6000 or 2x T4.
* **Storage Footprint:** < 15 MB raw JSONL telemetry in `evidence/sam_ai_runs/`.
* **Current Blockers:**
  1. Local environment has no GPU and no weights (`config.json` only).
  2. ARC-3 competition environment files missing on local machine.
  3. Execution locked awaiting user authorization.
