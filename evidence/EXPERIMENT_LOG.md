# SAM-AI Controlled Experiment Log

*Adheres strictly to Workspace Rules: 1 experiment at a time, evidence-backed, labels verified, git commit tracked.*

---

## Experiment 1: Concurrency & SGLang Queue Alignment (Baseline B2)

* **Experiment ID:** `EXP-01-CONCURRENCY-ALIGNMENT`
* **Status:** `COMPLETE`
* **Outcome:** `PROVISIONAL (INFRASTRUCTURE REPRODUCTION ONLY)`
* **Label:** `VERIFIED_REPRODUCTION`
* **Date / Time:** 2026-10-10 21:09 UTC+5:30
* **Git Commit:** `c2b756d` (`feat(b2): establish baseline B2 with Reki toggles and Exp 1 concurrency alignment`)
* **Kaggle Kernel ID:** `samrishb/arc-agi-3-sam-ai-solver-b2` (Version 1)
* **Kaggle URL:** `https://www.kaggle.com/code/samrishb/arc-agi-3-sam-ai-solver-b2`
* **Hardware:** NVIDIA RTX Pro 6000 (Blackwell Server Edition, 96 GB VRAM)
* **Target Stack:** Daniel Franzen Milestone 2 Solver (Intel Qwen 3.8B AutoRound + Albucino drafter, SGLang Pennyroyal, Tufa Duck Harness). NOTE: SAM-AI 14B model/adapter NOT running.
* **Single Variable Changed:** `bm.solver.concurrency = 10` (matching SGLang `CFG["MAXREQ"] = 10`), down from legacy `120`. `bm.solver.request_timeout = 180.0`.
* **Hypothesis:** Eliminates HTTP read timeouts on the 10-game offline suite by preventing client request flooding.
* **Measured Empirical Results (Audited):**
  - **Pass 0 (First Attempt, Like-for-Like vs Upstream):** **46.07** (vs. Upstream Franzen Single-Pass of **43.36**, delta: **+2.71**).
  - **Pass 1 (Second Attempt):** **55.79**
  - **Pass 2 (Third Attempt):** **60.36**
  - **Pass 3 (Fourth Attempt):** **64.48**
  - **Project-Specific Best-of-4 Maximum:** **70.89** (NOTE: Not an official RHAE score; max-of-4 metric over 10 demo games).
  - **Diagnostic All-Pass Mean (40 runs):** **57.00**
  - **HTTP Read Timeouts:** **0** (dropped from 16 down to zero on the 10-game run).
  - **Connection Pool Errors:** **0**.
* **Audit Verdict & Limitations:** **PROVISIONAL**.
  - While zero timeouts were achieved on 10 games, setting concurrency to 10 in a 110-game competition run causes task starvation under the 9-hour limit, directly contributing to the reported 9.50 competition score.
  - The claim that concurrency was the "sole cause" is retracted.
  - This result is strictly an infrastructure reproduction of Daniel Franzen's Milestone 2 solver, NOT an evaluation of SAM-AI's sovereign reasoning system.

---

## Experiment 2: Pre-Registered SAM-AI Verification Guard Ablation (Single-Variable Isolation)
* **Experiment ID:** `EXP-SAM-01-SCAFFOLDING-EVAL`
* **Status:** `NOT_RUN (PROPOSED PRE-REGISTERED EXPERIMENT — PENDING USER APPROVAL)`
* **Specification File:** `research/EXP_SAM_01_SCAFFOLDING_EVAL.md`
* **Target System:** SAM-AI Sovereign Agent (`sam_ai.agent.arc3_agent.SAMARC3Agent`)
* **Hardware Required:** GPU Host with mounted model weights (NVIDIA L4 / RTX Pro 6000 or 2x T4).
* **Control Arm (Arm A):** Base foundation model in minimal action loop with strict schema parsing (`verification_guard=False`, `failure_memory=False`).
* **Treatment Arm (Arm B):** Base foundation model with verification guard ONLY (`verification_guard=True`, `failure_memory=False`). Failure memory held constant.
* **Subsequent Step (Arm C / EXP-02):** Failure memory deferred to separate follow-up experiment after Arm B is evaluated.
* **Primary Metric:** Action Efficiency (levels completed per action spent across 30 episodes, single-pass $N=1$).
* **Pre-Registered Success Criterion:** $\ge 25\%$ reduction in illegal/wasted moves, zero unhandled action exceptions reaching environment, and non-regressing Action Efficiency.
* **Current Blocker:** Real model weights missing locally; execution locked awaiting user review and approval.



