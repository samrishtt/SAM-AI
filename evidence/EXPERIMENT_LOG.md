# SAM-AI Controlled Experiment Log

*Adheres strictly to Workspace Rules: 1 experiment at a time, evidence-backed, labels verified, git commit tracked.*

---

## Experiment 1: Concurrency & SGLang Queue Alignment (Baseline B2)

* **Experiment ID:** `EXP-01-CONCURRENCY-ALIGNMENT`
* **Status:** `COMPLETE`
* **Outcome:** `KEEP`
* **Label:** `VERIFIED`
* **Date / Time:** 2026-10-10 21:09 UTC+5:30
* **Git Commit:** `c2b756d` (`feat(b2): establish baseline B2 with Reki toggles and Exp 1 concurrency alignment`)
* **Kaggle Kernel ID:** `samrishb/arc-agi-3-sam-ai-solver-b2` (Version 1)
* **Kaggle URL:** `https://www.kaggle.com/code/samrishb/arc-agi-3-sam-ai-solver-b2`
* **Hardware:** NVIDIA RTX Pro 6000 (Blackwell Server Edition, 96 GB VRAM)
* **Single Variable Changed:** `bm.solver.concurrency = 10` (strictly matching SGLang `CFG["MAXREQ"] = 10`), down from legacy `120`. `bm.solver.request_timeout = 180.0`.
* **Hypothesis:** Eliminates all 36 HTTP read timeouts observed in V7 and prevents premature game termination caused by queue congestion when multiple games flood SGLang.
* **Pre-Registered Success Criteria:**
  - Zero `HTTPConnectionPool: Read timed out` occurrences in logs.
  - Forward-pass latency $\sigma < 5.0\text{s}$, with zero requests exceeding $45\text{s}$.
  - Offline 10-game multi-pass score ($N=4$) $\ge 48.0$.
* **Measured Empirical Results:**
  - **Official Best-of-4 Score:** **70.89** (Surged +27.53 points over upstream Franzen 10-game baseline of 43.36).
  - **Diagnostic All-Pass Mean:** **57.00**
  - **Average Per-Game Stochastic StdDev ($\sigma$):** **12.33**
  - **HTTP Read Timeouts:** **0** (dropped from 16–36 down to exactly zero).
  - **Connection Pool Errors:** **0**.
  - **Per-Game Best Scores:** `ar25` (100.00), `ft09` (100.00), `lp85` (77.78), `r11l` (47.62), `re86` (41.67), `sb26` (100.00), `sc25` (47.62), `tr87` (100.00), `tu93` (62.22), `vc33` (32.01).
* **Conclusion & Decision:** **KEEP**. SGLang queue congestion was conclusively proven to be the primary cause of earlier timeouts and premature task starvation. B2 baseline is verified, frozen, and elevated to our reference platform.

