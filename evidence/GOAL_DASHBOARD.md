# SAM-AI ARC-AGI-3 Goal Dashboard (Deadline: October 28, 2026)

**Mission:** Deliver a frozen ARC-AGI-3 candidate (or confirm golden baseline) chosen on measured evidence, with variance estimates and a draft paper outline.

---

## 1. Goal Done Criteria Tracking

| # | Done Criterion | Status | Current Evidence / Artifact | Remaining Action |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Raw submissions output & score table in `evidence/`** | **COMPLETE** | `evidence/arc_agi_3_submissions_raw.txt`<br>`evidence/SCORE_RECORD.md` | Keep updated with any new approved submission. |
| **2** | **Run-to-run variance for chosen baseline measured & recorded** | **COMPLETE** | Measured on B2 ($N=4$ passes, 10 games): Official Best-of-4 = **70.89**, All-Pass Mean = **57.00**, Per-game $\sigma = 12.33$ ($CV = 0.22$). | Baseline variance characterized; ready for ablation comparisons. |
| **3** | **Every experiment recorded as KEEP/REJECT with raw logs** | **IN PROGRESS** | `evidence/EXPERIMENT_LOG.md`<br>Exp 1 (`EXP-01-CONCURRENCY-ALIGNMENT`): **KEEP** (70.89, 0 timeouts). | Proceed with approved sequence: Exp 5 (Frame Differencing). |
| **4** | **Final candidate release folder (hashes, config, license, run script)** | **READY (AUTOMATED)** | `scripts/package_frozen_candidate.py` built.<br>Target: `export/arc3_frozen_candidate/`. | Package final winning candidate once freeze criteria are met. |
| **5** | **NEEDS USER list complete** | **ACTIVE** | Documented below. Zero unauthorized actions. | Maintain hard-stop boundary on credentials, rules, submits, spending. |

---

## 2. Hard-Stop Boundary & NEEDS USER Registry

The following actions are strictly gated under **Hard Stops** and require manual user intervention:
1. **Competition Rules Acceptance:** Must be accepted by the user on Kaggle (`https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules`).
2. **Competition Submissions:** `kaggle competitions submit` is NEVER run by the agent.
3. **Secret/Credential Handling:** No passwords, tokens, or private keys printed or stored.
4. **Publishing / Open-Sourcing:** Code is kept in private development until the user authorizes release.
5. **Spending Money:** No paid cloud services, GPU rentals, or subscriptions triggered.
