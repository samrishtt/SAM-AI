# SAM-AI Workspace Rules

Follow these rules in every session:

## 1. Core Operating Rules
1. **Evidence or it didn't happen**: Every result needs the exact command and raw output saved under `evidence/`. Not run = `NOT_RUN`. No hardcoded success, no mock results.
2. **Never print, commit or store secrets**: Use the `credentials` skill. Report secret findings by file/commit/line only.
3. **NEVER run `kaggle competitions submit`, accept competition rules, publish/open-source anything, change licenses, post to GitHub/Kaggle/Discord, close PRs, or spend money**: Prepare the exact command and put it under `NEEDS USER`. Running `kaggle kernels push/status/output` on `samrishb` is allowed ONLY when I've approved that specific run.
4. **Never modify `baselines/ARC3_GOLDEN_BASELINE/`**: Work on copies.
5. **Competition integrity**: No hardcoded per-game solutions, no probing or inferring hidden games from submission feedback, no network use inside the Kaggle submission.
6. **Licensing**: Before reusing any third-party code/idea, record its license + attribution in `research/licenses.md`. Third-party work must be credited as theirs in all docs and the paper.
7. **Labels**: `VERIFIED` (command+output+date) / `INTERNAL_TEST` / `EXPERIMENTAL` / `PLANNED` / `NOT_RUN`. No marketing words.
8. **One change per experiment**: Keep negative results. Git commit before each run.
9. **Calendar (through Nov 8)**: Freeze Oct 28, Kaggle closes Nov 2, paper Nov 8. All non-ARC work (substrate, web app, coding agent, base-vs-LoRA benchmarks) is PARKED until after Nov 8.

## 2. Hard-Won Lessons & Corrected Failure Modes
10. **Zero Speculative Causality**: Never claim an algorithmic change caused a regression or gain without controlled ablation proof. (Disproven when V9 reverted Cell 7 heuristics and dropped further to 23.15; real causes were single-pass stochastic noise and SGLang queue congestion).
11. **Score & Historical Provenance Integrity**: Never misattribute historical scores as recent regressions (the 3.62 score was from Sept 3, not an October regression; peak is 29.21 on Oct 3; 28.29 was Oct 6). Ranks are verified as 385 (ARC-3) and 641 (ARC-2); expunge unverified estimates like "~91".
12. **Explicit Per-Run Push Gate**: High-level plan approval does NOT authorize remote cloud execution. Never run `kaggle kernels push` without stating the exact slug, diff, and receiving explicit per-run approval for that specific push.
13. **Variance Awareness Before Submission**: Live gateway submissions run on a single pass ($N=1$) with native $\pm 3-5$ point stochastic swings. Never evaluate a solver on a single live run; always validate locally on 10 games with $N=4$ multi-pass statistics ($\mu, \sigma, CV$) first.
14. **Concurrency-Queue Capacity Match**: Client concurrency (`bm.solver.concurrency`) must NEVER exceed SGLang's engine capacity (`MAXREQ = 10`). Launching 120 streams into a 10-stream engine caused 36 HTTP read timeouts and premature game termination.
