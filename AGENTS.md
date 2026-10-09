# SAM-AI Workspace Rules

Follow these rules in every session:

1. **Evidence or it didn't happen**: Every result needs the exact command and raw output saved under `evidence/`. Not run = `NOT_RUN`. No hardcoded success, no mock results.
2. **Never print, commit or store secrets**: Use the `credentials` skill. Report secret findings by file/commit/line only.
3. **NEVER run `kaggle competitions submit`, accept competition rules, publish/open-source anything, change licenses, post to GitHub/Kaggle/Discord, close PRs, or spend money**: Prepare the exact command and put it under `NEEDS USER`. Running `kaggle kernels push/status/output` on `samrishb` is allowed ONLY when I've approved that specific run.
4. **Never modify `baselines/ARC3_GOLDEN_BASELINE/`**: Work on copies.
5. **Competition integrity**: No hardcoded per-game solutions, no probing or inferring hidden games from submission feedback, no network use inside the Kaggle submission.
6. **Licensing**: Before reusing any third-party code/idea, record its license + attribution in `research/licenses.md`. Third-party work must be credited as theirs in all docs and the paper.
7. **Labels**: `VERIFIED` (command+output+date) / `INTERNAL_TEST` / `EXPERIMENTAL` / `PLANNED` / `NOT_RUN`. No marketing words.
8. **One change per experiment**: Keep negative results.
9. **Calendar (today Fri Oct 9)**: Freeze Oct 28, Kaggle closes Nov 2, paper Nov 8. All non-ARC work (substrate, web app, coding agent, base-vs-LoRA benchmarks) is PARKED until after Nov 8.
