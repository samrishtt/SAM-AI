# Kaggle Submissions Raw Evidence Directory

This directory stores raw, unmodified command-line outputs for Kaggle submissions and entered competitions.

## File Manifest & Notes

- `entered_competitions_2026-10-13.txt`:
  Raw output of `kaggle competitions list --group entered` executed on 2026-10-13. Shows all competitions entered by the authenticated account.

- `arc-prize-2026-arc-agi-3_submissions_2026-10-13.txt`:
  Raw output of `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` on 2026-10-13.
  Shows all submission records for ARC-AGI-3 (highest verified: 29.21; recent runs: 28.29, 26.04, 25.76, 23.94, 9.50).

- `arc-prize-2026-arc-agi-2_submissions_2026-10-13.txt`:
  Raw output of `kaggle competitions submissions -c arc-prize-2026-arc-agi-2` on 2026-10-13.
  Shows submission records for ARC-AGI-2 (top scores: 30.56, 30.14, 29.72, 29.44, 29.17, 28.61, 28.06, 26.94).

- `arc-prize-2026-paper-track_submissions_2026-10-13.txt`:
  Raw output of `kaggle competitions submissions -c arc-prize-2026-paper-track` on 2026-10-13 (0 submissions found).

### Disambiguation Note for 2026-10-12 Records
- `arc3_submissions_2026-10-12.txt`:
  Captured submissions for `arc-prize-2026-arc-agi-3` on 2026-10-12.
- `arc2_submissions_2026-10-12.txt`:
  Captured output from `kaggle competitions submissions -c arc-prize-2024`.
  **Clarification**: The slug `arc-prize-2024` was the 2024 competition (ARC-1), which returned a 403 HTTP Forbidden error because that competition was not active or entered under this profile. The actual active ARC-2 competition is `arc-prize-2026-arc-agi-2`, which is successfully queried and recorded in `arc-prize-2026-arc-agi-2_submissions_2026-10-13.txt`.
