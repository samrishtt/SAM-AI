#!/usr/bin/env python3
"""
Comprehensive Run Evaluator for SAM-AI ARC-AGI-3 Solvers.
Parses benchmark output artifacts (.per_game_cards.pkl, benchmark.json, diagnostics.html, logs).
Calculates:
  1. Official Best-of-4 score (what Kaggle scores on offline validation).
  2. Diagnostic all-pass mean score.
  3. Per-game and aggregate pass variance (sigma, CV).
  4. HTTP read timeouts and server queue errors.
  5. Latency distributions.
"""

import os
import sys
import json
import pickle
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def evaluate(output_dir):
    p = Path(output_dir)
    print("=" * 80)
    print(f"EVALUATING SAM-AI RUN ARTIFACTS IN: {p.resolve()}")
    print("=" * 80)

    cards_path = p / ".per_game_cards.pkl"
    bench_path = p / "benchmark.json"
    diag_path = p / "diagnostics.html"
    serve_log = p / "serve.log"

    # 1. Parse Game Cards
    if not cards_path.exists():
        print(f"[!] Warning: {cards_path} not found. Checking if run completed...")
        return None

    with open(cards_path, "rb") as f:
        cards = pickle.load(f)

    print(f"\n{'Game ID':<16} | {'Pass 0':<8} | {'Pass 1':<8} | {'Pass 2':<8} | {'Pass 3':<8} | {'Best':<8} | {'Mean':<8} | {'StdDev':<8} | {'CV':<6}")
    print("-" * 88)

    best_scores = []
    pass_scores_all = []
    stds = []

    for gid, v in sorted(cards.items()):
        s = v[0]
        parts = s.split("|")[3:]
        scores = []
        for part in parts:
            try:
                score_val = float(part.split(":")[-1])
                scores.append(score_val)
            except ValueError:
                pass

        if not scores:
            continue

        best = max(scores)
        mean_p = sum(scores) / len(scores)
        var_p = sum((x - mean_p) ** 2 for x in scores) / len(scores) if len(scores) > 1 else 0.0
        std_p = var_p ** 0.5
        cv_p = (std_p / mean_p) if mean_p > 0 else 0.0

        best_scores.append(best)
        pass_scores_all.append(scores)
        stds.append(std_p)

        p_strs = [f"{x:<8.2f}" for x in scores]
        while len(p_strs) < 4:
            p_strs.append(f"{'N/A':<8}")

        print(f"{gid:<16} | {p_strs[0]} | {p_strs[1]} | {p_strs[2]} | {p_strs[3]} | {best:<8.2f} | {mean_p:<8.2f} | {std_p:<8.2f} | {cv_p:<6.2f}")

    print("-" * 88)
    n_games = len(best_scores)
    official_score = sum(best_scores) / n_games if n_games > 0 else 0.0
    all_flat = [x for sub in pass_scores_all for x in sub]
    diag_mean = sum(all_flat) / len(all_flat) if all_flat else 0.0
    mean_std = sum(stds) / len(stds) if stds else 0.0

    print(f"Games Evaluated                 : {n_games}")
    print(f"OFFICIAL BEST-OF-4 MEAN SCORE   : {official_score:.2f}")
    print(f"DIAGNOSTIC ALL-PASS MEAN SCORE  : {diag_mean:.2f}")
    print(f"AVERAGE PER-GAME STOCHASTIC STD : {mean_std:.2f}")

    # 2. Parse Server Errors & Timeouts
    timeout_count = 0
    connection_errors = 0
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8", errors="ignore") as f:
            diag_text = f.read()
            timeout_count += diag_text.count("Read timed out")
            timeout_count += diag_text.count("ReadTimeout")
            connection_errors += diag_text.count("HTTPConnectionPool")
            connection_errors += diag_text.count("ConnectionError")

    for log_file in p.glob("*.log"):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lt = f.read()
            timeout_count += lt.count("Read timed out")
            timeout_count += lt.count("ReadTimeout")
            connection_errors += lt.count("HTTPConnectionPool")

    print(f"\n--- SGLang & HTTP Gateway Health ---")
    print(f"HTTP Read Timeouts              : {timeout_count}")
    print(f"Connection Pool Errors          : {connection_errors}")

    results = {
        "n_games": n_games,
        "official_best_of_4": round(official_score, 2),
        "diagnostic_mean": round(diag_mean, 2),
        "mean_std_per_game": round(mean_std, 2),
        "http_read_timeouts": timeout_count,
        "connection_errors": connection_errors
    }
    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "predictions/b2_exp1_output"
    evaluate(target)
