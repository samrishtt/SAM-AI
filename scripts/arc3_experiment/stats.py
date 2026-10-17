#!/usr/bin/env python3
"""
Noise-aware comparison of ARC-AGI-3 runs.

Input: JSONL files from runner.py (rows with arm, game, seed, score; rows with `error` are counted, not scored).
Output: per-arm mean score with a cluster bootstrap CI (resampling GAMES, then seeds within game), the
seed-to-seed noise floor, and for each pair of arms the paired difference with its CI.

Rule baked in: an arm is called "better" only if the 95% CI of the paired difference excludes 0.
One Kaggle private-set score is ONE sample; never conclude from a single submission difference.

Usage: python scripts/arc3_experiment/stats.py results/arc3/*.jsonl [--baseline random]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple


def load(paths: List[str]):
    rows, errors = [], defaultdict(int)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r or "score" not in r:
                    errors[r.get("arm", "?")] += 1
                    continue
                rows.append(r)
    return rows, dict(errors)


def by_arm_game(rows) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["arm"]][r["game"]].append(float(r["score"]))
    return out


def _boot_mean(game_scores: Dict[str, List[float]], rng: random.Random) -> float:
    games = list(game_scores)
    picks = [rng.choice(games) for _ in games]
    per_game = []
    for g in picks:
        s = game_scores[g]
        per_game.append(st.mean(rng.choice(s) for _ in s))
    return st.mean(per_game)


def ci(vals: List[float], lo=2.5, hi=97.5) -> Tuple[float, float]:
    v = sorted(vals)
    k = lambda p: v[min(len(v) - 1, max(0, int(round(p / 100 * (len(v) - 1)))))]
    return k(lo), k(hi)


def arm_summary(ag: Dict[str, List[float]], n_boot=2000, seed=0):
    rng = random.Random(seed)
    mean = st.mean(st.mean(s) for s in ag.values())
    boots = [_boot_mean(ag, rng) for _ in range(n_boot)]
    # seed-to-seed noise: SD of the overall score when each seed index is treated as a full run
    max_k = min(len(s) for s in ag.values())
    per_seed = [st.mean(s[i] for s in ag.values()) for i in range(max_k)] if max_k >= 2 else []
    return {
        "mean": mean, "ci95": ci(boots), "games": len(ag),
        "min_seeds": max_k, "seed_sd": (st.stdev(per_seed) if len(per_seed) >= 2 else None),
    }


def paired_diff(a: Dict[str, List[float]], b: Dict[str, List[float]], n_boot=2000, seed=1):
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    rng = random.Random(seed)
    # difference of per-game means; bootstrap over games, then over seeds within each game
    point = st.mean(st.mean(a[g]) - st.mean(b[g]) for g in shared)
    boots = []
    for _ in range(n_boot):
        picks = [rng.choice(shared) for _ in shared]
        boots.append(st.mean(
            st.mean(rng.choice(a[g]) for _ in a[g]) - st.mean(rng.choice(b[g]) for _ in b[g]) for g in picks))
    lo, hi = ci(boots)
    return {"diff": point, "ci95": (lo, hi), "games": len(shared), "distinguishable_from_zero": lo > 0 or hi < 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", default=None, help="arm name to compare every other arm against")
    args = ap.parse_args(argv)
    rows, errors = load(args.files)
    arms = by_arm_game(rows)
    if not arms:
        print("no scored rows")
        return 1
    print(f"{'arm':28} {'mean':>7} {'95% CI':>17} {'games':>5} {'seeds':>5} {'seed-SD':>8} {'errors':>6}")
    summ = {}
    for arm, ag in sorted(arms.items()):
        s = arm_summary(ag)
        summ[arm] = s
        sd = "n/a" if s["seed_sd"] is None else f"{s['seed_sd']:.2f}"
        print(f"{arm:28} {s['mean']:7.2f} [{s['ci95'][0]:6.2f},{s['ci95'][1]:6.2f}] "
              f"{s['games']:5d} {s['min_seeds']:5d} {sd:>8} {errors.get(arm, 0):6d}")
    names = sorted(arms)
    base = args.baseline
    pairs = [(n, base) for n in names if n != base] if base else [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if pairs:
        print("\npaired differences (A - B), cluster bootstrap over games and seeds")
    for a, b in pairs:
        r = paired_diff(arms[a], arms[b])
        if r is None:
            print(f"  {a} vs {b}: no shared games")
            continue
        verdict = ("DISTINGUISHABLE from zero" if r["distinguishable_from_zero"]
                   else "NOT distinguishable from noise")
        print(f"  {a} - {b}: {r['diff']:+.2f}  CI95 [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
              f"({r['games']} games)  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
