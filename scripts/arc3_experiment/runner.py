#!/usr/bin/env python3
"""
ARC-AGI-3 experiment runner (public games, local scoring by the official `arc-agi` toolkit).

Purpose: measure agents with repeated seeds so differences can be separated from noise.
Every (game, seed) run gets its OWN scorecard, so per-run scores are not cumulative.

Usage (on a machine that can reach the ARC servers, or with games already downloaded):
    pip install arc-agi
    export ARC_API_KEY=...            # never commit this; an anonymous key is used if unset
    python scripts/arc3_experiment/runner.py --arm random --games ls20,ft09,vc33 \
        --seeds 0-4 --max-actions 400 --out results/arc3/random.jsonl
    python scripts/arc3_experiment/runner.py --arm mypkg.myagent:MyAgent --all-games ...

Agent interface (duck-typed):
    class MyAgent:
        def __init__(self, game_id: str, seed: int): ...
        def act(self, obs) -> tuple[GameAction, dict | None]
            # obs is the toolkit's FrameDataRaw: .state, .levels_completed, .win_levels,
            # .available_actions (ints), ._frame / frame data
Score definition (read from arc-agi 0.9.9 scorecard.py, not assumed):
    per level  = min((baseline_actions / actions_taken)^2 * 100, 115), 0 if not completed
    per game   = level-index-weighted mean of level scores, capped at the share of weight
                 held by levels with a nonzero score. Reasoning/tool calls are not actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from arcengine import GameAction, GameState


class RandomAgent:
    """Arm A0: uniform over the actions the game currently offers. The floor every other arm must beat."""

    def __init__(self, game_id: str, seed: int):
        self.rng = random.Random(f"{game_id}:{seed}")

    def act(self, obs) -> Tuple[GameAction, Optional[Dict[str, Any]]]:
        avail = [a for a in (obs.available_actions or []) if a != GameAction.RESET.value]
        if not avail:
            return GameAction.ACTION1, None
        action = GameAction.from_id(self.rng.choice(avail))
        if action.is_complex():
            return action, {"x": self.rng.randrange(64), "y": self.rng.randrange(64)}
        return action, None


def load_agent_class(spec: str) -> Callable[[str, int], Any]:
    if spec == "random":
        return RandomAgent
    mod, _, cls = spec.partition(":")
    if not cls:
        raise SystemExit("--arm must be 'random' or 'package.module:ClassName'")
    return getattr(importlib.import_module(mod), cls)


def run_episode(env, agent, max_actions: int) -> Dict[str, Any]:
    """Play one game. Counts only environment steps as actions. Pure duck-typing so it is unit-testable."""
    obs = env.reset()
    obs = obs if obs is not None else env.observation_space
    steps = resets = 0
    t0 = time.perf_counter()
    while steps < max_actions and obs is not None:
        if obs.state == GameState.WIN:
            break
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            steps += 1
            resets += 1
            continue
        action, data = agent.act(obs)
        obs = env.step(action, data) if data else env.step(action)
        steps += 1
    final = obs
    return {
        "env_steps": steps,
        "resets": resets,
        "final_state": getattr(getattr(final, "state", None), "value", str(getattr(final, "state", None))),
        "levels_completed": int(getattr(final, "levels_completed", 0) or 0),
        "win_levels": int(getattr(final, "win_levels", 0) or 0),
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="'random' or 'package.module:ClassName'")
    ap.add_argument("--arm-name", default=None, help="label stored in the log (default: --arm)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--games", help="comma separated game ids, e.g. ls20,ft09")
    g.add_argument("--all-games", action="store_true", help="every public game the toolkit lists")
    ap.add_argument("--seeds", default="0-2", help="e.g. 0-4 or 0,1,7")
    ap.add_argument("--max-actions", type=int, default=400)
    ap.add_argument("--environments-dir", default="environment_files")
    ap.add_argument("--offline", action="store_true",
                    help="use only games already on disk in --environments-dir (no network)")
    ap.add_argument("--out", required=True, help="JSONL output (appended; one line per game x seed)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    import arc_agi  # imported late so unit tests can run without the toolkit's network side effects

    mode = arc_agi.OperationMode.OFFLINE if args.offline else arc_agi.OperationMode.NORMAL
    arc = arc_agi.Arcade(operation_mode=mode, environments_dir=args.environments_dir)
    games = ([e.game_id.split("-", 1)[0] for e in arc.get_environments()] if args.all_games
             else [g.strip() for g in args.games.split(",") if g.strip()])
    if not games:
        print("No games available. Check network/API key or --environments-dir.", file=sys.stderr)
        return 2

    agent_cls = load_agent_class(args.arm)
    arm_name = args.arm_name or args.arm
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        from importlib.metadata import version as _pkg_version
        toolkit_version = _pkg_version("arc-agi")
    except Exception:
        toolkit_version = "unknown"
    meta = {"git": _git_rev(), "toolkit": toolkit_version,
            "max_actions": args.max_actions}
    cfg_hash = hashlib.sha256(json.dumps({**meta, "arm": arm_name}, sort_keys=True).encode()).hexdigest()[:12]

    with open(args.out, "a", encoding="utf-8") as fh:
        for game in games:
            for seed in parse_seeds(args.seeds):
                sc_id = arc.create_scorecard(tags=[arm_name, f"seed{seed}"])
                env = arc.make(game, seed=seed, scorecard_id=sc_id)
                if env is None:
                    rec = {"game": game, "seed": seed, "arm": arm_name, "error": "make() returned None"}
                else:
                    try:
                        rec = run_episode(env, agent_cls(game, seed), args.max_actions)
                        card = arc.get_scorecard(sc_id)
                        runs = [r for lst in (card.environments if card else []) for r in lst.runs]
                        rec["score"] = float(runs[-1].score) if runs else 0.0
                        rec["toolkit_actions"] = int(runs[-1].actions) if runs else None
                        rec["level_scores"] = runs[-1].level_scores if runs else None
                        rec["level_baseline_actions"] = runs[-1].level_baseline_actions if runs else None
                        rec.update(game=game, seed=seed, arm=arm_name)
                    except Exception as e:  # keep going; the error is data
                        rec = {"game": game, "seed": seed, "arm": arm_name,
                               "error": f"{type(e).__name__}: {e}"[:300]}
                rec.update(meta, config_hash=cfg_hash, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{arm_name}] {game} seed={seed} score={rec.get('score')} "
                      f"levels={rec.get('levels_completed')}/{rec.get('win_levels')} "
                      f"steps={rec.get('env_steps')} err={rec.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
