"""Tests for the ARC-3 experiment harness using fake environments (no network, no real games)."""
import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace

try:
    from arcengine import GameAction, GameState
    HAVE_ARCENGINE = True
except Exception:  # pragma: no cover
    HAVE_ARCENGINE = False

if HAVE_ARCENGINE:
    from scripts.arc3_experiment import runner, stats


class FakeEnv:
    """Needs `target` correct moves in a row; wrong move -> GAME_OVER. Mirrors the wrapper's duck type."""

    def __init__(self, target=2):
        self.target, self.pos, self.state, self.steps = target, 0, GameState.NOT_PLAYED, 0
        self.levels = 0

    def _obs(self):
        return SimpleNamespace(state=self.state, levels_completed=self.levels, win_levels=1,
                               available_actions=[1, 2])

    def reset(self):
        self.pos, self.state = 0, GameState.NOT_FINISHED
        return self._obs()

    @property
    def observation_space(self):
        return self._obs()

    def step(self, action, data=None):
        self.steps += 1
        if action == GameAction.RESET:
            return self.reset()
        if action == GameAction.ACTION1:
            self.pos += 1
            if self.pos >= self.target:
                self.state, self.levels = GameState.WIN, 1
        else:
            self.state = GameState.GAME_OVER
        return self._obs()


@unittest.skipUnless(HAVE_ARCENGINE, "arcengine not installed")
class TestRunner(unittest.TestCase):
    def test_episode_counts_resets_and_stops_on_win(self):
        class Always1:
            def act(self, obs):
                return GameAction.ACTION1, None
        res = runner.run_episode(FakeEnv(target=3), Always1(), max_actions=50)
        self.assertEqual(res["final_state"], "WIN")
        self.assertEqual(res["env_steps"], 3)
        self.assertEqual(res["levels_completed"], 1)

    def test_game_over_triggers_reset_that_counts_as_a_step(self):
        class Wrong:
            n = 0
            def act(self, obs):
                Wrong.n += 1
                return (GameAction.ACTION2 if Wrong.n == 1 else GameAction.ACTION1), None
        res = runner.run_episode(FakeEnv(target=2), Wrong(), max_actions=50)
        self.assertEqual(res["final_state"], "WIN")
        self.assertEqual(res["resets"], 1)
        self.assertEqual(res["env_steps"], 4)  # wrong, RESET, right, right

    def test_action_cap_is_respected(self):
        class Wrong:
            def act(self, obs):
                return GameAction.ACTION2, None
        res = runner.run_episode(FakeEnv(target=2), Wrong(), max_actions=10)
        self.assertEqual(res["env_steps"], 10)
        self.assertNotEqual(res["final_state"], "WIN")

    def test_random_agent_is_deterministic_per_seed_and_avoids_reset(self):
        obs = SimpleNamespace(available_actions=[0, 1, 2, 6])
        a1 = [runner.RandomAgent("g", 3).act(obs) for _ in range(1)]
        a2 = [runner.RandomAgent("g", 3).act(obs) for _ in range(1)]
        self.assertEqual(a1[0][0], a2[0][0])
        ag = runner.RandomAgent("g", 4)
        for _ in range(200):
            act, data = ag.act(obs)
            self.assertNotEqual(act, GameAction.RESET)
            if act.is_complex():
                self.assertTrue(0 <= data["x"] < 64 and 0 <= data["y"] < 64)

    def test_parse_seeds(self):
        self.assertEqual(runner.parse_seeds("0-2,7"), [0, 1, 2, 7])


TOY_GAME = """
from arcengine import ARCBaseGame, Level, Sprite, GameAction

def _lvl():
    return Level(sprites=[Sprite([[1]], name="p", x=0, y=0)], grid_size=(8, 8))

class Toy1(ARCBaseGame):
    def __init__(self, seed=0):
        super().__init__(game_id="toy1", levels=[_lvl(), _lvl()], available_actions=[1, 2], seed=seed)
        self.k = 0

    def on_set_level(self, level):
        self.k = 0

    def step(self):
        a = self.action.id
        if a == GameAction.ACTION1:
            self.k += 1
            if self.k >= 2:
                self.next_level()
        elif a == GameAction.ACTION2:
            self.lose()
        self.complete_action()
"""


@unittest.skipUnless(HAVE_ARCENGINE, "arcengine not installed")
class TestEndToEndOffline(unittest.TestCase):
    """Runs the real `arc-agi` toolkit on a locally authored toy game (NOT a real ARC-AGI-3 game)."""

    def test_main_offline_writes_scored_rows(self):
        try:
            import arc_agi  # noqa: F401
        except Exception:
            self.skipTest("arc-agi toolkit not installed")
        with tempfile.TemporaryDirectory() as d:
            gdir = os.path.join(d, "environment_files", "toy1")
            os.makedirs(gdir)
            with open(os.path.join(gdir, "metadata.json"), "w") as f:
                json.dump({"game_id": "toy1-abc12345", "title": "toy", "class_name": "Toy1",
                           "baseline_actions": [2, 2]}, f)
            with open(os.path.join(gdir, "toy1.py"), "w") as f:
                f.write(TOY_GAME)
            out = os.path.join(d, "res", "random.jsonl")
            rc = runner.main(["--arm", "random", "--arm-name", "A0_random", "--games", "toy1", "--seeds", "0-3",
                              "--max-actions", "300", "--offline",
                              "--environments-dir", os.path.join(d, "environment_files"), "--out", out])
            self.assertEqual(rc, 0)
            rows = [json.loads(l) for l in open(out)]
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertNotIn("error", r, r)
                self.assertIn("score", r)
                self.assertEqual(r["arm"], "A0_random")
                self.assertLessEqual(r["score"], 100.0)
            # scorecards are per run: one run's score must not accumulate other runs' levels
            self.assertTrue(all(r["levels_completed"] <= r["win_levels"] for r in rows))


@unittest.skipUnless(HAVE_ARCENGINE, "arcengine not installed")
class TestStats(unittest.TestCase):
    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def _rows(self, arm, mu, sd, games=8, seeds=5, seed=0):
        rng = random.Random(seed)
        return [{"arm": arm, "game": f"g{g}", "seed": s, "score": max(0.0, rng.gauss(mu, sd))}
                for g in range(games) for s in range(seeds)]

    def test_identical_arms_are_not_distinguishable(self):
        rows = self._rows("A", 20, 5, seed=1) + self._rows("B", 20, 5, seed=2)
        arms = stats.by_arm_game(stats.load([self._write(rows)])[0])
        r = stats.paired_diff(arms["A"], arms["B"])
        self.assertFalse(r["distinguishable_from_zero"], r)

    def test_real_gap_is_detected(self):
        rows = self._rows("A", 30, 3, seed=1) + self._rows("B", 15, 3, seed=2)
        arms = stats.by_arm_game(stats.load([self._write(rows)])[0])
        r = stats.paired_diff(arms["A"], arms["B"])
        self.assertTrue(r["distinguishable_from_zero"], r)
        self.assertGreater(r["diff"], 10)

    def test_error_rows_are_counted_not_scored(self):
        rows = self._rows("A", 20, 1, games=2, seeds=2) + [{"arm": "A", "game": "g9", "seed": 0, "error": "x"}]
        loaded, errors = stats.load([self._write(rows)])
        self.assertEqual(errors, {"A": 1})
        self.assertEqual(len(loaded), 4)

    def test_summary_reports_seed_noise(self):
        rows = self._rows("A", 20, 5, games=6, seeds=6)
        arms = stats.by_arm_game(stats.load([self._write(rows)])[0])
        s = stats.arm_summary(arms["A"])
        self.assertIsNotNone(s["seed_sd"])
        self.assertTrue(s["ci95"][0] <= s["mean"] <= s["ci95"][1])


if __name__ == "__main__":
    unittest.main()
