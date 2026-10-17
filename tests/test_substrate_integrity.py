#!/usr/bin/env python3
"""
Regression tests for evaluation-integrity and safety flaws in the substrate:
  1. MathDomainVerifier must never eval() model output.
  2. CodeDomainVerifier must not be spoofable by printing a marker / exiting early,
     and must not leak the parent environment (API keys) to candidate code.
  3. Strategies must not leak reference answers through feedback, and must not use the
     grader to choose the winner unless oracle=True (which is labeled pass@N).
  4. ArcDomainVerifier must return FAIL (not raise) on malformed grids.
"""
import os
import re
import shutil
import tempfile
import time
import unittest

from sam_ai.substrate import (
    BenchmarkTask,
    DeterministicModelRunner,
    CognitiveEngine,
    StructuredFailureMemory,
    MathDomainVerifier,
    CodeDomainVerifier,
    ArcDomainVerifier,
    SearchStrategy,
    BestOfNStrategy,
    BenchmarkEvaluator,
    VerificationStatus,
)


class TestMathVerifierSafety(unittest.TestCase):
    def setUp(self):
        self.v = MathDomainVerifier()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_text_is_never_executed(self):
        marker = os.path.join(self.tmp, "marker.txt")
        payload = f"__import__('pathlib').Path(r'{marker}').write_text('x')"
        res = self.v.verify(payload, "999")
        self.assertEqual(res.status, VerificationStatus.FAIL)
        self.assertFalse(os.path.exists(marker), "candidate text was executed")

    def test_dangerous_syntax_rejected(self):
        for bad in ["__import__('os')", "(lambda: 1)()", "[1,2][0]", "open('x')", "x.__class__", "1; 2"]:
            self.assertEqual(self.v.verify(bad, "1").status, VerificationStatus.FAIL, bad)

    def test_resource_bombs_rejected_fast(self):
        t0 = time.perf_counter()
        for bomb in ["9**9**9**9", "10**100000", "((10**1000)**1000)**1000", "9" * 5000]:
            self.assertEqual(self.v.verify(bomb, "1").status, VerificationStatus.FAIL, bomb[:30])
        self.assertLess(time.perf_counter() - t0, 5.0)

    def test_legitimate_equivalences_still_pass(self):
        cases = [
            ("1/2", "0.5"), (r"\boxed{3/4}", "0.75"), ("<answer>2^10</answer>", "1024"),
            ("x^2+2*x+1", "(x+1)^2"), ("sqrt(2)", "1.41421356"), ("-3", "-3.0"),
        ]
        for cand, truth in cases:
            res = self.v.verify(cand, truth)
            self.assertEqual(res.status, VerificationStatus.PASS, (cand, truth, res.details))

    def test_wrong_answers_fail_and_public_text_hides_reference(self):
        res = self.v.verify("41", "42")
        self.assertEqual(res.status, VerificationStatus.FAIL)
        self.assertIn("42", res.details)               # human log may contain it
        self.assertNotIn("42", res.public_details)     # model-facing text must not


class TestCodeVerifierIntegrity(unittest.TestCase):
    def setUp(self):
        self.v = CodeDomainVerifier()

    def test_fixed_marker_spoof_does_not_pass(self):
        code = "import sys\nprint('__VERIFICATION_PASSED__')\nsys.exit(0)\ndef add(a,b): return a+b"
        res = self.v.verify(code, "assert add(2, 3) == 10")
        self.assertNotEqual(res.status, VerificationStatus.PASS)

    def test_early_clean_exit_does_not_pass(self):
        code = "import sys\ndef add(a,b): return a+b\nsys.exit(0)"
        res = self.v.verify(code, "assert add(2, 3) == 10")
        self.assertEqual(res.status, VerificationStatus.FAIL)

    def test_real_pass_and_real_fail(self):
        code = "def add(a,b): return a+b"
        self.assertEqual(self.v.verify(code, "assert add(2,3)==5").status, VerificationStatus.PASS)
        self.assertEqual(self.v.verify(code, "assert add(2,3)==6").status, VerificationStatus.FAIL)

    def test_parent_secrets_not_visible_to_candidate(self):
        os.environ["SAM_TEST_FAKE_SECRET_TOKEN"] = "not-a-real-secret"
        try:
            res = self.v.verify(
                "def f(): return 1",
                "import os\nassert 'SAM_TEST_FAKE_SECRET_TOKEN' not in os.environ\nassert f() == 1",
            )
        finally:
            del os.environ["SAM_TEST_FAKE_SECRET_TOKEN"]
        self.assertEqual(res.status, VerificationStatus.PASS, res.details)

    def test_infinite_loop_times_out(self):
        res = self.v.verify("def f():\n    while True: pass", "f()", {"timeout_seconds": 1.0})
        self.assertEqual(res.status, VerificationStatus.TIMEOUT)

    def test_hidden_test_source_not_in_public_feedback(self):
        res = self.v.verify("def add(a,b): return a+b", "assert add(2, 3) == 12345")
        self.assertEqual(res.status, VerificationStatus.FAIL)
        self.assertNotIn("12345", res.public_details)
        vis = self.v.verify("def add(a,b): return a+b", "assert add(2, 3) == 12345", {"tests_visible": True})
        self.assertIn("12345", vis.public_details)


class TestArcVerifierRobustness(unittest.TestCase):
    def setUp(self):
        self.v = ArcDomainVerifier()
        self.truth = [[1, 2], [3, 4]]

    def test_malformed_candidates_fail_instead_of_raising(self):
        for bad in ([[1, 2], [3]], [[1, 2], [3, "x"]], [[True, 2], [3, 4]], [], [[]], "grid", [1, 2]):
            res = self.v.verify(bad, self.truth)
            self.assertEqual(res.status, VerificationStatus.FAIL, bad)

    def test_public_feedback_hides_target_values_and_shape(self):
        res = self.v.verify([[1, 2], [3, 9]], self.truth)
        self.assertEqual(res.status, VerificationStatus.FAIL)
        self.assertNotRegex(res.public_details, r"\d")
        res = self.v.verify([[1, 2, 3], [3, 4, 5]], self.truth)
        self.assertNotRegex(res.public_details, r"\d")


class TestStrategyEvaluationIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.memory = StructuredFailureMemory(storage_path=os.path.join(self.tmp, "f.jsonl"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _engine(self, fn):
        return CognitiveEngine(model=DeterministicModelRunner(name="stub", generator_fn=fn),
                               failure_memory=self.memory)

    def test_feedback_never_contains_hidden_answer(self):
        prompts = []

        def copycat(prompt):
            prompts.append(prompt)
            m = re.search(r"expected '([^']*)'", prompt)
            return m.group(1) if m else "wrong-first-guess"

        eng = self._engine(copycat)
        task = BenchmarkTask(task_id="m", domain="math", prompt="secret?", ground_truth="31415")
        for oracle in (False, True):
            prompts.clear()
            rec = eng.solve(task, strategy=SearchStrategy(oracle=oracle), search_budget=3)
            self.assertNotEqual(rec.verification.status, VerificationStatus.PASS, f"oracle={oracle}")
            self.assertFalse(any("31415" in p for p in prompts), f"answer leaked, oracle={oracle}")

    def test_default_best_of_n_is_not_oracle_selected(self):
        answers = iter(["a", "b", "c", "42"])
        eng = self._engine(lambda p: next(answers))
        task = BenchmarkTask(task_id="b", domain="math", prompt="q", ground_truth="42")
        rec = eng.solve(task, strategy=BestOfNStrategy(), search_budget=4)
        self.assertEqual(rec.selection_mode, "unsupervised")
        self.assertEqual(rec.verification.status, VerificationStatus.FAIL)  # lucky 4th sample not credited

    def test_oracle_best_of_n_is_labeled_pass_at_n(self):
        answers = iter(["a", "b", "c", "42"])
        eng = self._engine(lambda p: next(answers))
        task = BenchmarkTask(task_id="b", domain="math", prompt="q", ground_truth="42")
        rec = eng.solve(task, strategy=BestOfNStrategy(oracle=True), search_budget=4)
        self.assertEqual(rec.verification.status, VerificationStatus.PASS)
        self.assertEqual(rec.selection_mode, "oracle_pass_at_n")

    def test_majority_vote_selects_consistent_answer(self):
        answers = iter(["42", "7", "42"])
        eng = self._engine(lambda p: next(answers))
        task = BenchmarkTask(task_id="mv", domain="math", prompt="q", ground_truth="42")
        rec = eng.solve(task, strategy=BestOfNStrategy(), search_budget=3)
        self.assertEqual(rec.verification.status, VerificationStatus.PASS)
        self.assertEqual(rec.selection_mode, "unsupervised")

    def test_report_labels_oracle_runs_as_pass_at_n(self):
        answers = iter(["a", "42"] * 5)
        eng = self._engine(lambda p: next(answers))
        from sam_ai.substrate import HeuristicReasoningRouter
        eng.router = HeuristicReasoningRouter(oracle=True)
        ev = BenchmarkEvaluator(engine=eng)
        tasks = [BenchmarkTask(task_id="t1", domain="math", prompt="q", ground_truth="42")]
        rep = ev.run_suite("S", tasks, search_budget=2, save_report=False)
        self.assertEqual(rep.selection_mode, "oracle_pass_at_n")
        self.assertIn("pass@2", rep.metric_name)
        self.assertIn("oracle", rep.to_markdown())

    def test_report_for_default_run_is_plain_accuracy(self):
        eng = self._engine(lambda p: "42")
        ev = BenchmarkEvaluator(engine=eng)
        tasks = [BenchmarkTask(task_id="t1", domain="math", prompt="q", ground_truth="42")]
        rep = ev.run_suite("S", tasks, save_report=False)
        self.assertEqual(rep.selection_mode, "unsupervised")
        self.assertEqual(rep.metric_name, "accuracy")
        self.assertEqual(rep.strategy_name, "direct")


if __name__ == "__main__":
    unittest.main()
