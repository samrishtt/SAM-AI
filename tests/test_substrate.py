#!/usr/bin/env python3
"""
Unit and Integration Tests for SAM-AI Intelligence Substrate
============================================================
Tests verifiers, models, strategies, failure memory, cognitive engine, and benchmark evaluator.
Enforces:
- Strict verification statuses: PASS, FAIL, NOT_RUN, ERROR, TIMEOUT
- Zero simulated benchmarks
- Failure taxonomy classification
"""

import os
import shutil
import tempfile
import unittest

from sam_ai.substrate import (
    VerificationStatus,
    FailureCategory,
    BenchmarkTask,
    DeterministicModelRunner,
    OpenAICompatibleModelRunner,
    MathDomainVerifier,
    CodeDomainVerifier,
    ArcDomainVerifier,
    ScienceDomainVerifier,
    AgentDomainVerifier,
    StructuredFailureMemory,
    DirectStrategy,
    BestOfNStrategy,
    SearchStrategy,
    PlanningStrategy,
    CognitiveEngine,
    HeuristicReasoningRouter,
    HeuristicFailureClassifier,
    BenchmarkEvaluator,
)


class TestSubstrateVerifiers(unittest.TestCase):
    def setUp(self):
        self.math_v = MathDomainVerifier()
        self.code_v = CodeDomainVerifier()
        self.arc_v = ArcDomainVerifier()
        self.sci_v = ScienceDomainVerifier()
        self.agent_v = AgentDomainVerifier()

    def test_math_verifier_exact_and_symbolic(self):
        # Exact string match
        res1 = self.math_v.verify("42", "42")
        self.assertEqual(res1.status, VerificationStatus.PASS)

        # Numerical equivalence
        res2 = self.math_v.verify("1/2", "0.5")
        self.assertEqual(res2.status, VerificationStatus.PASS)

        # LaTeX boxed extraction
        res3 = self.math_v.verify(r"The result is \boxed{3/4}", "0.75")
        self.assertEqual(res3.status, VerificationStatus.PASS)

        # Incorrect
        res4 = self.math_v.verify("100", "42")
        self.assertEqual(res4.status, VerificationStatus.FAIL)

    def test_code_verifier_execution(self):
        # Syntax error
        res_syn = self.code_v.verify("def bad_func(:\n    pass", "assert True")
        self.assertEqual(res_syn.status, VerificationStatus.FAIL)
        self.assertIn("SyntaxError", res_syn.details)

        # Missing tests -> NOT_RUN
        res_notrun = self.code_v.verify("def good(): return 1", None)
        self.assertEqual(res_notrun.status, VerificationStatus.NOT_RUN)

        # Valid passing code
        code_pass = "def add(a, b): return a + b"
        tests_pass = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        res_pass = self.code_v.verify(code_pass, tests_pass)
        self.assertEqual(res_pass.status, VerificationStatus.PASS)

        # Valid code failing assertion
        tests_fail = "assert add(2, 3) == 10"
        res_fail = self.code_v.verify(code_pass, tests_fail)
        self.assertEqual(res_fail.status, VerificationStatus.FAIL)

    def test_arc_verifier(self):
        g1 = [[1, 2], [3, 4]]
        g2 = [[1, 2], [3, 4]]
        g3 = [[1, 2], [3, 5]]
        g_diff_dim = [[1, 2, 0], [3, 4, 0]]

        # Pass
        res1 = self.arc_v.verify(g1, g2)
        self.assertEqual(res1.status, VerificationStatus.PASS)

        # Cell mismatch
        res2 = self.arc_v.verify(g1, g3)
        self.assertEqual(res2.status, VerificationStatus.FAIL)
        self.assertIn("Cell mismatch", res2.details)

        # Dimension mismatch
        res3 = self.arc_v.verify(g1, g_diff_dim)
        self.assertEqual(res3.status, VerificationStatus.FAIL)
        self.assertIn("Width mismatch", res3.details)

    def test_agent_verifier(self):
        state_goal = {"pos": (3, 4), "has_key": True}
        state_act = {"pos": (3, 4), "has_key": True}
        state_bad = {"pos": (3, 3), "has_key": True}

        res_ok = self.agent_v.verify(state_act, state_goal)
        self.assertEqual(res_ok.status, VerificationStatus.PASS)

        res_bad = self.agent_v.verify(state_bad, state_goal)
        self.assertEqual(res_bad.status, VerificationStatus.FAIL)


class TestSubstrateEngineAndMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mem_file = os.path.join(self.temp_dir, "test_failures.jsonl")
        self.memory = StructuredFailureMemory(storage_path=self.mem_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_model_and_engine(self):
        # Create deterministic model responses
        resp_map = {
            "Solve 2+2": "4",
            "Solve 5*5": "wrong_answer",
            "Write function square": "def square(x): return x * x",
        }
        model = DeterministicModelRunner(name="test-runner", response_map=resp_map)
        engine = CognitiveEngine(model=model, failure_memory=self.memory)

        # Task 1: Math pass
        t1 = BenchmarkTask(task_id="math-1", domain="math", prompt="Solve 2+2", ground_truth="4")
        rec1 = engine.solve(t1)
        self.assertEqual(rec1.verification.status, VerificationStatus.PASS)
        self.assertIsNone(rec1.failure)

        # Task 2: Math failure (should be classified and recorded in memory)
        t2 = BenchmarkTask(task_id="math-2", domain="math", prompt="Solve 5*5", ground_truth="25")
        rec2 = engine.solve(t2)
        self.assertEqual(rec2.verification.status, VerificationStatus.FAIL)
        self.assertIsNotNone(rec2.failure)
        self.assertEqual(rec2.failure.failure_category, FailureCategory.REASONING)

        # Check that failure was recorded in StructuredFailureMemory
        failures = self.memory.get_failures(domain="math")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].task_id, "math-2")
        self.assertEqual(self.memory.distribution()["reasoning"], 1)

    def test_best_of_n_strategy(self):
        # Model returns different attempts
        attempts = ["bad_answer", "42"]
        curr = [0]
        def gen(prompt):
            ans = attempts[curr[0] % len(attempts)]
            curr[0] += 1
            return ans

        model = DeterministicModelRunner(name="multi-runner", generator_fn=gen)
        engine = CognitiveEngine(model=model, failure_memory=self.memory)

        task = BenchmarkTask(task_id="bon-1", domain="math", prompt="What is the answer?", ground_truth="42")
        # With Best-of-N budget 2, it should find 42 on attempt 2 and pass
        strat = BestOfNStrategy()
        rec = engine.solve(task, strategy=strat, search_budget=2)
        self.assertEqual(rec.verification.status, VerificationStatus.PASS)
        self.assertEqual(rec.selected_candidate, "42")

    def test_evaluator_suite_run(self):
        resp_map = {
            "Q1": "10",
            "Q2": "20",
            "Q3": "wrong",
        }
        model = DeterministicModelRunner(name="eval-model", response_map=resp_map)
        engine = CognitiveEngine(model=model, failure_memory=self.memory)
        evaluator = BenchmarkEvaluator(engine=engine)

        tasks = [
            BenchmarkTask(task_id="t1", domain="math", prompt="Q1", ground_truth="10"),
            BenchmarkTask(task_id="t2", domain="math", prompt="Q2", ground_truth="20"),
            BenchmarkTask(task_id="t3", domain="math", prompt="Q3", ground_truth="30"),
        ]

        report = evaluator.run_suite("MathMiniSuite", tasks, save_report=False)
        self.assertEqual(report.total_tasks, 3)
        self.assertEqual(report.pass_count, 2)
        self.assertEqual(report.fail_count, 1)
        self.assertAlmostEqual(report.accuracy, 2.0 / 3.0, places=3)
        self.assertEqual(report.verification_rate, 1.0)
        self.assertIn("reasoning", report.failure_distribution)


if __name__ == "__main__":
    unittest.main()
