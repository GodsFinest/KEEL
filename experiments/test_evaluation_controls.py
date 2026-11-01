#!/usr/bin/env python3
"""Regression test for the work-conserving shared controls."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvaluationControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = None
        if len(sys.argv) > 1:
            candidate = Path(sys.argv[1]).resolve()
            if candidate.is_file():
                cls.evaluator = candidate
        if cls.evaluator is None:
            raise unittest.SkipTest("CTest supplies the keel_eval executable path")

    def test_wait_control_admits_b_before_a_finishes_and_censors_failures(self):
        evaluator = self.evaluator
        with tempfile.TemporaryDirectory(prefix="keel-control-test-") as directory:
            output = Path(directory) / "control.jsonl"
            command = [
                str(evaluator), "--tier", "control-test", "--seed-lo", "1",
                "--seed-hi", "1", "--repeats", "1", "--families", "chain",
                "--policies", "conservative", "--pressures", "0.75",
                "--budgets", "1.0", "--frontiers",
                "0.50", "--grant-profile", "tight_b", "--old-grant-kb",
                "32", "--b-grant-kb", "16", "--out", str(output),
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["wait_b_admitted"])
        self.assertLess(row["wait_b_admission_us"], row["wait_a_completion_us"])
        self.assertTrue(row["wait_b_ok"])
        self.assertGreater(row["wait_b_completion_us"], 0)
        self.assertEqual(row["wait_b_deadline_us"], row["wait_b_completion_us"])
        self.assertTrue(row["drain_b_admitted"])
        self.assertGreaterEqual(row["drain_b_admission_us"],
                                row["drain_a_completion_us"])

    def test_joint_success_does_not_depend_on_reuse_or_overlap_labels(self):
        evaluator = self.evaluator
        with tempfile.TemporaryDirectory(prefix="keel-control-test-") as directory:
            output = Path(directory) / "control.jsonl"
            command = [
                str(evaluator), "--tier", "control-test", "--seed-lo", "1",
                "--seed-hi", "1", "--repeats", "1", "--families", "chain",
                "--policies", "conservative", "--pressures", "0.75",
                "--budgets", "1.0", "--frontiers", "0.50",
                "--grant-profile", "roomy_b", "--old-grant-kb", "32",
                "--b-grant-kb", "32", "--out", str(output),
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["joint_success"],
                         row["committed"] and row["a_ok"] and row["b_ok"] and
                         row["a_correct"] and row["b_correct"] and
                         row["invariants"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
