#!/usr/bin/env python3
"""Small, registered tests for the independent evaluation analysis path."""

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "keel_evaluation_analysis", ROOT / "experiments" / "analyze_evaluation.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def row(cluster, policy, repeat=0, **values):
    result = {
        "workload_instance_id": cluster,
        "family": "synthetic",
        "seed": int(cluster[-1]),
        "graph_fingerprint": cluster,
        "grant_profile": "tight_b",
        "pressure": 0.75,
        "budget_fraction": 0.5,
        "frontier_fraction": 0.5,
        "repeat": repeat,
        "policy": policy,
    }
    result.update(values)
    return result


class EvaluationIntegrityTests(unittest.TestCase):
    def test_clustered_paired_delta_is_not_row_bootstrap(self):
        rows = []
        for cluster in ("graph-1", "graph-2"):
            for repeat in (0, 1):
                rows += [
                    row(cluster, "hybrid", repeat, joint_success=True),
                    row(cluster, "conservative", repeat, joint_success=False),
                ]
        result = ANALYSIS.paired_cluster_delta(
            rows, "hybrid", "conservative", "joint_success"
        )
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["mean"], 1.0)

    def test_identical_policy_has_zero_cluster_effect(self):
        rows = []
        for cluster in ("graph-1", "graph-2"):
            rows += [
                row(cluster, "hybrid", joint_success=True),
                row(cluster, "conservative", joint_success=True),
            ]
        result = ANALYSIS.paired_cluster_delta(
            rows, "hybrid", "conservative", "joint_success"
        )
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])

    def test_missing_numeric_effect_is_excluded(self):
        rows = [
            row("graph-1", "hybrid", wait_b_benefit_us=10),
            row("graph-1", "conservative", wait_b_benefit_us=4),
            row("graph-2", "hybrid", wait_b_benefit_us=None),
            row("graph-2", "conservative", wait_b_benefit_us=4),
        ]
        result = ANALYSIS.paired_cluster_delta(
            rows, "hybrid", "conservative", "wait_b_benefit_us"
        )
        self.assertEqual(result["clusters"], 1)
        self.assertEqual(result["mean"], 6.0)

    def test_commit_only_metrics_exclude_rejection_sentinels(self):
        rows = [
            row("graph-1", "hybrid", committed=True, a_slowdown_ratio=2.0),
            row("graph-1", "conservative", committed=False, a_slowdown_ratio=0.0),
            row("graph-2", "hybrid", committed=False, a_slowdown_ratio=0.0),
            row("graph-2", "conservative", committed=False, a_slowdown_ratio=0.0),
        ]
        result = ANALYSIS.paired_cluster_delta(
            rows, "hybrid", "conservative", "a_slowdown_ratio"
        )
        self.assertEqual(result["clusters"], 0)
        self.assertIsNone(result["mean"])

    def test_bootstrap_is_deterministic_and_p99_is_censored_for_small_n(self):
        values = [1, 2, 3, 4]
        self.assertEqual(ANALYSIS.mean_ci(values), ANALYSIS.mean_ci(values))
        self.assertEqual(ANALYSIS.quantiles(values)["p99"], None)
        self.assertIn("p99_note", ANALYSIS.quantiles(values))

    def test_svg_coordinates_stay_inside_declared_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            bar = Path(directory) / "bar.svg"
            lines = Path(directory) / "lines.svg"
            scatter = Path(directory) / "scatter.svg"
            ANALYSIS.svg_bar(bar, "bar", ["a", "b"], [0.1, 0.9], "rate")
            ANALYSIS.svg_lines(lines, "lines", ["0", "1"], [("p", [0.1, 0.9])], "rate")
            ANALYSIS.svg_scatter(scatter, "scatter", [("p", 1.0, -2.0)], "x", "y")
            for path, width, height in ((bar, 1000, 560), (lines, 1000, 600),
                                        (scatter, 1000, 600)):
                text = path.read_text()
                self.assertIn(f'viewBox="0 0 {width} {height}"', text)
                for number in re.findall(r'\b(?:x|x1|x2|y|y1|y2|cx|cy)="(-?[0-9]+(?:\.[0-9]+)?)"', text):
                    value = float(number)
                    self.assertGreaterEqual(value, 0.0, (path, number))
                    self.assertLessEqual(value, max(width, height), (path, number))


if __name__ == "__main__":
    unittest.main()
