import csv
from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

from overfitting_study import (
    StudyConfig,
    one_sided_normal_tail_p,
    run_study,
    write_csv,
)


class RepeatedStudyTests(unittest.TestCase):
    def test_invalid_parameters(self):
        invalid = (
            {"repetitions": 1},
            {"training_days": 20},
            {"holdout_days": 20},
            {"candidate_counts": ()},
            {"candidate_counts": (1, 1)},
            {"candidate_counts": (0, 5)},
            {"daily_volatility": 0.0},
            {"planted_daily_alpha": 0.0},
            {"significance_level": 1.0},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                StudyConfig(**settings)

    def test_normal_approximation_tail_and_guard(self):
        self.assertAlmostEqual(one_sided_normal_tail_p(0.0, 756), 0.5)
        self.assertLess(one_sided_normal_tail_p(2.0, 756), 0.001)
        self.assertGreater(one_sided_normal_tail_p(-2.0, 756), 0.999)
        with self.assertRaises(ValueError):
            one_sided_normal_tail_p(float("nan"), 756)

    def test_reproducibility_and_nested_candidate_sets(self):
        config = StudyConfig(
            repetitions=12,
            training_days=80,
            holdout_days=60,
            candidate_counts=(12, 1, 5),
            seed=42,
        )
        first = run_study(config)
        second = run_study(config)
        self.assertEqual([row.candidates for row in first], [1, 5, 12])
        self.assertEqual([asdict(row) for row in first], [asdict(row) for row in second])
        for row in first:
            self.assertLessEqual(
                row.null_bonferroni_false_positive_rate,
                row.null_naive_false_positive_rate,
            )
            self.assertLessEqual(
                row.signal_correct_detection_rate + row.signal_false_selection_rate,
                1.0,
            )
            self.assertLessEqual(row.null_bonferroni_fpr_ci95_low, 1.0)
            self.assertGreaterEqual(row.null_bonferroni_fpr_ci95_high, 0.0)

    def test_global_null_inflation_and_multiple_testing_guard(self):
        """Predeclared broad tolerances, never benchmark-fitting a lucky seed."""
        config = StudyConfig(
            repetitions=240,
            seed=20261112,
            candidate_counts=(1, 5, 25, 100),
        )
        by_count = {row.candidates: row for row in run_study(config)}
        self.assertGreater(
            by_count[100].null_mean_optimism_gap,
            by_count[1].null_mean_optimism_gap + 0.5,
        )
        self.assertGreater(by_count[100].null_naive_false_positive_rate, 0.8)
        self.assertLess(by_count[100].null_bonferroni_false_positive_rate, 0.15)
        for n in by_count:
            self.assertLess(abs(by_count[n].null_mean_selected_holdout_sharpe), 0.25)
        self.assertGreater(by_count[100].signal_correct_detection_rate, 0.20)

    def test_csv_contains_experimental_metadata(self):
        config = StudyConfig(
            repetitions=5,
            training_days=50,
            holdout_days=50,
            candidate_counts=(1, 5),
            seed=71,
        )
        rows = run_study(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results" / "summary.csv"
            write_csv(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[1]["candidates"], "5")
            self.assertEqual(stored[1]["seed"], "71")
            self.assertIn("null_gap_ci95_low", stored[1])
            self.assertIn("signal_correct_detection_rate", stored[1])


if __name__ == "__main__":
    unittest.main()
