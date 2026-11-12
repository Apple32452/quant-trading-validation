import math
import unittest

import numpy as np

from quant_models import (
    annualized_sharpe,
    black_scholes_closed_form_call,
    black_scholes_finite_difference_call,
    chronological_split,
    prices_to_daily_returns,
    select_and_evaluate_rule,
    simulate_blackjack,
)


class BlackjackTests(unittest.TestCase):
    def test_shapes_reproducibility_and_valid_bets(self):
        first = simulate_blackjack(n_sessions=4, n_decks=5, seed=6, batch_decks=2)
        second = simulate_blackjack(n_sessions=4, n_decks=5, seed=6, batch_decks=2)
        self.assertEqual(first.true_counts.shape, (4, 5, 7))
        self.assertEqual(first.adaptive_cumulative.shape, (4, 35))
        np.testing.assert_array_equal(first.true_counts, second.true_counts)
        np.testing.assert_array_equal(first.adaptive_cumulative, second.adaptive_cumulative)
        np.testing.assert_array_equal(first.flat_cumulative, second.flat_cumulative)

        # Correct per-hand payoffs are +/- stake, not outcome + stake.
        adaptive_steps = np.diff(
            first.adaptive_cumulative, axis=1,
            prepend=np.zeros((4, 1), dtype=np.int64),
        )
        flat_steps = np.diff(
            first.flat_cumulative, axis=1,
            prepend=np.zeros((4, 1), dtype=np.int64),
        )
        bets = np.where(first.true_counts > 1.0, 500, 50).reshape(4, -1)
        np.testing.assert_array_equal(np.abs(adaptive_steps), bets)
        np.testing.assert_array_equal(np.abs(flat_steps), 50)

    def test_deck_zero_sum(self):
        deck = np.repeat([1, 0, -1], [20, 12, 20])
        self.assertEqual(int(deck.sum()), 0)
        self.assertEqual(deck.size, 52)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            simulate_blackjack(n_sessions=0)


class BlackScholesTests(unittest.TestCase):
    def test_numerical_matches_closed_form_and_boundaries(self):
        spots, values = black_scholes_finite_difference_call()
        theoretical = black_scholes_closed_form_call(100, 100, 0.05, 0.2, 1)
        numerical = float(np.interp(100.0, spots, values))
        self.assertAlmostEqual(theoretical, 10.4505835721856, places=7)
        self.assertLess(abs(numerical - theoretical), 0.08)
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(values[-1], 200 - 100 * math.exp(-0.05), places=8)
        self.assertTrue(np.all(np.diff(values) >= -1e-8))

    def test_unstable_time_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "unstable"):
            black_scholes_finite_difference_call(time_steps=10)


class BacktestTests(unittest.TestCase):
    def test_sample_sharpe_formula(self):
        returns = np.array([0.01, -0.02, 0.03, 0.02])
        expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(annualized_sharpe(returns), expected)

    def test_zero_volatility_and_split(self):
        self.assertTrue(math.isnan(annualized_sharpe(np.zeros(10))))
        left, right = chronological_split(np.arange(10), cut=6)
        np.testing.assert_array_equal(left, np.arange(6))
        np.testing.assert_array_equal(right, np.arange(6, 10))
        with self.assertRaises(ValueError):
            chronological_split(np.arange(10), cut=9)

    def test_price_returns(self):
        out = prices_to_daily_returns(np.array([100.0, 110.0, 99.0]))
        np.testing.assert_allclose(out, [0.10, -0.10], atol=1e-12)

    def test_only_winner_sees_holdout(self):
        seen = {"A": [], "B": []}

        def a(r):
            seen["A"].append(len(r))
            return r

        def b(r):
            seen["B"].append(len(r))
            return -r

        returns = np.array([
            0.03, -0.01, 0.02, -0.005, 0.02, -0.007,
            0.01, -0.01, 0.02, -0.02,
        ])
        result = select_and_evaluate_rule(returns, {"A": a, "B": b}, cut=6)
        self.assertEqual(result.selected_rule, "A")
        self.assertEqual(seen["A"], [6, 4])
        self.assertEqual(seen["B"], [6])
        self.assertEqual(result.training_periods, 6)
        self.assertEqual(result.holdout_periods, 4)

    def test_invalid_rule_output(self):
        with self.assertRaises(ValueError):
            select_and_evaluate_rule(
                np.arange(10) / 1_000, {"bad": lambda r: r[:-1]}, cut=6
            )


if __name__ == "__main__":
    unittest.main()
