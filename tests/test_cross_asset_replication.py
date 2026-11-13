"""Synthetic-only tests; the registered historical outcomes remain unseen."""

from datetime import date, timedelta
import unittest

import numpy as np

from cross_asset_replication import (
    BENCHMARK,
    BOOTSTRAP_REPETITIONS,
    EXPECTED_FIRST,
    EXPECTED_LAST,
    SYMBOLS,
    TEST_FIRST,
    _benchmark_enter_at_test_start,
    _verify_histories,
    holm_two_sided_adjusted,
    paired_circular_block_ci97_5,
    run_replication,
)
from factor_alpha import FactorArchive
from market_backtest import PriceHistory, period_mask


SYNTHETIC_SHA = "test_fixture_not_market_data"


def fake_histories() -> dict[str, PriceHistory]:
    dates = []
    current = EXPECTED_FIRST
    while current <= EXPECTED_LAST:
        if current.weekday() < 5 and current != date(2026, 5, 25):
            dates.append(current)
        current += timedelta(days=1)
    dt = tuple(dates)
    cash_pct = np.full(len(dt), 3.0)
    result = {}
    for offset, symbol in enumerate((BENCHMARK, *SYMBOLS)):
        rng = np.random.default_rng(500 + offset)
        daily = rng.normal(0.0003, 0.01, len(dt) - 1)
        prices = 100 * np.r_[1.0, np.cumprod(1.0 + daily)]
        result[symbol] = PriceHistory(
            dates=dt,
            adjusted_close=prices,
            annual_cash_yield_pct=cash_pct.copy(),
            source_sha256=SYNTHETIC_SHA,
            symbol=symbol,
        )
    return result


def fake_factors(dates: tuple[date, ...]) -> dict[str, FactorArchive]:
    rng = np.random.default_rng(47)
    observations = {"ff3": {}, "ff5": {}, "mom": {}}
    for day in dates:
        vals = rng.normal(0, 0.005, 6)
        common = {
            "Mkt-RF": float(vals[0]),
            "SMB": float(vals[1]),
            "HML": float(vals[2]),
            "RF": 0.0001,
        }
        observations["ff3"][day] = common
        observations["ff5"][day] = {
            **common, "RMW": float(vals[3]), "CMA": float(vals[4])
        }
        observations["mom"][day] = {"Mom": float(vals[5])}
    return {
        name: FactorArchive(
            name=name,
            dates_to_factors=points,
            sha256="synthetic-not-an-official-archive",
            first_date=dates[0],
            last_date=dates[-1],
        )
        for name, points in observations.items()
    }


class RegisteredMethodTests(unittest.TestCase):
    def test_exact_paired_bootstrap_confidence_and_determinism(self):
        diffs = np.linspace(-0.0005, 0.0015, 300)
        first = paired_circular_block_ci97_5(diffs)
        second = paired_circular_block_ci97_5(diffs)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence"], 0.975)
        self.assertEqual(first["repetitions"], BOOTSTRAP_REPETITIONS)
        self.assertAlmostEqual(first["annualized_arithmetic_mean_gap"], 0.126)
        self.assertLess(first["ci97_5_low"], first["annualized_arithmetic_mean_gap"])
        self.assertGreater(first["ci97_5_high"], first["annualized_arithmetic_mean_gap"])

    def test_holm_correction_prevents_secondary_winner_cherry_picking(self):
        p = holm_two_sided_adjusted({"IWM": 0.03, "EFA": 0.001})
        self.assertAlmostEqual(p["IWM"], 0.03)
        self.assertAlmostEqual(p["EFA"], 0.002)
        with self.assertRaises(ValueError):
            holm_two_sided_adjusted({"IWM": float("nan")})

    def test_buy_hold_charges_entry_cost_on_first_test_day_only(self):
        dates = tuple(date(2022, 12, 20) + timedelta(days=k) for k in range(350))
        mask = period_mask(dates, TEST_FIRST, date(2024, 1, 1))
        net, traded, weights = _benchmark_enter_at_test_start(
            np.ones(len(dates)) * 100, np.zeros(len(dates) - 1), mask
        )
        self.assertEqual(int(np.count_nonzero(traded)), 1)
        self.assertAlmostEqual(net[np.flatnonzero(mask)[0]], -0.0005)
        self.assertTrue(np.all(weights == 1))

    def test_requires_identical_full_vintage_and_sessions(self):
        samples = fake_histories()
        dates = _verify_histories(samples, SYNTHETIC_SHA)
        self.assertEqual(dates[0], EXPECTED_FIRST)
        self.assertEqual(dates[-1], EXPECTED_LAST)
        with self.assertRaisesRegex(ValueError, "unpinned"):
            _verify_histories(samples, "another_hash")
        missing = {**samples, "EFA": PriceHistory(
            dates=samples["EFA"].dates[1:],
            adjusted_close=samples["EFA"].adjusted_close[1:],
            annual_cash_yield_pct=samples["EFA"].annual_cash_yield_pct[1:],
            symbol="EFA", source_sha256=SYNTHETIC_SHA,
        )}
        with self.assertRaisesRegex(ValueError, "non-identical"):
            _verify_histories(missing, SYNTHETIC_SHA)

    def test_one_fixed_rule_end_to_end_synthetic_not_real_data(self):
        histories = fake_histories()
        factors = fake_factors(histories[BENCHMARK].dates)
        result = run_replication(
            histories, factors, expected_source_sha=SYNTHETIC_SHA
        )
        self.assertIn("PRE_REGISTERED", result["status"])
        self.assertEqual(result["previously_committed_protocol"]["primary"], "QQQ")
        self.assertEqual(set(result["results_by_symbol"]), set(SYMBOLS))
        self.assertEqual(result["fixed_method"]["cost_bps"], 5.0)
        self.assertEqual(result["fixed_method"]["strategy"]["lookback"], 20)
        self.assertEqual(result["fixed_method"]["bootstrap"],
                         "5,000 paired circular 20-session blocks; seed 20261114")
        expected_days = int(period_mask(histories[BENCHMARK].dates,
                                        TEST_FIRST, date(2026, 8, 11)).sum())
        self.assertEqual(result["data"]["test_sessions"], expected_days)
        for name in SYMBOLS:
            row = result["results_by_symbol"][name]
            self.assertEqual(row["ff5_plus_momentum_alpha"]["hac_lags"], 20)
            self.assertIn("HAC lag-20", row["ff5_plus_momentum_alpha"]["uncertainty_note"])
            self.assertEqual(row["ff5_plus_momentum_alpha"]["observations"], expected_days)
            self.assertEqual(len(row["cost_sensitivity"]), 4)
            self.assertEqual(row["net_strategy_minus_spy_buy_hold"]["confidence"], 0.975)
            self.assertEqual(row["same_etf_buy_and_hold_metrics"]["days"], expected_days)
        status = result["primary_QQQ_two_required_conditions"]
        self.assertEqual(
            status["both_conditions_met"],
            status["qqq_strategy_vs_spy_CI_lower_positive"]
            and status["qqq_strategy_ff5_momentum_alpha_CI_lower_positive"],
        )


if __name__ == "__main__":
    unittest.main()
