import csv
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fetch_market_data import verified_download
from market_backtest import (
    PriceHistory,
    Strategy,
    annualized_sharpe_excess,
    load_adjusted_prices,
    paired_block_bootstrap_mean_ci,
    period_mask,
    period_metrics,
    portfolio_returns,
    run_market_research,
    target_weights,
    training_only_selection,
)


class HistoricalDataTests(unittest.TestCase):
    def create_prices(
        self,
        dest: Path,
        *,
        gap_at: int | None = None,
        missing_yield_at: int | None = None,
    ) -> None:
        with dest.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["Date", "SPY", "^IRX"])
            for i in range(270):
                today = date(2020, 1, 1) + timedelta(days=i)
                close = "" if i == gap_at else f"{100 + i * 0.10:.4f}"
                writer.writerow([today, close, "" if i == missing_yield_at else 4.0])
            # A final trailing missing close is permitted in the source.
            writer.writerow([date(2020, 1, 1) + timedelta(days=270), "", 4.0])

    def test_loader_validation_and_trailing_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "prices.csv"
            self.create_prices(file)
            result = load_adjusted_prices(file, first_date=date(2020, 1, 1))
            self.assertEqual(len(result.dates), 270)
            self.assertEqual(result.dates[-1], date(2020, 1, 1) + timedelta(days=269))
            np.testing.assert_allclose(result.daily_cash_return, 0.04 / 252)
            self.assertEqual(result.source_sha256, hashlib.sha256(file.read_bytes()).hexdigest())
            self.create_prices(file, gap_at=100)
            with self.assertRaisesRegex(ValueError, "Internal missing"):
                load_adjusted_prices(file, first_date=date(2020, 1, 1))
            self.create_prices(file, missing_yield_at=100)
            carried = load_adjusted_prices(file, first_date=date(2020, 1, 1))
            self.assertEqual(carried.carried_forward_cash_yield_days, 1)
            self.assertAlmostEqual(carried.annual_cash_yield_pct[100], 4.0)

    def test_pinned_download_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "source.csv"
            dst = Path(folder) / "result.csv"
            src.write_bytes(b"simple public data")
            dst.write_bytes(b"old content")
            checksum = hashlib.sha256(src.read_bytes()).hexdigest()
            self.assertEqual(
                verified_download(dst, url=src.as_uri(), expected_sha256=checksum),
                checksum,
            )
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verified_download(dst, url=src.as_uri(), expected_sha256="0" * 64)
            self.assertEqual(dst.read_bytes(), src.read_bytes())

    def test_known_exchange_holiday_skipped_but_other_gaps_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "holidays.csv"
            begin = date(2025, 7, 1)
            with file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Date", "SPY", "^IRX"])
                for i in range(400):
                    today = begin + timedelta(days=i)
                    close = "" if today == date(2026, 5, 25) else f"{100 + i / 10:.2f}"
                    writer.writerow([today, close, 4.0])
            loaded = load_adjusted_prices(file, first_date=begin)
            self.assertEqual(loaded.skipped_known_exchange_holidays, 1)
            self.assertNotIn(date(2026, 5, 25), loaded.dates)
            self.assertEqual(len(loaded.dates), 399)


class TradingLogicTests(unittest.TestCase):
    def test_signals_do_not_look_ahead(self):
        prices = np.array([100., 102., 103., 101., 99., 98., 100., 101., 102.])
        for strategy in (
            Strategy("m", "momentum", 2),
            Strategy("r", "reversion", 2),
            Strategy("v", "volatility", 3, 0.1),
        ):
            with self.subTest(strategy=strategy.name):
                before = target_weights(prices, strategy)
                future_altered = prices.copy()
                future_altered[7:] *= 10.0
                after = target_weights(future_altered, strategy)
                np.testing.assert_allclose(before[:7], after[:7])
                self.assertTrue(np.all((before >= 0) & (before <= 1)))

    def test_close_signal_executes_only_at_following_close(self):
        prices = np.array([100., 101., 104., 105., 98., 99., 100.])
        weights = target_weights(prices, Strategy("momentum_2", "momentum", 2))
        # Price[2] / price[0] reveals an uptrend AFTER close 2.
        # We cannot capture close[2]->close[3]; it executes at close 3.
        self.assertEqual(weights[2], 0.0)
        self.assertEqual(weights[3], 1.0)
        # The reversal becomes known after close 4, executed at close 5.
        self.assertEqual(weights[5], 0.0)

    def test_costs_cash_and_one_day_lag(self):
        prices = np.array([100., 110., 99., 108.9])
        cash = np.array([0.001, 0.001, 0.001])
        weights = np.array([0., 1., 0.])
        net, traded = portfolio_returns(prices, cash, weights, cost_bps=10)
        np.testing.assert_allclose(net, [0.001, -0.101, 0.0])
        np.testing.assert_allclose(traded, [0, 1, 1])
        with self.assertRaises(ValueError):
            portfolio_returns(prices, cash, weights, cost_bps=-1)

    def test_training_winner_ignores_holdout(self):
        training = np.array([True, True, True, True, False, False])
        cash = np.zeros(6)
        rules = {
            "A": np.array([0.01, 0.02, -0.005, 0.015, -0.50, -0.50]),
            "B": np.array([-0.01, 0.01, -0.01, 0.01, 0.50, 0.50]),
        }
        winner, scores = training_only_selection(rules, cash, training)
        self.assertEqual(winner, "A")
        self.assertGreater(scores["A"], scores["B"])
        altered = {name: np.r_[returns[:4], -returns[4:]] for name, returns in rules.items()}
        self.assertEqual(training_only_selection(altered, cash, training)[0], winner)

    def test_metrics_and_bootstrap(self):
        daily = np.array([0.001, 0.002, -0.001, 0.0] * 10)
        cash = np.full_like(daily, 0.0001)
        metric = period_metrics(daily, cash, np.zeros(40), np.ones(40))
        self.assertEqual(metric["days"], 40)
        self.assertLess(metric["max_drawdown"], 0)
        self.assertGreater(annualized_sharpe_excess(daily, cash), 0)
        first = paired_block_bootstrap_mean_ci(
            daily - cash, repetitions=120, block_size=5, seed=7
        )
        second = paired_block_bootstrap_mean_ci(
            daily - cash, repetitions=120, block_size=5, seed=7
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci95_low"], first["ci95_high"])

    def test_full_pipeline_uses_only_development_years_for_folds(self):
        rng = np.random.default_rng(99)
        begin = date(2009, 1, 1)
        end = date(2025, 1, 1)
        dates = []
        day = begin
        while day < end:
            if day.weekday() < 5:
                dates.append(day)
            day += timedelta(days=1)
        returns = rng.normal(0.0002, 0.009, size=len(dates) - 1)
        prices = 100 * np.r_[1.0, np.cumprod(1.0 + returns)]
        hist = PriceHistory(
            dates=tuple(dates),
            adjusted_close=prices,
            annual_cash_yield_pct=np.full(len(dates), 3.5),
        )
        report = run_market_research(hist)
        self.assertEqual(report["source"]["matches_pinned_snapshot"], False)
        self.assertEqual(len(report["walk_forward_development_only"]), 5)
        self.assertEqual(report["walk_forward_development_only"][-1]["test_year"], 2022)
        self.assertEqual(report["selected"]["final_holdout"]["days"], int(
            period_mask(hist.dates, date(2023, 1, 1), date(2100, 1, 1)).sum()
        ))
        self.assertEqual(len(report["final_cost_sensitivity"]), 4)
        self.assertFalse(math.isnan(report["selected"]["final_holdout"]["annualized_return"]))


if __name__ == "__main__":
    unittest.main()
