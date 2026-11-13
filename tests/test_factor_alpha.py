"""Synthetic tests for the factor-alpha diagnostic (no live data required)."""

from datetime import date, timedelta
import hashlib
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from factor_alpha import (
    FactorArchive,
    _aligned_arrays,
    hac_regression,
    load_french_archive,
    run_factor_diagnostic,
)
from market_backtest import FINAL_FIRST, PriceHistory, period_mask


def business_days(start: date, count: int) -> tuple[date, ...]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def synthetic_factor_archives(
    dates: tuple[date, ...], *, seed: int = 44
) -> dict[str, FactorArchive]:
    rng = np.random.default_rng(seed)
    out = {name: {} for name in ("ff3", "ff5", "mom")}
    for d in dates:
        mkt = float(rng.normal(0.0002, 0.01))
        smb = float(rng.normal(0, 0.006))
        hml = float(rng.normal(0, 0.006))
        rwm = float(rng.normal(0, 0.006))
        cma = float(rng.normal(0, 0.006))
        mom = float(rng.normal(0, 0.006))
        base = {"Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": 0.0001}
        out["ff3"][d] = base
        out["ff5"][d] = {**base, "RMW": rwm, "CMA": cma}
        out["mom"][d] = {"Mom": mom}
    return {
        name: FactorArchive(
            name=name,
            dates_to_factors=observations,
            sha256="synthetic_fixture_not_an_official_pin",
            first_date=dates[0],
            last_date=dates[-1],
        )
        for name, observations in out.items()
    }


class FrenchArchiveTests(unittest.TestCase):
    def test_daily_csv_metadata_footer_and_percent_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.zip"
            dates = business_days(date(2020, 1, 1), 1200)
            body = (
                "Unofficial SYNTHETIC test metadata only\n"
                "Prepared for parser tests\n"
                ",Mkt-RF,SMB,HML,RF\n"
                + "\n".join(
                    f"{d:%Y%m%d},1.50,-2.00,3.00,0.02" for d in dates
                )
                + "\n\nAnnual Factors: ignore this footer\n"
            )
            with ZipFile(path, "w") as zf:
                zf.writestr("mock_daily.csv", body)
            archive = load_french_archive(path, "ff3")
            self.assertEqual(len(archive.dates_to_factors), 1200)
            self.assertAlmostEqual(archive.dates_to_factors[dates[0]]["Mkt-RF"], 0.015)
            self.assertAlmostEqual(archive.dates_to_factors[dates[0]]["RF"], 0.0002)
            self.assertEqual(archive.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "vintage changed"):
                load_french_archive(path, "ff3", verify_pinned=True)

    def test_duplicate_dates_and_missing_values_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.zip"
            dates = business_days(date(2014, 1, 1), 1200)
            rows = [f"{d:%Y%m%d},1,2,3,0.02" for d in dates]
            rows.append(rows[-1])
            with ZipFile(path, "w") as zf:
                zf.writestr("mock.csv", ",Mkt-RF,SMB,HML,RF\n" + "\n".join(rows))
            with self.assertRaisesRegex(ValueError, "duplicate factor date"):
                load_french_archive(path, "ff3")
            rows.pop()  # Test the missing marker without the duplicate.
            rows[-1] = f"{dates[-1]:%Y%m%d},-99.99,1,1,0.02"
            with ZipFile(path, "w") as zf:
                zf.writestr("mock.csv", ",Mkt-RF,SMB,HML,RF\n" + "\n".join(rows))
            with self.assertRaisesRegex(ValueError, "missing factor"):
                load_french_archive(path, "ff3")


class HACRegressionTests(unittest.TestCase):
    def test_known_planted_alpha_and_model_beta(self):
        rng = np.random.default_rng(301)
        n = 4500
        factors = rng.normal(0, 0.01, (n, 2))
        # Ground-truth DAILY intercept 0.04%, factor loadings 0.7 and -0.3.
        y = 0.0004 + factors @ np.array([0.7, -0.3]) + rng.normal(0, 0.002, n)
        output = hac_regression(y, factors, ("Mkt-RF", "SMB"))
        self.assertAlmostEqual(output["annualized_arithmetic_alpha"], 0.1008, delta=0.02)
        self.assertAlmostEqual(output["betas"]["Mkt-RF"], 0.7, delta=0.02)
        self.assertAlmostEqual(output["betas"]["SMB"], -0.3, delta=0.02)
        self.assertGreater(output["annualized_alpha_ci95_low"], 0.0)
        self.assertLess(output["p_two_sided_nominal"], 0.001)

    def test_noise_only_alpha_not_guaranteed_significant(self):
        rng = np.random.default_rng(101)
        n = 4500
        factors = rng.normal(0, 0.01, (n, 1))
        y = factors[:, 0] * 0.3 + rng.normal(0, 0.002, n)
        output = hac_regression(y, factors, ("Mkt-RF",))
        self.assertLess(output["annualized_alpha_ci95_low"], 0)
        self.assertGreater(output["annualized_alpha_ci95_high"], 0)
        self.assertGreater(output["p_two_sided_nominal"], 0.05)

    def test_rejects_missing_and_degenerate_regressors(self):
        y = np.linspace(-0.001, 0.001, 300)
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            hac_regression(y, np.ones((300, 1)), ("constant",))
        with self.assertRaisesRegex(ValueError, "finite"):
            hac_regression(y, np.full((300, 1), np.nan), ("broken",))


class AlignmentTests(unittest.TestCase):
    def test_dates_exactly_aligned_and_cash_subtracted(self):
        dates = business_days(date(2024, 1, 2), 160)
        factors = synthetic_factor_archives(dates)
        prices = 100.0 * np.exp(np.arange(160) * 0.001)
        net = np.full(159, 0.002)
        aligned = _aligned_arrays(dates, np.ones(159, dtype=bool), net, prices, factors)
        np.testing.assert_allclose(aligned["y"], 0.0019)
        np.testing.assert_allclose(aligned["spy"], np.expm1(0.001) - 0.0001)
        self.assertEqual(aligned["ff5"].shape, (159, 5))

    def test_missing_factor_date_fails_closed(self):
        dates = business_days(date(2024, 1, 2), 160)
        factors = synthetic_factor_archives(dates)
        del factors["mom"].dates_to_factors[dates[31]]
        with self.assertRaisesRegex(ValueError, "missing dates"):
            _aligned_arrays(
                dates, np.ones(159, dtype=bool), np.zeros(159),
                100 * np.ones(160), factors,
            )

    def test_disagreeing_risk_free_factors_fail_closed(self):
        dates = business_days(date(2024, 1, 2), 160)
        factors = synthetic_factor_archives(dates)
        factors["ff5"].dates_to_factors[dates[30]]["RF"] += 0.001
        with self.assertRaisesRegex(ValueError, "risk-free series disagree"):
            _aligned_arrays(
                dates, np.ones(159, dtype=bool), np.zeros(159),
                100 * np.ones(160), factors,
            )


class FrozenRuleTests(unittest.TestCase):
    def test_original_rule_is_used_and_holdout_marked_post_hoc(self):
        dates = business_days(date(2009, 1, 2), 4300)
        rng = np.random.default_rng(505)
        prices = 100.0 * np.r_[1.0, np.cumprod(1 + rng.normal(0.0003, 0.008, len(dates) - 1))]
        history = PriceHistory(
            dates=dates,
            adjusted_close=prices,
            annual_cash_yield_pct=np.full(len(dates), 3.0),
        )
        factors = synthetic_factor_archives(dates)
        output = run_factor_diagnostic(history, factors)
        self.assertEqual(output["frozen_strategy"], "volatility_target_20")
        self.assertIn("POST_HOC", output["status"])
        self.assertEqual(
            set(output["regressions"]["already_seen_final_2023_2026"]),
            {"SPY_market", "FF_market", "FF3", "FF5", "FF5_plus_momentum"},
        )
        for item in output["regressions"]["already_seen_final_2023_2026"].values():
            self.assertEqual(item["hac_lags"], 5)
            self.assertEqual(
                item["observations"],
                int(period_mask(dates, FINAL_FIRST, date(2100, 1, 1)).sum()),
            )
            self.assertGreaterEqual(item["p_bonferroni_across_five_models"], item["p_two_sided_nominal"])


if __name__ == "__main__":
    unittest.main()
