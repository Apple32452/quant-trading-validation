"""Synthetic-only tests: no real prospective observations or orders."""

import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from factor_alpha import FactorArchive
from prospective_paper import (
    BOOTSTRAP_CONFIDENCE,
    REQUIRED_REALIZED_SESSIONS,
    START,
    WARMUP_END,
    _seal_record,
    append_close,
    final_report,
    initialize,
    is_nyse_session,
    locked_returns,
    make_observation,
    market_close_utc,
    next_nyse_session,
    public_status,
    read_journal,
    read_warmup_csv,
    signal_at_last_close,
    verify_records,
)


UTC = timezone.utc


def warmup_file(path: Path):
    days = []
    current = date(2026, 10, 15)
    while current <= WARMUP_END:
        if is_nyse_session(current):
            days.append(current)
        current += timedelta(days=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("Date", "Close", "DividendPerShare", "SplitRatio", "CashYieldPct"))
        for index, day in enumerate(days):
            # Variable returns, zero dividends/splits, fully labeled mock data.
            writer.writerow([day.isoformat(), 100.0 + index * 0.3 + np.sin(index), 0, 1, 3.5])
    return days


def mock_now(day: date) -> datetime:
    return market_close_utc(day) + timedelta(minutes=15)


class ProspectiveClockAndStateTests(unittest.TestCase):
    def test_calendar_holidays_and_early_closes(self):
        self.assertEqual(next_nyse_session(date(2026, 11, 25)), date(2026, 11, 27))
        self.assertEqual(market_close_utc(date(2026, 11, 27)).hour, 18)  # 13 ET
        self.assertEqual(next_nyse_session(date(2026, 12, 24)), date(2026, 12, 28))
        self.assertEqual(START, date(2026, 11, 16))
        self.assertEqual(REQUIRED_REALIZED_SESSIONS, 252)

    def test_warmup_and_both_time_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            warmup = Path(folder) / "warmup.csv"
            journal = Path(folder) / "ledger.jsonl"
            warmup_file(warmup)
            rows = read_warmup_csv(warmup)
            self.assertGreaterEqual(len(rows), 21)
            with self.assertRaisesRegex(ValueError, "only after 11/13"):
                initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 13, 20, 0, tzinfo=UTC))
            status = initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 14, 16, 0, tzinfo=UTC))
            self.assertEqual(status["completed_return_sessions"], 0)
            self.assertTrue(status["interim_performance_hidden"])
            self.assertNotIn("net_cagr", status)
            self.assertNotIn("sharpe", status)
            self.assertEqual(len(read_journal(journal)), 1)
            with self.assertRaises(FileExistsError):
                initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 14, 16, 0, tzinfo=UTC))

    def test_append_order_and_single_lagged_return_with_entry_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            warmup = Path(folder) / "warmup.csv"
            journal = Path(folder) / "ledger.jsonl"
            warmup_file(warmup)
            rows = read_warmup_csv(warmup)
            initial_signal = signal_at_last_close(rows)
            initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 14, 16, tzinfo=UTC))
            monday = make_observation(START, 125, 0, 1, 3.6)
            late = datetime(2026, 11, 17, 16, tzinfo=UTC)
            with self.assertRaisesRegex(ValueError, "Late, early"):
                append_close(journal, monday, mock_now(START), "MOCK", now=late)
            first = append_close(journal, monday, mock_now(START), "MOCK", now=mock_now(START))
            self.assertEqual(first["completed_return_sessions"], 0)
            self.assertTrue(first["interim_performance_hidden"])
            with self.assertRaisesRegex(ValueError, "immediately following"):
                append_close(journal, make_observation(date(2026, 11, 18), 127, 0, 1, 3.6),
                             mock_now(date(2026, 11, 18)), "MOCK", now=mock_now(date(2026, 11, 18)))
            day2 = date(2026, 11, 17)
            second = append_close(journal, make_observation(day2, 126, 0.3, 1, 3.7),
                                  mock_now(day2), "MOCK", now=mock_now(day2))
            self.assertEqual(second["completed_return_sessions"], 1)
            self.assertTrue(second["interim_performance_hidden"])
            events = read_journal(journal)
            returns = locked_returns(events)
            self.assertEqual(len(returns["strategy"]), 1)
            self.assertAlmostEqual(returns["weights"][0], initial_signal)
            asset = (126 + 0.3) / 125 - 1.0
            expected = initial_signal * asset + (1.0 - initial_signal) * (3.6 / 100 / 252) - 0.0005 * initial_signal
            self.assertAlmostEqual(returns["strategy"][0], expected)
            self.assertAlmostEqual(returns["benchmark"][0], asset - 0.0005)
            with self.assertRaisesRegex(ValueError, "before 252"):
                final_report(events, None, None)

    def test_append_only_hash_chain_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            warmup = Path(folder) / "warmup.csv"
            journal = Path(folder) / "ledger.jsonl"
            warmup_file(warmup)
            initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 14, 16, tzinfo=UTC))
            append_close(journal, make_observation(START, 120, 0, 1, 3.5),
                         mock_now(START), "MOCK", now=mock_now(START))
            original = journal.read_bytes()
            self.assertEqual(len(verify_records(original)), 2)
            # Corruption of a prior close, even if valid JSON, breaks chain.
            journal.write_bytes(original.replace(b'"close": 120.0', b'"close": 121.0'))
            with self.assertRaisesRegex(ValueError, "hash chain broken"):
                read_journal(journal)

    def test_full_length_synthetic_finalizer_only_after_252(self):
        with tempfile.TemporaryDirectory() as folder:
            warmup = Path(folder) / "warmup.csv"
            journal = Path(folder) / "ledger.jsonl"
            warmup_file(warmup)
            initialize(warmup, journal, "SYNTHETIC", now=datetime(2026, 11, 14, 16, tzinfo=UTC))
            header = read_journal(journal)[0]
            events = [header]
            day = START
            rng = np.random.default_rng(17)
            close = 110.0
            for _ in range(253):  # One entry close, then 252 realized intervals.
                close *= 1 + float(rng.normal(0.0002, 0.008))
                obs = make_observation(day, close, 0.0, 1.0, 3.7)
                now = mock_now(day)
                events.append(_seal_record({
                    "kind": "close", "observation": obs,
                    "source_observed_at_utc": now.isoformat(),
                    "recorded_at_utc": now.isoformat(), "source_ref": "MOCK",
                }, events[-1]["record_sha256"]))
                day = next_nyse_session(day)
            self.assertEqual(public_status(events)["completed_return_sessions"], 252)
            dates = [date.fromisoformat(x["observation"]["date"]) for x in events[2:]]
            rng = np.random.default_rng(88)
            ff5, mom = {}, {}
            for d in dates:
                vals = rng.normal(0, 0.01, 6)
                ff5[d] = {"Mkt-RF": float(vals[0]), "SMB": float(vals[1]),
                          "HML": float(vals[2]), "RMW": float(vals[3]),
                          "CMA": float(vals[4]), "RF": 0.0001}
                mom[d] = {"Mom": float(vals[5])}
            source = lambda name, data: FactorArchive(name, data, "SYNTHETIC", dates[0], dates[-1])
            report = final_report(events, source("ff5", ff5), source("mom", mom))
            self.assertEqual(report["sessions"], 252)
            self.assertEqual(report["ff5_momentum_alpha"]["hac_lags"], 20)
            self.assertEqual(report["net_minus_spy_active_mean"]["confidence"], BOOTSTRAP_CONFIDENCE)
            self.assertEqual(report["strategy_net"]["days"], 252)
            self.assertEqual(report["spy_buy_hold_net"]["days"], 252)
            self.assertEqual(report["status"], "PROSPECTIVE_PAPER_SIMULATION_COMPLETED_NO_LIVE_TRADES")


if __name__ == "__main__":
    unittest.main()
