"""All Alpaca API requests are mocked: no key, network, or order is used."""

import csv
from datetime import date, datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

from collect_alpaca_spy import (
    SOURCE_ID,
    SUPPLEMENT_COLUMNS,
    _NoRedirect,
    alpaca_url,
    collect,
    fetch_alpaca_response,
    parse_daily_closes,
    paths_for,
    read_supplements,
    validate_collection_time,
    verify_archive,
    warmup_days,
    write_supplement_template,
)
from prospective_paper import NEW_YORK, START, WARMUP_END, is_nyse_session, market_close_utc


UTC = timezone.utc


def mock_response(days, *, omit=None, duplicate=False, bad_close=False, paginated=False):
    bars = []
    for i, day in enumerate(days):
        if day == omit:
            continue
        stamp = datetime.combine(day, datetime.min.time(), NEW_YORK).astimezone(UTC)
        bars.append({"t": stamp.isoformat(), "c": 100.0 + i, "o": 100.0, "v": 10000})
    if duplicate:
        bars.append(dict(bars[-1]))
    if bad_close:
        bars[-1]["c"] = "not-a-number"
    return json.dumps({
        "bars": {"SPY": bars},
        "next_page_token": "unfinished" if paginated else None,
    }).encode()


def write_test_supplements(path, days, *, omit_yield=False, blank_action_ref=False):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(SUPPLEMENT_COLUMNS)
        for d in days:
            writer.writerow([
                d.isoformat(), "0", "1", "" if omit_yield else "3.6",
                "" if blank_action_ref else f"TEST_CORPORATE_ACTIONS_{d}",
                f"TEST_IRX_YIELD_{d}",
            ])


class MockOpener:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, req, timeout):
        self.calls.append((req, timeout))
        return io.BytesIO(self.response)


class AlpacaReadOnlyCollectionTests(unittest.TestCase):
    def test_warmup_calendar_exactly_21_nyse_sessions(self):
        days = warmup_days()
        self.assertEqual(len(days), 21)
        self.assertEqual(days[-1], WARMUP_END)
        self.assertEqual(len(set(days)), 21)
        self.assertTrue(all(is_nyse_session(x) for x in days))

    def test_missing_or_future_market_times_fail_closed(self):
        self.assertRaises(ValueError, validate_collection_time,
                          "warmup", warmup_days(), datetime(2026, 11, 13, 21, 19, tzinfo=UTC))
        validate_collection_time(
            "warmup", warmup_days(), datetime(2026, 11, 13, 21, 20, tzinfo=UTC))
        self.assertRaises(ValueError, validate_collection_time,
                          "warmup", warmup_days(), datetime(2026, 11, 16, 14, 30, tzinfo=UTC))
        self.assertRaises(ValueError, validate_collection_time,
                          "daily", [START], datetime(2026, 11, 16, 21, 0, tzinfo=UTC))
        validate_collection_time(
            "daily", [START], datetime(2026, 11, 16, 21, 20, tzinfo=UTC))
        self.assertRaises(ValueError, validate_collection_time,
                          "daily", [date(2026, 11, 15)], datetime(2026, 11, 16, 21, 20, tzinfo=UTC))

    def test_template_is_empty_and_private_never_invents_auxiliary_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "supplement.csv"
            write_supplement_template("warmup", path)
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 21)
            self.assertEqual(rows[0]["CashYieldPct"], "")
            self.assertEqual(rows[-1]["DividendPerShare"], "")
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_supplement_template("warmup", path)
            with self.assertRaises(ValueError):
                read_supplements(path, warmup_days())

    def test_only_read_only_data_endpoint_raw_sip_and_secret_headers(self):
        days = [START]
        now = datetime(2026, 11, 16, 21, 30, tzinfo=UTC)
        opener = MockOpener(mock_response(days))
        raw, url = fetch_alpaca_response(
            days, now, opener=opener, credentials=("PAPER_KEY", "PAPER_SECRET")
        )
        self.assertEqual(parse_daily_closes(raw, days)[START], 100.0)
        req, timeout = opener.calls[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(urlparse(url).hostname, "data.alpaca.markets")
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["adjustment"], ["raw"])
        self.assertEqual(params["feed"], ["sip"])
        self.assertEqual(params["timeframe"], ["1Day"])
        self.assertEqual(params["symbols"], ["SPY"])
        self.assertNotIn("PAPER_SECRET", url)
        self.assertNotIn("PAPER_KEY", url)
        self.assertEqual(req.get_header("Apca-api-key-id"), "PAPER_KEY")
        self.assertEqual(req.get_header("Apca-api-secret-key"), "PAPER_SECRET")
        self.assertEqual(timeout, 30)

    def test_missing_credentials_fails_without_http(self):
        opener = MockOpener(mock_response([START]))
        with self.assertRaisesRegex(RuntimeError, "credentials are missing"):
            fetch_alpaca_response([START], datetime(2026, 11, 16, 21, 30, tzinfo=UTC),
                                  opener=opener, credentials=("", ""))
        self.assertFalse(opener.calls)

    def test_redirect_cannot_forward_api_key_to_another_host(self):
        req = Request("https://data.alpaca.markets/v2/stocks/bars", headers={
            "APCA-API-KEY-ID": "SECRET_KEY_ID",
        })
        with self.assertRaisesRegex(HTTPError, "HTTP Error 302"):
            _NoRedirect().redirect_request(
                req, None, 302, "Redirect", {}, "https://invalid.example/steal"
            )

    def test_fails_on_missing_duplicate_invalid_or_paginated_prices(self):
        dates = warmup_days()
        for content in (
            mock_response(dates, omit=dates[-2]),
            mock_response(dates, duplicate=True),
            mock_response(dates, bad_close=True),
            mock_response(dates, paginated=True),
            json.dumps({"bars": {"SPY": [], "QQQ": []}}).encode(),
        ):
            with self.subTest(raw=content[-40:]):
                with self.assertRaises(ValueError):
                    parse_daily_closes(content, dates)

    def test_all_supplement_values_and_sources_are_mandatory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "aux.csv"
            for kwarg in (dict(omit_yield=True), dict(blank_action_ref=True)):
                write_test_supplements(path, [START], **kwarg)
                with self.assertRaises(ValueError):
                    read_supplements(path, [START])
            write_test_supplements(path, [START])
            items, _ = read_supplements(path, [START])
            self.assertEqual(items[START]["split"], "1")

    def test_end_to_end_warmup_archives_exact_source_and_validates(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            days = warmup_days()
            supplements = base / "aux.csv"
            write_test_supplements(supplements, days)
            opener = MockOpener(mock_response(days))
            now = datetime(2026, 11, 13, 22, 0, tzinfo=UTC)
            prefix, manifest = collect(
                "warmup", supplements, base / "out", now=now,
                opener=opener, credentials=("KEY", "SECRET"))
            self.assertEqual(manifest["source_id"], SOURCE_ID)
            self.assertEqual(manifest["sessions"], [str(x) for x in days])
            files = paths_for(prefix)
            self.assertEqual(files["raw"].read_bytes(), mock_response(days))
            self.assertEqual(os.stat(files["manifest"]).st_mode & 0o777, 0o600)
            self.assertEqual(verify_archive(prefix), manifest)
            before = len(opener.calls)
            with self.assertRaises(FileExistsError):
                collect("warmup", supplements, base / "out", now=now,
                        opener=opener, credentials=("KEY", "SECRET"))
            self.assertEqual(len(opener.calls), before)

    def test_daily_cannot_be_backfilled_or_entered_into_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            supplements = base / "daily.csv"
            write_test_supplements(supplements, [START])
            opener = MockOpener(mock_response([START]))
            now = datetime(2026, 11, 16, 21, 30, tzinfo=UTC)
            prefix, manifest = collect("daily", supplements, base / "archive",
                                       day=START, now=now, opener=opener,
                                       credentials=("KEY", "SECRET"))
            self.assertEqual(verify_archive(prefix), manifest)
            self.assertEqual(len(paths_for(prefix)["csv"].read_text().splitlines()), 2)
            self.assertFalse((base / "archive" / "spy_paper_ledger.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "only after"):
                collect("daily", supplements, base / "other", day=START,
                        now=now + timedelta(days=1), opener=opener,
                        credentials=("KEY", "SECRET"))

    def test_tampered_source_or_csv_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            supplements = base / "aux.csv"
            write_test_supplements(supplements, [START])
            prefix, _ = collect(
                "daily", supplements, base / "out", day=START,
                now=datetime(2026, 11, 16, 21, 20, tzinfo=UTC),
                opener=MockOpener(mock_response([START])),
                credentials=("KEY", "SECRET"),
            )
            csv_file = paths_for(prefix)["csv"]
            csv_file.write_bytes(csv_file.read_bytes().replace(b"100.0", b"110.0"))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_archive(prefix)


if __name__ == "__main__":
    unittest.main()
