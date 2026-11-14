"""Read-only Alpaca SIP SPY daily-bar archival for the frozen paper study.

Only GETs https://data.alpaca.markets/v2/stocks/bars.  It never imports a
trading client, contacts a trading endpoint, or submits an order.  Dividends,
splits, and ^IRX cash yields require separately verified, explicit inputs:
Alpaca stock bars do not provide all fields in the frozen prospective ledger.

Run ``python collect_alpaca_spy.py --help`` for the private CSV workflow.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from math import isfinite
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import HTTPRedirectHandler, Request, build_opener

from prospective_paper import (
    NEW_YORK,
    START,
    WARMUP_COLUMNS,
    WARMUP_END,
    is_nyse_session,
    make_observation,
    market_close_utc,
    market_open_utc,
    next_nyse_session,
    read_warmup_csv,
)


DATA_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
DEFAULT_OUTPUT_DIR = Path("data/prospective/alpaca")
SUPPLEMENT_COLUMNS = (
    "Date", "DividendPerShare", "SplitRatio", "CashYieldPct",
    "CorporateActionSourceRef", "CashYieldSourceRef",
)
SOURCE_ID = "alpaca_v2_stocks_bars_SPY_1Day_sip_raw"
MIN_PUBLICATION_DELAY = timedelta(minutes=20)  # Basic SIP delay + buffer.


class _NoRedirect(HTTPRedirectHandler):
    """Never follow an HTTP redirect with the market-data API key headers."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def warmup_days() -> list[date]:
    """Exactly 21 sessions ending on the preregistered warmup endpoint."""
    days: list[date] = []
    day = WARMUP_END
    while len(days) < 21:
        if is_nyse_session(day):
            days.append(day)
        day -= timedelta(days=1)
    return list(reversed(days))


def validate_collection_time(kind: str, days: list[date], now: datetime) -> None:
    if now.tzinfo is None:
        raise ValueError("Local collection timestamp needs a timezone")
    now = now.astimezone(timezone.utc)
    if kind == "warmup":
        if days != warmup_days():
            raise ValueError("Warmup dates do not match frozen protocol")
        lower = market_close_utc(WARMUP_END) + MIN_PUBLICATION_DELAY
        upper = market_open_utc(START)
    elif kind == "daily":
        if len(days) != 1 or days[0] < START or not is_nyse_session(days[0]):
            raise ValueError("Expected one eligible prospective SPY session")
        lower = market_close_utc(days[0]) + MIN_PUBLICATION_DELAY
        upper = market_open_utc(next_nyse_session(days[0]))
    else:
        raise ValueError(f"Unsupported collection type: {kind}")
    if not lower <= now < upper:
        raise ValueError(
            f"Collection is permitted only after {lower.isoformat()} "
            f"and before {upper.isoformat()}, not at {now.isoformat()}"
        )


def alpaca_url(first: date, now: datetime) -> str:
    """Always request unadjusted consolidated SIP 1-day bars, never IEX."""
    # Basic market-data access excludes the most recent 15 minutes of SIP.
    # The request end must also stay behind that entitlement boundary.
    end = now.astimezone(timezone.utc) - timedelta(minutes=16)
    query = urlencode({
        "symbols": "SPY", "timeframe": "1Day", "start": first.isoformat(),
        "end": end.isoformat(), "adjustment": "raw", "feed": "sip",
        "sort": "asc", "limit": 1000,
    })
    return DATA_ENDPOINT + "?" + query


def fetch_alpaca_response(
    days: list[date], now: datetime, *, opener=None, credentials=None,
) -> tuple[bytes, str]:
    if credentials is None:
        credentials = (
            os.environ.get("ALPACA_API_KEY_ID", ""),
            os.environ.get("ALPACA_API_SECRET_KEY", ""),
        )
    key, secret = credentials
    if not key.strip() or not secret.strip():
        raise RuntimeError(
            "Alpaca data API credentials are missing. Set ALPACA_API_KEY_ID "
            "and ALPACA_API_SECRET_KEY locally; never commit or share them."
        )
    url = alpaca_url(days[0], now)
    req = Request(url, method="GET", headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
        "User-Agent": "quant-trading-validation-prospective-data/1.0",
    })
    if req.get_method() != "GET" or urlparse(url).hostname != "data.alpaca.markets":
        raise ValueError("Unexpected non-data API request")
    if opener is None:
        opener = build_opener(_NoRedirect()).open
    try:
        with opener(req, timeout=30) as response:
            raw = response.read(3_000_001)
    except HTTPError as exc:
        raise RuntimeError(
            f"Alpaca market-data HTTP {exc.code}; check key, SIP entitlement, "
            "and publication delay. No orders were attempted."
        ) from None
    except URLError as exc:
        raise RuntimeError("Alpaca market-data connection failed") from exc
    if not raw or len(raw) > 3_000_000:
        raise ValueError("Empty or unexpectedly large Alpaca daily-bar response")
    return raw, url


def parse_daily_closes(raw: bytes, expected_days: list[date]) -> dict[date, float]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Alpaca JSON daily-bar response") from exc
    if not isinstance(payload, dict) or payload.get("next_page_token"):
        raise ValueError("Incomplete/paginated Alpaca response; refusing partial quotes")
    bars = payload.get("bars")
    if not isinstance(bars, dict) or set(bars) != {"SPY"} or not isinstance(bars["SPY"], list):
        raise ValueError("Expected Alpaca stock bars only for SPY")
    closes: dict[date, float] = {}
    for bar in bars["SPY"]:
        if not isinstance(bar, dict) or not isinstance(bar.get("t"), str):
            raise ValueError("Alpaca bar missing timestamp")
        try:
            stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid Alpaca bar timestamp") from exc
        if stamp.tzinfo is None:
            raise ValueError("Alpaca bar timestamp is not timezone-aware")
        day = stamp.astimezone(NEW_YORK).date()
        raw_close = bar.get("c")
        if isinstance(raw_close, bool) or not isinstance(raw_close, (int, float)):
            raise ValueError("Alpaca close must be numeric, not a string or null")
        close = float(raw_close)
        if day in closes or day not in expected_days or not isfinite(close) or close <= 0:
            raise ValueError("Alpaca returned unexpected/duplicate day or invalid close")
        closes[day] = close
    if set(closes) != set(expected_days):
        missing = sorted(set(expected_days) - set(closes))
        raise ValueError(f"Missing SPY SIP raw daily close(s): {missing}")
    return closes


def _source_reference(value: str | None, name: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if len(result) < 8 or result.lower() in {
        "unknown", "unverified", "not verified", "none", "n/a", "todo",
        "placeholder", "example", "test", "synthetic",
    }:
        raise ValueError(f"{name} must cite a genuine checked source/date")
    return result


def read_supplements(path: Path, expected_days: list[date]) -> tuple[dict[date, dict], bytes]:
    """Never infer dividend=0, split=1, or cash yield from an absent source."""
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != list(SUPPLEMENT_COLUMNS):
        raise ValueError(f"Supplement CSV header must be exactly {SUPPLEMENT_COLUMNS}")
    results = {}
    for row in reader:
        if None in row:
            raise ValueError("Supplement has extra fields; check CSV quoting")
        try:
            day = date.fromisoformat(row["Date"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Supplement has missing/invalid Date") from exc
        if day in results or day not in expected_days:
            raise ValueError("Duplicate or out-of-range supplement date")
        for name in ("DividendPerShare", "SplitRatio", "CashYieldPct"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"{name} missing for {day}; do not guess zero or one")
        action_ref = _source_reference(row["CorporateActionSourceRef"], "CorporateActionSourceRef")
        yield_ref = _source_reference(row["CashYieldSourceRef"], "CashYieldSourceRef")
        results[day] = {
            "dividend": row["DividendPerShare"],
            "split": row["SplitRatio"],
            "yield": row["CashYieldPct"],
            "corporate_action_source": action_ref,
            "cash_yield_source": yield_ref,
        }
    if set(results) != set(expected_days):
        raise ValueError(f"Missing supplement dates: {sorted(set(expected_days) - set(results))}")
    return results, raw


def _csv_bytes(observations: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(WARMUP_COLUMNS)
    for obs in observations:
        writer.writerow((
            obs["date"], obs["close"], obs["dividend_per_share"],
            obs["split_ratio"], obs["cash_yield_pct"],
        ))
    return output.getvalue().encode("utf-8")


def paths_for(prefix: Path) -> dict[str, Path]:
    return {
        "csv": prefix.with_name(prefix.name + ".csv"),
        "raw": prefix.with_name(prefix.name + ".alpaca_raw.json"),
        "manifest": prefix.with_name(prefix.name + ".manifest.json"),
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())


def collect(
    kind: str, supplement_path: Path, output_dir: Path, *,
    day: date | None = None, now: datetime | None = None, opener=None, credentials=None,
) -> tuple[Path, dict]:
    test_clock = now is not None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Collection clock must have a timezone")
    now = now.astimezone(timezone.utc)
    days = warmup_days() if kind == "warmup" else [day]
    if None in days:
        raise ValueError("--date is required for daily collection")
    validate_collection_time(kind, days, now)
    identifier = f"warmup_through_{WARMUP_END}" if kind == "warmup" else f"session_{day}"
    prefix = output_dir / identifier
    files = paths_for(prefix)
    # Check for existing evidence before calling the provider at all.
    if any(path.exists() for path in files.values()):
        raise FileExistsError("An archive already exists; refusing to overwrite prospective evidence")
    supplements, supplement_bytes = read_supplements(supplement_path, days)
    raw, url = fetch_alpaca_response(days, now, opener=opener, credentials=credentials)
    # A network stall must not permit retroactive post-open 'prospective' data.
    received_at = now if test_clock else datetime.now(timezone.utc)
    validate_collection_time(kind, days, received_at)
    closes = parse_daily_closes(raw, days)
    observations = []
    for session in days:
        info = supplements[session]
        observations.append(make_observation(
            session, closes[session], info["dividend"],
            info["split"], info["yield"],
        ))
    csv_bytes = _csv_bytes(observations)
    manifest = {
        "kind": kind,
        "symbol": "SPY",
        "source_id": SOURCE_ID,
        "data_feed": "sip",
        "adjustment": "raw",
        "timeframe": "1Day",
        "retrieved_at_utc": received_at.isoformat(),
        "query_url_without_credentials": url,
        "sessions": [d.isoformat() for d in days],
        "raw_alpaca_response_sha256": hashlib.sha256(raw).hexdigest(),
        "supplements_sha256": hashlib.sha256(supplement_bytes).hexdigest(),
        "output_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "source_references": [
            {"date": d.isoformat(),
             "corporate_actions": supplements[d]["corporate_action_source"],
             "cash_yield": supplements[d]["cash_yield_source"]}
            for d in days
        ],
        "limitations": [
            "SIP raw daily bars are vendor aggregates, not independently audited official auction closes.",
            "Dividends, splits, and ^IRX yields require human-verified supplemental sources.",
            "A source reference records an operator attestation, not independent API verification.",
            "Data may later be revised by the vendor; archive is an as-retrieved snapshot.",
            "Never sends paper or live orders; never auto-commits to the prospective ledger.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write the raw response first; preserve partial evidence if interrupted.
    _exclusive_write(files["raw"], raw)
    _exclusive_write(files["csv"], csv_bytes)
    _exclusive_write(files["manifest"], (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8"))
    verify_archive(prefix)
    return prefix, manifest


def verify_archive(prefix: Path) -> dict:
    """Check archived bytes, frozen schema, and raw/CSV price equivalence."""
    files = paths_for(prefix)
    manifest = json.loads(files["manifest"].read_bytes())
    raw = files["raw"].read_bytes()
    csv_bytes = files["csv"].read_bytes()
    if (manifest.get("source_id") != SOURCE_ID or
            manifest.get("data_feed") != "sip" or manifest.get("adjustment") != "raw"):
        raise ValueError("Archive is not for SPY 1Day unadjusted SIP bars")
    query = urlparse(manifest["query_url_without_credentials"])
    params = parse_qs(query.query)
    if (query.scheme, query.netloc, query.path) != (
        "https", "data.alpaca.markets", "/v2/stocks/bars"
    ) or any(params.get(k) != [v] for k, v in (
        ("symbols", "SPY"), ("timeframe", "1Day"),
        ("feed", "sip"), ("adjustment", "raw"),
    )):
        raise ValueError("Archived query does not match the approved read-only endpoint")
    if hashlib.sha256(raw).hexdigest() != manifest["raw_alpaca_response_sha256"]:
        raise ValueError("Archived raw Alpaca response checksum mismatch")
    if hashlib.sha256(csv_bytes).hexdigest() != manifest["output_csv_sha256"]:
        raise ValueError("Archived CSV checksum mismatch")
    days = [date.fromisoformat(x) for x in manifest["sessions"]]
    if manifest["kind"] == "warmup":
        if days != warmup_days():
            raise ValueError("Archive warmup days changed")
        rows = read_warmup_csv(files["csv"])
    elif manifest["kind"] == "daily":
        if len(days) != 1 or days[0] < START or not is_nyse_session(days[0]):
            raise ValueError("Archive daily session invalid")
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
        if reader.fieldnames != list(WARMUP_COLUMNS):
            raise ValueError("Daily CSV header changed")
        content = list(reader)
        if len(content) != 1:
            raise ValueError("Daily CSV must have exactly one row")
        row = content[0]
        rows = [make_observation(
            date.fromisoformat(row["Date"]), row["Close"],
            row["DividendPerShare"], row["SplitRatio"], row["CashYieldPct"],
        )]
    else:
        raise ValueError("Unrecognized archive kind")
    closes = parse_daily_closes(raw, days)
    if len(rows) != len(days) or any(
        row["date"] != day.isoformat() or row["close"] != closes[day]
        for row, day in zip(rows, days)
    ):
        raise ValueError("CSV prices differ from originally archived Alpaca bars")
    if len(manifest["source_references"]) != len(days):
        raise ValueError("Source references are missing")
    for item, day in zip(manifest["source_references"], days):
        if item["date"] != day.isoformat():
            raise ValueError("Source reference date mismatch")
        _source_reference(item["corporate_actions"], "corporate_actions")
        _source_reference(item["cash_yield"], "cash_yield")
    return manifest


def write_supplement_template(kind: str, output: Path, *, day: date | None = None) -> None:
    days = warmup_days() if kind == "warmup" else [day]
    if day is None and kind == "daily":
        raise ValueError("--date is required for a daily template")
    if kind not in ("warmup", "daily"):
        raise ValueError("Unsupported supplement template mode")
    if kind == "daily" and (day < START or not is_nyse_session(day)):
        raise ValueError("Daily date is outside prospective NYSE sessions")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = io.StringIO(newline="")
    writer = csv.writer(content, lineterminator="\n")
    writer.writerow(SUPPLEMENT_COLUMNS)
    for session in days:
        writer.writerow([session.isoformat(), "", "", "", "", ""])
    _exclusive_write(output, content.getvalue().encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Generate unfilled, private supplemental CSV")
    template.add_argument("--kind", choices=("warmup", "daily"), required=True)
    template.add_argument("--date", type=date.fromisoformat)
    template.add_argument("--output", type=Path, required=True)
    for command in ("warmup", "daily"):
        p = sub.add_parser(command, help=f"Read-only collect verified {command} data")
        p.add_argument("--supplements", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if command == "daily":
            p.add_argument("--date", type=date.fromisoformat, required=True)
    verify = sub.add_parser("verify", help="Revalidate a previously archived source bundle")
    verify.add_argument("--bundle", type=Path, required=True,
                        help="Archive path prefix without .csv or .manifest.json")
    args = parser.parse_args()
    if args.command == "template":
        write_supplement_template(args.kind, args.output, day=args.date)
        print("Created EMPTY supplement template:", args.output)
        print("Fill only verified numbers and document both source references; no automatic defaults.")
        return
    if args.command == "verify":
        manifest = verify_archive(args.bundle)
        print(json.dumps({"status": "ARCHIVE_VALID", "source_id": manifest["source_id"],
                          "sessions": manifest["sessions"],
                          "csv_sha256": manifest["output_csv_sha256"]}, indent=2))
        return
    prefix, manifest = collect(
        args.command, args.supplements, args.output_dir,
        day=getattr(args, "date", None),
    )
    print(json.dumps({
        "status": "READ_ONLY_SNAPSHOT_ARCHIVED_NOT_ENTERED_IN_PAPER_LEDGER",
        "csv": str(paths_for(prefix)["csv"]),
        "manifest": str(paths_for(prefix)["manifest"]),
        "raw_snapshot": str(paths_for(prefix)["raw"]),
        "source_id": manifest["source_id"],
        "csv_sha256": manifest["output_csv_sha256"],
        "sessions": manifest["sessions"],
        "real_orders": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
