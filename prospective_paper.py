"""Append-only, forward-only SPY PAPER simulation: NEVER sends live orders.

The rule is the previously frozen volatility_target_20. The prospective
experiment cannot start until a verified warm-up through 2026-11-13 exists.
Daily records must be appended before the following NYSE session opens.
No PnL, Sharpe, or alpha is revealed until 252 subsequent return sessions.
See PROSPECTIVE_PAPER_RUNBOOK.md for input/interpretation limitations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time, timedelta, timezone
import fcntl
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np

from cross_asset_replication import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_SEED,
    REGRESSION_FACTORS,
    REGRESSION_HAC_LAGS,
    paired_circular_block_ci97_5,
)
from factor_alpha import hac_regression, load_french_archive
from market_backtest import (
    CANDIDATES,
    REFERENCE_COST_BPS,
    TRADING_DAYS,
    period_metrics,
    portfolio_returns,
    target_weights,
)


NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
START = date(2026, 11, 16)
WARMUP_END = date(2026, 11, 13)
REQUIRED_REALIZED_SESSIONS = 252
PREVIOUS_FROZEN_COMMIT = "cc3919d28362dc5026942d5ce1e259436f2cad6a"
NYSE_CALENDAR_REFERENCE = "https://www.nyse.com/trade/hours-calendars"
FROZEN_RULE = "volatility_target_20"
JOURNAL_VERSION = 1
DEFAULT_JOURNAL = Path("data/prospective/spy_paper_ledger.jsonl")

# Official NYSE 2026/2027 published calendar. Unexpected emergency closures
# require a documented protocol amendment; never silently invent sessions.
NYSE_CLOSED = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
})
NYSE_EARLY_CLOSE = frozenset({
    date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26),
})
WARMUP_COLUMNS = ("Date", "Close", "DividendPerShare", "SplitRatio", "CashYieldPct")


def frozen_rule():
    rule = next((x for x in CANDIDATES if x.name == FROZEN_RULE), None)
    if rule is None or (rule.kind, rule.lookback, rule.annual_target_volatility) != (
        "volatility", 20, 0.10
    ) or REFERENCE_COST_BPS != 5.0:
        raise ValueError("Frozen strategy or transaction cost has changed")
    return rule


def is_nyse_session(day: date) -> bool:
    return day.year in (2026, 2027) and day.weekday() < 5 and day not in NYSE_CLOSED


def next_nyse_session(day: date) -> date:
    candidate = day + timedelta(days=1)
    for _ in range(15):
        if is_nyse_session(candidate):
            return candidate
        candidate += timedelta(days=1)
    raise ValueError("Session calendar outside frozen 2026-2027 scope")


def market_close_utc(day: date) -> datetime:
    if not is_nyse_session(day):
        raise ValueError(f"No NYSE core trading session on {day}")
    hour = 13 if day in NYSE_EARLY_CLOSE else 16
    return datetime.combine(day, time(hour), NEW_YORK).astimezone(UTC)


def market_open_utc(day: date) -> datetime:
    if not is_nyse_session(day):
        raise ValueError(f"No NYSE core trading session on {day}")
    return datetime.combine(day, time(9, 30), NEW_YORK).astimezone(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamps must contain a timezone offset")
    return value.astimezone(UTC)


def _value(raw: object, *, field: str) -> float:
    number = float(raw)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def make_observation(
    day: date, close: float, dividend: float, split_ratio: float,
    cash_yield_pct: float,
) -> dict:
    if not is_nyse_session(day):
        raise ValueError(f"Not a registered NYSE session: {day}")
    close = _value(close, field="Close")
    dividend = _value(dividend, field="DividendPerShare")
    split_ratio = _value(split_ratio, field="SplitRatio")
    cash_yield_pct = _value(cash_yield_pct, field="CashYieldPct")
    if close <= 0 or dividend < 0 or split_ratio <= 0 or not -20 < cash_yield_pct < 100:
        raise ValueError("Invalid close, dividend, split ratio, or cash yield")
    return {
        "date": day.isoformat(), "close": close,
        "dividend_per_share": dividend, "split_ratio": split_ratio,
        "cash_yield_pct": cash_yield_pct,
    }


def read_warmup_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(WARMUP_COLUMNS):
            raise ValueError(f"Warmup CSV columns must be exactly {WARMUP_COLUMNS}")
        rows = [make_observation(
            date.fromisoformat(row["Date"]), row["Close"],
            row["DividendPerShare"], row["SplitRatio"], row["CashYieldPct"]
        ) for row in reader]
    validate_warmup_rows(rows)
    return rows


def validate_warmup_rows(rows: list[dict]) -> None:
    if len(rows) < 21 or date.fromisoformat(rows[-1]["date"]) != WARMUP_END:
        raise ValueError("Need >=21 consecutive warmup closes ending 2026-11-13")
    for row in rows:
        normalized = make_observation(
            date.fromisoformat(row["date"]), row["close"],
            row["dividend_per_share"], row["split_ratio"], row["cash_yield_pct"]
        )
        if normalized != row:
            raise ValueError("Warmup records have unexpected extra fields")
    for prior, current in zip(rows, rows[1:]):
        if date.fromisoformat(current["date"]) != next_nyse_session(date.fromisoformat(prior["date"])):
            raise ValueError("Warmup has missing, duplicated, or out-of-order NYSE sessions")
    total_return_index(rows)  # Preflight all price/dividend/split transitions.


def total_return_index(observations: list[dict]) -> np.ndarray:
    """Online cash-dividend-reinvested index from unadjusted point-in-time closes.

    SplitRatio is NEW shares per OLD share, and DividendPerShare is the
    ex-date cash entitlement for one post-split share. A zero dividend and
    unit split are explicit, never silently filled.
    """
    index = np.ones(len(observations), dtype=float) * 100.0
    for i in range(1, len(observations)):
        previous, current = observations[i - 1], observations[i]
        growth = (current["split_ratio"] *
                  (current["close"] + current["dividend_per_share"]) /
                  previous["close"])
        if not 0 < growth < 100 or not isfinite(growth):
            raise ValueError("Invalid price, dividend, or split transition")
        index[i] = index[i - 1] * growth
    if not np.all(np.isfinite(index)) or np.any(index <= 0):
        raise ValueError("Non-finite total return index")
    return index


def signal_at_last_close(observations: list[dict]) -> float:
    if len(observations) < 21:
        raise ValueError("Need 20 fully observed trailing daily returns")
    index = total_return_index(observations[-21:])
    recent = index[1:] / index[:-1] - 1.0
    vol = float(np.std(recent, ddof=1)) * np.sqrt(TRADING_DAYS)
    return float(min(1.0, 0.10 / max(vol, 1e-12)))


def _sha_record(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _seal_record(record: dict, previous: str) -> dict:
    body = {**record, "previous_sha256": previous}
    return {**body, "record_sha256": _sha_record(body)}


def verify_records(raw: bytes) -> list[dict]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Missing or incomplete append-only journal")
    rows = []
    previous = "0" * 64
    for line in raw.splitlines():
        event = json.loads(line)
        digest = event.get("record_sha256")
        body = {key: value for key, value in event.items() if key != "record_sha256"}
        if event.get("previous_sha256") != previous or digest != _sha_record(body):
            raise ValueError("Journal hash chain broken: possible modification")
        if not rows and event.get("kind") != "initialize":
            raise ValueError("Missing initialization event")
        if rows and event.get("kind") != "close":
            raise ValueError("Expected an appended close observation")
        rows.append(event)
        previous = digest
    return rows


def initialize(
    warmup_csv: Path, journal: Path, source_id: str, *, now: datetime | None = None
) -> dict:
    """One-time, strictly pre-entry initialization. No automatic orders."""
    now = _utc(now or datetime.now(UTC))
    if now < market_close_utc(WARMUP_END) or now >= market_open_utc(START):
        raise ValueError("Initialize only after 11/13 close and before 11/16 open")
    if not source_id.strip():
        raise ValueError("Document the price source ID")
    rule = frozen_rule()
    warmup = read_warmup_csv(warmup_csv)
    payload = {
        "kind": "initialize", "version": JOURNAL_VERSION,
        "created_at_utc": now.isoformat(),
        "start_date": START.isoformat(), "warmup_end_date": WARMUP_END.isoformat(),
        "source_id": source_id.strip(),
        "warmup_file_sha256": hashlib.sha256(warmup_csv.read_bytes()).hexdigest(),
        "warmup_observations": warmup,
        "frozen_commit": PREVIOUS_FROZEN_COMMIT,
        "frozen_rule": rule.name,
        "next_close_execution_weight": signal_at_last_close(warmup),
        "zero_real_orders": True,
        "method": "no model changes, next-close execution, 5-bp one-way cost",
    }
    sealed = _seal_record(payload, "0" * 64)
    journal.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write((json.dumps(sealed, sort_keys=True, allow_nan=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        # Preserve even a partial journal for forensic inspection; never
        # silently delete or overwrite the only provenance record.
        raise
    return public_status([sealed])


def _observations(events: list[dict]) -> list[dict]:
    return list(events[0]["warmup_observations"]) + [
        event["observation"] for event in events[1:]
    ]


def public_status(events: list[dict]) -> dict:
    warmup = events[0]["warmup_observations"]
    closes = len(events) - 1
    count = max(0, closes - 1)
    if count > REQUIRED_REALIZED_SESSIONS:
        raise ValueError("Experiment extended past its frozen endpoint")
    observations = _observations(events)
    next_execution = signal_at_last_close(observations)
    return {
        "status": "COMPLETE_AWAIT_FINAL_REPORT" if count == REQUIRED_REALIZED_SESSIONS
                  else "FORWARD_PAPER_RECORDING_NOT_A_PERFORMANCE_CLAIM",
        "experiment": "frozen SPY volatility_target_20 vs SPY buy-and-hold",
        "last_committed_close": observations[-1]["date"],
        "completed_return_sessions": count,
        "required_return_sessions": REQUIRED_REALIZED_SESSIONS,
        "next_executable_signal_after_one_more_close": next_execution,
        "journal_head_sha256": events[-1]["record_sha256"],
        "real_orders_placed": 0,
        "interim_performance_hidden": count < REQUIRED_REALIZED_SESSIONS,
    }


def read_journal(journal: Path) -> list[dict]:
    events = verify_records(journal.read_bytes())
    header = events[0]
    if (header["version"], header["start_date"], header["warmup_end_date"],
        header["frozen_rule"], header["frozen_commit"]) != (
        JOURNAL_VERSION, START.isoformat(), WARMUP_END.isoformat(),
        FROZEN_RULE, PREVIOUS_FROZEN_COMMIT
    ):
        raise ValueError("Journal header no longer matches frozen specification")
    validate_warmup_rows(header["warmup_observations"])
    created = _utc(datetime.fromisoformat(header["created_at_utc"]))
    if not market_close_utc(WARMUP_END) <= created < market_open_utc(START):
        raise ValueError("Journal was not initialized within the pre-entry window")
    if abs(header["next_close_execution_weight"] - signal_at_last_close(header["warmup_observations"])) > 1e-12:
        raise ValueError("Frozen initialization signal differs from warmup data")
    if not header["source_id"].strip() or header["zero_real_orders"] is not True:
        raise ValueError("Missing price source or incorrect paper-only marker")
    prior = WARMUP_END
    for event in events[1:]:
        obs = event["observation"]
        current = date.fromisoformat(obs["date"])
        if current != next_nyse_session(prior):
            raise ValueError("Journal skips, reorders, or duplicates an NYSE session")
        make_observation(
            current, obs["close"], obs["dividend_per_share"],
            obs["split_ratio"], obs["cash_yield_pct"],
        )
        seen = _utc(datetime.fromisoformat(event["source_observed_at_utc"]))
        recorded = _utc(datetime.fromisoformat(event["recorded_at_utc"]))
        if not market_close_utc(current) <= seen <= recorded < market_open_utc(next_nyse_session(current)):
            raise ValueError("Close recorded outside the prospective arrival window")
        if not event["source_ref"].strip():
            raise ValueError("Missing observation provenance")
        prior = current
    total_return_index(_observations(events))
    public_status(events)
    return events


def append_close(
    journal: Path, observation: dict, source_observed_at: datetime,
    source_ref: str, *, now: datetime | None = None,
) -> dict:
    """Append precisely one new observed close while holding the file lock."""
    now = _utc(now or datetime.now(UTC))
    source_observed_at = _utc(source_observed_at)
    if not source_ref.strip():
        raise ValueError("A source retrieval or archival reference is required")
    day = date.fromisoformat(observation["date"])
    accepted = make_observation(
        day, observation["close"], observation["dividend_per_share"],
        observation["split_ratio"], observation["cash_yield_pct"],
    )
    if not market_close_utc(day) <= source_observed_at <= now < market_open_utc(next_nyse_session(day)):
        raise ValueError("Late, early, or future close: cannot claim prospective record")
    with journal.open("r+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        events = verify_records(f.read())
        # Complete verification before writing anything.
        if events != read_journal(journal):
            raise ValueError("Journal changed during the locked update")
        state = public_status(events)
        if state["completed_return_sessions"] >= REQUIRED_REALIZED_SESSIONS:
            raise ValueError("Frozen 252-return experiment already complete")
        previous = date.fromisoformat(state["last_committed_close"])
        if day != next_nyse_session(previous):
            raise ValueError("New close must be the immediately following NYSE session")
        event = _seal_record({
            "kind": "close", "observation": accepted,
            "source_observed_at_utc": source_observed_at.isoformat(),
            "recorded_at_utc": now.isoformat(),
            "source_ref": source_ref.strip(),
        }, events[-1]["record_sha256"])
        f.seek(0, os.SEEK_END)
        f.write((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    return public_status(events + [event])


def locked_returns(events: list[dict]) -> dict:
    """Construct returns only for final analysis; never for interim display."""
    observations = _observations(events)
    n_warm = len(events[0]["warmup_observations"])
    n_new = len(events) - 1
    if n_new < 2:
        raise ValueError("No realized prospective return yet")
    prices = total_return_index(observations)
    cash = np.asarray([obs["cash_yield_pct"] / (100 * TRADING_DAYS) for obs in observations[:-1]])
    weights = target_weights(prices, frozen_rule())
    net, traded = portfolio_returns(prices, cash, weights, cost_bps=REFERENCE_COST_BPS)
    first = n_warm
    # At the FIRST prospective executed close, portfolio is initially cash.
    # Remove the imaginary pre-experiment turnover and charge entry from zero.
    base = REFERENCE_COST_BPS / 10_000.0
    net[first] += base * (abs(weights[first] - weights[first - 1]) - abs(weights[first]))
    traded[first] = abs(weights[first])
    last_exclusive = first + n_new - 1
    asset = prices[1:] / prices[:-1] - 1.0
    benchmark = asset[first:last_exclusive].copy()
    benchmark[0] -= base  # SPY initial buy at the exact same entry close.
    selected = slice(first, last_exclusive)
    return {
        "strategy": net[selected], "benchmark": benchmark,
        "cash": cash[selected], "weights": weights[selected],
        "traded": traded[selected],
        "end_dates": [obs["date"] for obs in observations[n_warm + 1:]],
    }


def final_report(events: list[dict], ff5_archive, momentum_archive) -> dict:
    if public_status(events)["completed_return_sessions"] != REQUIRED_REALIZED_SESSIONS:
        raise ValueError("Prospective inference forbidden before 252 realized sessions")
    series = locked_returns(events)
    dates = [date.fromisoformat(x) for x in series["end_dates"]]
    if len(dates) != REQUIRED_REALIZED_SESSIONS:
        raise ValueError("Incorrect final session count")
    missing = [d for d in dates if d not in ff5_archive.dates_to_factors or
               d not in momentum_archive.dates_to_factors]
    if missing:
        raise ValueError(f"Official factors unavailable for {len(missing)} dates, first {missing[0]}")
    ff5 = [ff5_archive.dates_to_factors[d] for d in dates]
    mom = [momentum_archive.dates_to_factors[d] for d in dates]
    matrix = np.asarray([
        [row[x] for x in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")] + [tail["Mom"]]
        for row, tail in zip(ff5, mom)
    ])
    rf = np.asarray([row["RF"] for row in ff5])
    alpha = hac_regression(series["strategy"] - rf, matrix, REGRESSION_FACTORS,
                           lags=REGRESSION_HAC_LAGS)
    critical = NormalDist().inv_cdf((1 + BOOTSTRAP_CONFIDENCE) / 2)
    alpha["ci97_5_low"] = alpha["annualized_arithmetic_alpha"] - critical * alpha["annualized_alpha_se_hac"]
    alpha["ci97_5_high"] = alpha["annualized_arithmetic_alpha"] + critical * alpha["annualized_alpha_se_hac"]
    active = paired_circular_block_ci97_5(series["strategy"] - series["benchmark"])
    benchmark_traded = np.zeros(REQUIRED_REALIZED_SESSIONS)
    benchmark_traded[0] = 1.0
    return {
        "status": "PROSPECTIVE_PAPER_SIMULATION_COMPLETED_NO_LIVE_TRADES",
        "prospective_first_return_end_date": dates[0].isoformat(),
        "prospective_last_return_end_date": dates[-1].isoformat(),
        "sessions": len(dates),
        "journal_sha256": events[-1]["record_sha256"],
        "factor_sha256": {"ff5": ff5_archive.sha256, "mom": momentum_archive.sha256},
        "method": {
            "frozen_rule": FROZEN_RULE, "cost_bps": REFERENCE_COST_BPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_block_size": BOOTSTRAP_BLOCK_SIZE,
            "factor_model": "FF5 plus momentum, HAC(20), two-sided 97.5% interval",
            "execution": "next-close paper simulation, not observed broker fills",
            "source_limitations": "provided daily close/dividend/split/yield data; not audited",
        },
        "strategy_net": period_metrics(series["strategy"], series["cash"],
                                       series["traded"], series["weights"]),
        "spy_buy_hold_net": period_metrics(series["benchmark"], series["cash"],
                                           benchmark_traded, np.ones(len(dates))),
        "net_minus_spy_active_mean": active,
        "ff5_momentum_alpha": alpha,
        "primary_success_both": bool(active["ci97_5_low"] > 0 and alpha["ci97_5_low"] > 0),
        "warnings": [
            "This is close-based prospective PAPER simulation, not an executable fill record",
            "Factor data release vintage must be recorded; revised factor datasets can change regression",
            "Assumed 5-bp costs and approximate ^IRX cash accrual are not audited",
            "Statistical results do not guarantee future returns or causally identify alpha",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Prepare after 11/13 close, before 11/16 open")
    init.add_argument("--warmup", type=Path, required=True)
    init.add_argument("--source-id", type=str, required=True)
    record = sub.add_parser("record", help="Append one same-session after-close quote")
    record.add_argument("--date", type=date.fromisoformat, required=True)
    record.add_argument("--close", type=float, required=True)
    record.add_argument("--dividend-per-share", type=float, required=True)
    record.add_argument("--split-ratio", type=float, required=True)
    record.add_argument("--cash-yield-pct", type=float, required=True)
    record.add_argument("--observed-at-utc", type=datetime.fromisoformat, required=True)
    record.add_argument("--source-ref", type=str, required=True)
    sub.add_parser("status", help="Audit journal and report progress, not interim PnL")
    finish = sub.add_parser("finalize", help="One-time inference after 252 completed returns")
    finish.add_argument("--ff5-zip", type=Path, required=True)
    finish.add_argument("--momentum-zip", type=Path, required=True)
    finish.add_argument("--output", type=Path, default=Path("experiments/prospective_final.json"))
    for command in (init, record, sub.choices["status"], finish):
        command.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    args = parser.parse_args()
    if args.command == "init":
        status = initialize(args.warmup, args.journal, args.source_id)
    elif args.command == "record":
        obs = make_observation(args.date, args.close, args.dividend_per_share,
                               args.split_ratio, args.cash_yield_pct)
        status = append_close(args.journal, obs, args.observed_at_utc, args.source_ref)
    elif args.command == "status":
        status = public_status(read_journal(args.journal))
    else:
        events = read_journal(args.journal)
        report = final_report(
            events,
            load_french_archive(args.ff5_zip, "ff5", verify_pinned=False),
            load_french_archive(args.momentum_zip, "mom", verify_pinned=False),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print("Final, pre-specified paper simulation report:", args.output)
        return
    print(json.dumps(status, indent=2, sort_keys=True))
    print("PAPER SIMULATION ONLY. No order placed; no interim performance disclosed.")


if __name__ == "__main__":
    main()
