"""Run the FROZEN, pre-registered QQQ/IWM/EFA historical replication ONCE.

Read INDEPENDENT_REPLICATION_PROTOCOL.md before interpreting the results.
No model selection, parameter search, or post-hoc modification is performed.
Historical cross-asset observations are correlated with the inspected SPY
period and cannot establish independent prospective market alpha.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import NormalDist

import numpy as np

from factor_alpha import (
    FROZEN_STRATEGY,
    _aligned_arrays,
    hac_regression,
    load_french_archive,
)
from fetch_factor_data import ARCHIVES, FACTOR_SOURCE_ROOT
from fetch_market_data import EXPECTED_SHA256
from market_backtest import (
    CANDIDATES,
    COST_SENSITIVITY_BPS,
    FINAL_FIRST,
    REFERENCE_COST_BPS,
    TRADING_DAYS,
    PriceHistory,
    load_adjusted_prices,
    period_mask,
    period_metrics,
    portfolio_returns,
    target_weights,
)


PRE_REGISTRATION_COMMIT = "733a75afe72826f4163cca002ed3b12ac2c53bf9"
PROTOCOL_FILE = "INDEPENDENT_REPLICATION_PROTOCOL.md"
SYMBOLS = ("QQQ", "IWM", "EFA")
BENCHMARK = "SPY"
EXPECTED_FIRST = date(2009, 1, 2)
EXPECTED_LAST = date(2026, 8, 10)
TEST_FIRST = date(2023, 1, 1)
TEST_LAST_EXCLUSIVE = date(2026, 8, 11)
BOOTSTRAP_CONFIDENCE = 0.975  # PREDECLARED two-sided 97.5% interval
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_BLOCK_SIZE = 20
BOOTSTRAP_SEED = 20261114
REGRESSION_HAC_LAGS = 20
REGRESSION_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")


def paired_circular_block_ci97_5(daily_differences: np.ndarray) -> dict:
    """Predeclared two-sided 97.5% interval for annual mean active return."""
    diffs = np.asarray(daily_differences, dtype=float)
    if diffs.ndim != 1 or diffs.size < 200 or not np.all(np.isfinite(diffs)):
        raise ValueError("Paired bootstrap needs >=200 finite daily differences")
    n = diffs.size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = ceil(n / BOOTSTRAP_BLOCK_SIZE)
    offsets = np.arange(BOOTSTRAP_BLOCK_SIZE)
    means = np.empty(BOOTSTRAP_REPETITIONS)
    for i in range(BOOTSTRAP_REPETITIONS):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        means[i] = TRADING_DAYS * float(np.mean(diffs[indices]))
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "annualized_arithmetic_mean_gap": float(TRADING_DAYS * np.mean(diffs)),
        "ci97_5_low": float(np.quantile(means, tail)),
        "ci97_5_high": float(np.quantile(means, 1.0 - tail)),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "block_size_sessions": BOOTSTRAP_BLOCK_SIZE,
        "seed": BOOTSTRAP_SEED,
        "method": "paired circular moving-block bootstrap",
    }


def holm_two_sided_adjusted(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; valid for dependent hypotheses."""
    if not p_values or any(not np.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Expected finite p-values in [0, 1]")
    ordered = sorted(p_values, key=lambda name: p_values[name])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    floor = 0.0
    for index, name in enumerate(ordered):
        floor = max(floor, min(1.0, p_values[name] * (m - index)))
        adjusted[name] = floor
    return adjusted


def _benchmark_enter_at_test_start(
    prices: np.ndarray, cash: np.ndarray, test_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold a single ETF; pay 5bp at the test-boundary initial purchase."""
    weights = np.ones_like(cash)
    net, _ = portfolio_returns(prices, cash, weights, cost_bps=0.0)
    traded = np.zeros_like(cash)
    first = int(np.flatnonzero(test_mask)[0])
    net[first] -= REFERENCE_COST_BPS / 10_000.0
    traded[first] = 1.0
    return net, traded, weights


def _verify_histories(histories: dict[str, PriceHistory], expected_sha: str) -> tuple[date, ...]:
    if set(histories) != {BENCHMARK, *SYMBOLS}:
        raise ValueError("Required exactly SPY, QQQ, IWM and EFA histories")
    reference = histories[BENCHMARK]
    if (reference.dates[0], reference.dates[-1]) != (EXPECTED_FIRST, EXPECTED_LAST):
        raise ValueError("Source history boundaries differ from preregistration")
    for symbol, history in histories.items():
        if history.symbol != symbol or history.source_sha256 != expected_sha:
            raise ValueError(f"{symbol}: wrong symbol or unpinned market source")
        if history.dates != reference.dates:
            raise ValueError(f"{symbol}: non-identical market session dates")
        if not np.array_equal(history.annual_cash_yield_pct, reference.annual_cash_yield_pct):
            raise ValueError(f"{symbol}: cash quotes differ across ETF histories")
    return reference.dates


def run_replication(
    histories: dict[str, PriceHistory],
    factors: dict,
    *,
    expected_source_sha: str = EXPECTED_SHA256,
) -> dict:
    """Compute all predeclared outcomes without optimizing any rule."""
    dates = _verify_histories(histories, expected_source_sha)
    if set(factors) != {"ff3", "ff5", "mom"}:
        raise ValueError("Required exactly FF3, FF5 and momentum factors")
    if REFERENCE_COST_BPS != 5.0 or COST_SENSITIVITY_BPS != (0., 5., 10., 20.):
        raise ValueError("Existing cost assumptions differ from preregistration")
    frozen = next((rule for rule in CANDIDATES if rule.name == FROZEN_STRATEGY), None)
    if frozen is None or (frozen.kind, frozen.lookback, frozen.annual_target_volatility) != ("volatility", 20, 0.10):
        raise ValueError("Frozen rule parameters differ from preregistration")
    mask = period_mask(dates, TEST_FIRST, TEST_LAST_EXCLUSIVE)
    if int(np.sum(mask)) < 200 or dates[-1] >= TEST_LAST_EXCLUSIVE:
        raise ValueError("Unexpected test interval or insufficient sessions")
    ref = histories[BENCHMARK]
    cash = ref.daily_cash_return
    spy_net, spy_traded, spy_weights = _benchmark_enter_at_test_start(
        ref.adjusted_close, cash, mask
    )
    spy_metrics = period_metrics(
        spy_net[mask], cash[mask], spy_traded[mask], spy_weights[mask]
    )
    per_symbol = {}
    for symbol in SYMBOLS:
        hist = histories[symbol]
        weights = target_weights(hist.adjusted_close, frozen)
        net, traded = portfolio_returns(
            hist.adjusted_close, cash, weights, cost_bps=REFERENCE_COST_BPS
        )
        own_benchmark, own_traded, own_weights = _benchmark_enter_at_test_start(
            hist.adjusted_close, cash, mask
        )
        aligned = _aligned_arrays(dates, mask, net, ref.adjusted_close, factors)
        design = np.column_stack((aligned["ff5"], aligned["mom"]))
        alpha = hac_regression(
            aligned["y"], design, REGRESSION_FACTORS, lags=REGRESSION_HAC_LAGS
        )
        # The generic regression reports a nominal 95% interval. Preserve
        # that, and add the *predeclared* 97.5% two-sided confidence interval.
        critical = NormalDist().inv_cdf((1.0 + BOOTSTRAP_CONFIDENCE) / 2.0)
        estimate = alpha["annualized_arithmetic_alpha"]
        standard_error = alpha["annualized_alpha_se_hac"]
        alpha["ci97_5_low"] = estimate - critical * standard_error
        alpha["ci97_5_high"] = estimate + critical * standard_error
        alpha["confidence_preregistered"] = BOOTSTRAP_CONFIDENCE

        vs_spy = paired_circular_block_ci97_5(net[mask] - spy_net[mask])
        vs_own = paired_circular_block_ci97_5(net[mask] - own_benchmark[mask])
        cost_sensitivity = []
        for bps in COST_SENSITIVITY_BPS:
            alternative_net, alternative_traded = portfolio_returns(
                hist.adjusted_close, cash, weights, cost_bps=bps
            )
            cost_sensitivity.append({
                "one_way_cost_bps": bps,
                "net_strategy_metrics": period_metrics(
                    alternative_net[mask], cash[mask],
                    alternative_traded[mask], weights[mask],
                ),
            })
        per_symbol[symbol] = {
            "net_strategy_metrics": period_metrics(net[mask], cash[mask], traded[mask], weights[mask]),
            "same_etf_buy_and_hold_metrics": period_metrics(
                own_benchmark[mask], cash[mask], own_traded[mask], own_weights[mask]
            ),
            "spy_buy_and_hold_metrics": spy_metrics,
            "net_strategy_minus_spy_buy_hold": vs_spy,
            "net_strategy_minus_own_etf_buy_hold": vs_own,
            "ff5_plus_momentum_alpha": alpha,
            "cost_sensitivity": cost_sensitivity,
        }

    primary = per_symbol["QQQ"]
    beat_spy_ci = primary["net_strategy_minus_spy_buy_hold"]["ci97_5_low"] > 0
    positive_alpha_ci = primary["ff5_plus_momentum_alpha"]["ci97_5_low"] > 0
    secondary_p = {
        symbol: per_symbol[symbol]["ff5_plus_momentum_alpha"]["p_two_sided_nominal"]
        for symbol in ("IWM", "EFA")
    }
    holm = holm_two_sided_adjusted(secondary_p)
    for symbol in holm:
        per_symbol[symbol]["ff5_plus_momentum_alpha"]["p_holm_secondary_two_tests"] = holm[symbol]

    protocol_path = Path(__file__).with_name(PROTOCOL_FILE)
    return {
        "status": "PRE_REGISTERED_HISTORICAL_CROSS_ASSET_REPLICATION_COMPLETE_NOT_PROSPECTIVE_ALPHA",
        "previously_committed_protocol": {
            "github_commit": PRE_REGISTRATION_COMMIT,
            "file": PROTOCOL_FILE,
            "file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "primary": "QQQ",
            "secondary": ["IWM", "EFA"],
        },
        "data": {
            "source_sha256": expected_source_sha,
            "source_is_third_party_unverified_adjusted_close_mirror": True,
            "first_price": dates[0].isoformat(),
            "last_price": dates[-1].isoformat(),
            "first_test_return_end_date": dates[int(np.flatnonzero(mask)[0]) + 1].isoformat(),
            "last_test_return_end_date": dates[int(np.flatnonzero(mask)[-1]) + 1].isoformat(),
            "test_sessions": int(np.sum(mask)),
            "test_date_alignment": "all four ETF calendars exactly identical; no missing factor observations",
            "factor_source": FACTOR_SOURCE_ROOT,
            "factor_zip_sha256": {name: factors[name].sha256 for name in ("ff3", "ff5", "mom")},
        },
        "fixed_method": {
            "strategy": asdict(frozen),
            "signal_execution": "after close t observed; execute close t+1; earn t+1 to t+2",
            "cost_bps": REFERENCE_COST_BPS,
            "buy_hold_test_entry_cost_bps": REFERENCE_COST_BPS,
            "cash_proxy": "^IRX 13-week annual discount yield / (100*252); approximate",
            "factor_rf": "official one-month FF daily risk-free factor; differs from strategy cash proxy",
            "alpha": "FF5+Mom, annualized arithmetic intercept; Newey-West HAC 20 lags",
            "co_primary_two_sided_confidence": BOOTSTRAP_CONFIDENCE,
            "bootstrap": "5,000 paired circular 20-session blocks; seed 20261114",
            "cost_sensitivity_bps": list(COST_SENSITIVITY_BPS),
            "cross_asset_warning": "same dates and market exposures as seen SPY period; NOT fully independent",
            "prospective_warning": "no live trades, no audited costs or prospective findings",
        },
        "primary_QQQ_two_required_conditions": {
            "qqq_strategy_vs_spy_CI_lower_positive": bool(beat_spy_ci),
            "qqq_strategy_ff5_momentum_alpha_CI_lower_positive": bool(positive_alpha_ci),
            "both_conditions_met": bool(beat_spy_ci and positive_alpha_ci),
            "judgment": (
                "PRE_REGISTERED_HISTORICAL_POSITIVE_RISK_ADJUSTED_OUTPERFORMANCE"
                if beat_spy_ci and positive_alpha_ci
                else "PRE_REGISTERED_PRIMARY_SUCCESS_CRITERIA_NOT_MET"
            ),
        },
        "secondary_factor_p_holm_two_sided": holm,
        "results_by_symbol": per_symbol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-csv", type=Path, default=Path("data/raw/daily_adjclose.csv"))
    parser.add_argument("--factors-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("experiments/cross_asset_replication.json"))
    args = parser.parse_args()
    histories = {
        symbol: load_adjusted_prices(args.market_csv, symbol=symbol)
        for symbol in (BENCHMARK, *SYMBOLS)
    }
    factors = {
        name: load_french_archive(args.factors_dir / spec["local"], name, verify_pinned=True)
        for name, spec in ARCHIVES.items()
    }
    report = run_replication(histories, factors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("PRE-REGISTERED ONE-TIME HISTORICAL REPLICATION; NOT PROSPECTIVE EVIDENCE")
    for symbol in SYMBOLS:
        entry = report["results_by_symbol"][symbol]
        active = entry["net_strategy_minus_spy_buy_hold"]
        alpha = entry["ff5_plus_momentum_alpha"]
        print(
            f"{symbol}: strategy CAGR {entry['net_strategy_metrics']['annualized_return']:+.2%}; "
            f"SPY CAGR {entry['spy_buy_and_hold_metrics']['annualized_return']:+.2%}; "
            f"active mean {active['annualized_arithmetic_mean_gap']:+.2%} "
            f"(97.5% CI {active['ci97_5_low']:+.2%}, {active['ci97_5_high']:+.2%}); "
            f"FF5+Mom alpha {alpha['annualized_arithmetic_alpha']:+.2%} "
            f"(97.5% CI {alpha['ci97_5_low']:+.2%}, {alpha['ci97_5_high']:+.2%})"
        )
    print("Primary status:", report["primary_QQQ_two_required_conditions"]["judgment"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
