"""Leakage-aware, cost-adjusted historical ETF backtest and walk-forward study.

Research only: not trading advice, not live execution, and not proof of alpha.
Signals computed from the close are EXECUTED at the following day's close;
returns are realized during the subsequent close-to-close interval.
The strategy family and all evaluation windows are fixed in this source code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from math import ceil, sqrt
from pathlib import Path

import numpy as np

from fetch_market_data import (
    EXPECTED_SHA256,
    SOURCE_REPO,
    SOURCE_REVISION,
)


TRADING_DAYS = 252
TRAIN_FIRST = date(2010, 1, 1)
VALIDATION_FIRST = date(2019, 1, 1)
FINAL_FIRST = date(2023, 1, 1)
REFERENCE_COST_BPS = 5.0
COST_SENSITIVITY_BPS = (0.0, 5.0, 10.0, 20.0)
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_REPETITIONS = 1500
BOOTSTRAP_SEED = 20261113
# Verified against the official NYSE 2026 trading-holiday calendar. The
# upstream wide CSV has a row for this non-trading day with no ETF close.
KNOWN_EXCHANGE_CLOSED_DATES = frozenset({date(2026, 5, 25)})


@dataclass(frozen=True)
class PriceHistory:
    dates: tuple[date, ...]
    adjusted_close: np.ndarray
    annual_cash_yield_pct: np.ndarray
    source_sha256: str = ""
    carried_forward_cash_yield_days: int = 0
    skipped_known_exchange_holidays: int = 0
    symbol: str = "SPY"

    @property
    def daily_cash_return(self) -> np.ndarray:
        # 13-week Treasury bill discount yields are quoted as annual percent.
        # This is only an APPROXIMATE daily interest accrual, not realized bills.
        return self.annual_cash_yield_pct[:-1] / (100.0 * TRADING_DAYS)


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    lookback: int
    annual_target_volatility: float = 0.0


# This set is predeclared; no choices are made using final-holdout returns.
CANDIDATES: tuple[Strategy, ...] = (
    Strategy("momentum_20", "momentum", 20),
    Strategy("momentum_60", "momentum", 60),
    Strategy("momentum_120", "momentum", 120),
    Strategy("mean_reversion_3", "reversion", 3),
    Strategy("mean_reversion_5", "reversion", 5),
    Strategy("mean_reversion_10", "reversion", 10),
    Strategy("volatility_target_20", "volatility", 20, 0.10),
)


def load_adjusted_prices(
    csv_path: Path,
    *,
    symbol: str = "SPY",
    yield_column: str = "^IRX",
    first_date: date = date(2009, 1, 1),
) -> PriceHistory:
    """Read wide Date/symbol/yield CSV; reject duplicates and internal gaps.

    The data source includes a final trailing row with missing ETF prices and
    one documented 2026 NYSE holiday row. These are discarded. Any OTHER
    internal missing close fails rather than quietly forward-filling prices.
    The ^IRX field is the prior-close yield used for subsequent cash accrual.
    """
    dates: list[date] = []
    closes: list[float] = []
    rates: list[float] = []
    last_known_rate: float | None = None
    consecutive_missing_rates = 0
    carried_forward_rates = 0
    known_holidays_skipped = 0
    digest = hashlib.sha256()
    with csv_path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    trailing_gap = False
    previous_date: date | None = None
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"Date", symbol, yield_column}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"CSV requires Date, {symbol}, and {yield_column}")
        for row in reader:
            current = date.fromisoformat(row["Date"])
            if previous_date is not None and current <= previous_date:
                raise ValueError("CSV dates must be strictly increasing without duplicates")
            previous_date = current
            if current < first_date:
                continue
            raw_close = (row[symbol] or "").strip()
            if not raw_close:
                if current in KNOWN_EXCHANGE_CLOSED_DATES:
                    known_holidays_skipped += 1
                    continue
                if dates:
                    trailing_gap = True
                continue
            if trailing_gap:
                raise ValueError("Internal missing market price; do not forward fill")
            raw_rate = (row[yield_column] or "").strip()
            if not raw_rate:
                # Treasury markets can close on a holiday when NYSE is open.
                # Reuse ONLY the last published rate, never a future observation.
                consecutive_missing_rates += 1
                if last_known_rate is None or consecutive_missing_rates > 5:
                    raise ValueError(f"Missing/stale {yield_column} on {current}")
                annual_rate_pct = last_known_rate
                carried_forward_rates += 1
            else:
                annual_rate_pct = float(raw_rate)
                last_known_rate = annual_rate_pct
                consecutive_missing_rates = 0
            price = float(raw_close)
            if (
                not np.isfinite(price)
                or price <= 0
                or not np.isfinite(annual_rate_pct)
            ):
                raise ValueError(f"Invalid close or yield on {current}")
            dates.append(current)
            closes.append(price)
            rates.append(annual_rate_pct)
    if len(dates) < 252:
        raise ValueError("Need at least 252 market-price observations")
    return PriceHistory(
        dates=tuple(dates),
        adjusted_close=np.asarray(closes, dtype=float),
        annual_cash_yield_pct=np.asarray(rates, dtype=float),
        source_sha256=digest.hexdigest(),
        carried_forward_cash_yield_days=carried_forward_rates,
        skipped_known_exchange_holidays=known_holidays_skipped,
        symbol=symbol,
    )


def target_weights(prices: np.ndarray, strategy: Strategy) -> np.ndarray:
    """Execute at the first NEXT close after a signal becomes observable.

    Signal s[t] is computed after observing prices[t]. Execution of that
    signal is delayed until close t+1: weights[t+1] = s[t], held from
    close t+1 to t+2. This extra bar prevents same-close execution bias.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 3 or np.any(prices <= 0):
        raise ValueError("Expected a 1D strictly-positive price series")
    if (
        not np.all(np.isfinite(prices))
        or strategy.lookback < 2
        or strategy.kind not in ("momentum", "reversion", "volatility")
    ):
        raise ValueError("Invalid prices or lookback")
    signals = np.zeros(prices.size - 1, dtype=float)
    for t in range(strategy.lookback, prices.size - 1):
        if strategy.kind == "momentum":
            signals[t] = float(prices[t] > prices[t - strategy.lookback])
        elif strategy.kind == "reversion":
            signals[t] = float(prices[t] < prices[t - strategy.lookback])
        elif strategy.kind == "volatility":
            if not 0 < strategy.annual_target_volatility <= 1:
                raise ValueError("Target annual volatility must be between 0 and 1")
            past = prices[
                t - strategy.lookback + 1 : t + 1
            ] / prices[t - strategy.lookback : t] - 1.0
            estimated_vol = float(np.std(past, ddof=1)) * sqrt(TRADING_DAYS)
            signals[t] = min(1.0, strategy.annual_target_volatility / max(
                estimated_vol, 1e-12
            ))
        else:
            raise ValueError(f"Unknown strategy kind: {strategy.kind}")
    return np.r_[0.0, signals[:-1]]


def portfolio_returns(
    prices: np.ndarray,
    daily_cash_returns: np.ndarray,
    weights: np.ndarray,
    *,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple daily net portfolio returns and traded notional fractions.

    A change from zero to fully invested costs one-way cost_bps basis points.
    Holding one unit of exposure earns the adjusted-close asset return;
    uninvested weight accrues an approximate T-bill cash return.
    """
    px = np.asarray(prices, dtype=float)
    rf = np.asarray(daily_cash_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (
        px.ndim != 1
        or rf.shape != (px.size - 1,)
        or w.shape != rf.shape
        or px.size < 3
        or np.any(px <= 0)
        or not np.all(np.isfinite(px))
        or not np.all(np.isfinite(rf))
        or not np.all(np.isfinite(w))
        or np.any((w < 0) | (w > 1))
        or cost_bps < 0
        or not np.isfinite(cost_bps)
    ):
        raise ValueError("Invalid aligned prices, cash returns, weights, or costs")
    asset_return = px[1:] / px[:-1] - 1.0
    previous_weight = np.r_[0.0, w[:-1]]
    traded = np.abs(w - previous_weight)
    net = w * asset_return + (1.0 - w) * rf - (cost_bps / 10_000) * traded
    return net, traded


def period_mask(
    dates: tuple[date, ...], first: date, last_exclusive: date
) -> np.ndarray:
    return np.fromiter(
        (first <= d < last_exclusive for d in dates[1:]),
        dtype=bool,
        count=len(dates) - 1,
    )


def annualized_sharpe_excess(
    net_returns: np.ndarray, cash_returns: np.ndarray
) -> float | None:
    excess = np.asarray(net_returns) - np.asarray(cash_returns)
    if excess.ndim != 1 or excess.size < 2 or not np.all(np.isfinite(excess)):
        raise ValueError("Need at least two finite excess daily returns")
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return None
    return float(np.mean(excess) / std * sqrt(TRADING_DAYS))


def period_metrics(
    net_returns: np.ndarray,
    cash_returns: np.ndarray,
    traded: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int | None]:
    r = np.asarray(net_returns, dtype=float)
    if r.size < 30 or np.any(r <= -1) or not np.all(np.isfinite(r)):
        raise ValueError("Period needs 30 valid days with no total loss")
    equity = np.r_[1.0, np.cumprod(1.0 + r)]
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    annualized_return = float(np.expm1(np.log1p(r).sum() * TRADING_DAYS / r.size))
    return {
        "days": int(r.size),
        "total_return": float(equity[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": float(np.std(r, ddof=1) * sqrt(TRADING_DAYS)),
        "sharpe_vs_cash": annualized_sharpe_excess(r, cash_returns),
        "max_drawdown": float(np.min(drawdown)),
        "annualized_turnover": float(
            np.sum(traded) * TRADING_DAYS / r.size
        ),
        "mean_invested_fraction": float(np.mean(weights)),
    }


def paired_block_bootstrap_mean_ci(
    daily_differences: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    block_size: int = BOOTSTRAP_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Circular moving-block bootstrap for annualized arithmetic mean gap.

    Resamples the PAIRWISE active-minus-benchmark DAILY return differences,
    preserving local dependence. This is not a CI for CAGR or future alpha.
    """
    differences = np.asarray(daily_differences, dtype=float)
    if differences.ndim != 1 or differences.size < 30:
        raise ValueError("Need at least 30 daily paired differences")
    if not np.all(np.isfinite(differences)) or repetitions < 100 or block_size < 1:
        raise ValueError("Invalid bootstrap values or parameters")
    rng = np.random.default_rng(seed)
    n = differences.size
    n_blocks = ceil(n / block_size)
    offsets = np.arange(block_size)
    means = np.empty(repetitions)
    for b in range(repetitions):
        starts = rng.integers(0, n, size=n_blocks)
        sampled_indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[b] = TRADING_DAYS * float(np.mean(differences[sampled_indices]))
    return {
        "point_estimate": float(TRADING_DAYS * np.mean(differences)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_repetitions": repetitions,
        "block_size_days": block_size,
        "seed": seed,
    }


def _score(
    net: np.ndarray, daily_rf: np.ndarray, mask: np.ndarray
) -> float:
    value = annualized_sharpe_excess(net[mask], daily_rf[mask])
    return -float("inf") if value is None else value


def training_only_selection(
    candidate_net: dict[str, np.ndarray],
    daily_rf: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Winner chosen WITHOUT ANY INPUT from validation or final holdout."""
    if not candidate_net or not np.any(train_mask):
        raise ValueError("Need nonempty candidates and training dates")
    scores = {
        name: _score(net, daily_rf, train_mask) for name, net in candidate_net.items()
    }
    if not any(np.isfinite(v) for v in scores.values()):
        raise ValueError("No candidate has a finite training Sharpe")
    return max(scores, key=scores.get), scores


def run_market_research(
    history: PriceHistory,
    *,
    cost_bps: float = REFERENCE_COST_BPS,
) -> dict:
    """Predeclared SPY experiment; evaluate only the frozen winner on final."""
    dates = history.dates
    prices = history.adjusted_close
    cash = history.daily_cash_return
    training = period_mask(dates, TRAIN_FIRST, VALIDATION_FIRST)
    validation = period_mask(dates, VALIDATION_FIRST, FINAL_FIRST)
    final = period_mask(dates, FINAL_FIRST, date(2100, 1, 1))
    if min(int(training.sum()), int(validation.sum()), int(final.sum())) < 200:
        raise ValueError("Require >= 200 daily returns in each split")

    candidate_weights = {
        candidate.name: target_weights(prices, candidate) for candidate in CANDIDATES
    }
    candidate_backtests = {
        name: portfolio_returns(prices, cash, weights, cost_bps=cost_bps)
        for name, weights in candidate_weights.items()
    }
    candidate_net = {
        name: results[0] for name, results in candidate_backtests.items()
    }
    winner, training_scores = training_only_selection(candidate_net, cash, training)
    winner_weights = candidate_weights[winner]
    selected_net, selected_traded = candidate_backtests[winner]
    benchmark_weights = np.ones_like(cash)
    benchmark_net, benchmark_traded = portfolio_returns(
        prices, cash, benchmark_weights, cost_bps=cost_bps
    )

    def metrics(net: np.ndarray, traded: np.ndarray, weights: np.ndarray, mask):
        return period_metrics(net[mask], cash[mask], traded[mask], weights[mask])

    selected_periods = {
        "training": metrics(selected_net, selected_traded, winner_weights, training),
        "validation": metrics(selected_net, selected_traded, winner_weights, validation),
        "final_holdout": metrics(selected_net, selected_traded, winner_weights, final),
    }
    benchmark_periods = {
        "training": metrics(
            benchmark_net, benchmark_traded, benchmark_weights, training
        ),
        "validation": metrics(
            benchmark_net, benchmark_traded, benchmark_weights, validation
        ),
        "final_holdout": metrics(
            benchmark_net, benchmark_traded, benchmark_weights, final
        ),
    }

    # All of these folds END in 2022. No walk-forward selection sees final data.
    folds = []
    for test_year in range(2018, 2023):
        previous_five_years = period_mask(
            dates, date(test_year - 5, 1, 1), date(test_year, 1, 1)
        )
        following_year = period_mask(
            dates, date(test_year, 1, 1), date(test_year + 1, 1, 1)
        )
        fold_winner, _ = training_only_selection(
            candidate_net, cash, previous_five_years
        )
        fold_net, fold_turnover = candidate_backtests[fold_winner]
        fold_metrics = metrics(
            fold_net, fold_turnover, candidate_weights[fold_winner], following_year
        )
        fold_benchmark = metrics(
            benchmark_net, benchmark_traded, benchmark_weights, following_year
        )
        folds.append(
            {
                "test_year": test_year,
                "training_first_year": test_year - 5,
                "selected_strategy": fold_winner,
                "net_annualized_return": fold_metrics["annualized_return"],
                "net_sharpe_vs_cash": fold_metrics["sharpe_vs_cash"],
                "benchmark_annualized_return": fold_benchmark["annualized_return"],
            }
        )

    # Fixed-winner cost stress tests: NO reselection at alternate cost levels.
    stress_tests = []
    for alternative_bps in COST_SENSITIVITY_BPS:
        alt_net, alt_traded = portfolio_returns(
            prices, cash, winner_weights, cost_bps=alternative_bps
        )
        stress_tests.append(
            {
                "cost_bps_one_way": alternative_bps,
                "selected_final_metrics": metrics(
                    alt_net, alt_traded, winner_weights, final
                ),
            }
        )

    return {
        "status": "HISTORICAL_RESEARCH_ONLY_NO_VERIFIED_ALPHA",
        "source": {
            "provider": (
                "third-party GitHub yfinance adjusted-close mirror"
                if history.source_sha256 == EXPECTED_SHA256
                else "user-supplied market CSV; provenance not independently verified"
            ),
            "repository_url": (
                SOURCE_REPO if history.source_sha256 == EXPECTED_SHA256 else None
            ),
            "pinned_revision": (
                SOURCE_REVISION if history.source_sha256 == EXPECTED_SHA256 else None
            ),
            "source_sha256": history.source_sha256,
            "matches_pinned_snapshot": history.source_sha256 == EXPECTED_SHA256,
            "price_basis": (
                "adjusted close, according to mirror; not audited"
                if history.source_sha256 == EXPECTED_SHA256
                else "close interpretation depends on supplied data"
            ),
            "symbol": history.symbol,
            "cash_proxy": "^IRX close / 100 / 252, approximate",
            "previous_observed_cash_yield_carry_days": (
                history.carried_forward_cash_yield_days
            ),
            "known_exchange_holiday_rows_skipped": (
                history.skipped_known_exchange_holidays
            ),
            "first_loaded": dates[0].isoformat(),
            "last_loaded": dates[-1].isoformat(),
            "rows_loaded": len(dates),
        },
        "method": {
            "candidate_names": [asdict(x) for x in CANDIDATES],
            "training_period": "2010-01-01 through 2018-12-31",
            "validation_period": "2019-01-01 through 2022-12-31",
            "final_period": "2023-01-01 through last valid price",
            "selection": "maximize training-only net annualized Sharpe vs cash",
            "rebalancing": (
                "signal formed after close t; execute at close t+1; "
                "earn return from close t+1 to close t+2"
            ),
            "long_only_fully_funded": True,
            "reference_one_way_cost_bps": cost_bps,
            "cost_components": "aggregate assumed spread, slippage, commissions",
            "risk_free": "approximate daily ^IRX; not an executable return",
            "missing_yields": "past-value carry forward, maximum five sessions",
            "missing_prices": (
                "skip only 2026-05-25 verified NYSE holiday and trailing blanks; "
                "reject other internal gaps"
            ),
            "folds": "2018-2022 yearly; trailing five calendar years training",
            "holdout_reuse_warning": "do not retune after examining these results",
        },
        "training_sharpe_by_candidate": training_scores,
        "selected_strategy": winner,
        "selected": selected_periods,
        "benchmark_buy_hold": benchmark_periods,
        "walk_forward_development_only": folds,
        "final_cost_sensitivity": stress_tests,
        "final_annualized_arithmetic_mean_gap_ci95": {
            "selected_minus_buy_hold": paired_block_bootstrap_mean_ci(
                selected_net[final] - benchmark_net[final]
            ),
            "selected_minus_cash": paired_block_bootstrap_mean_ci(
                selected_net[final] - cash[final]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("data/raw/daily_adjclose.csv")
    )
    parser.add_argument("--cost-bps", type=float, default=REFERENCE_COST_BPS)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/real_market_study.json")
    )
    args = parser.parse_args()
    history = load_adjusted_prices(args.csv)
    report = run_market_research(history, cost_bps=args.cost_bps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    chosen = report["selected_strategy"]
    print(f"Predeclared candidates: {len(CANDIDATES)} | training-only winner: {chosen}")
    for period in ("training", "validation", "final_holdout"):
        metrics = report["selected"][period]
        base = report["benchmark_buy_hold"][period]
        print(
            f"{period}: strategy net CAGR {metrics['annualized_return']:.2%}, "
            f"Sharpe {metrics['sharpe_vs_cash']:.3f}; "
            f"buy/hold CAGR {base['annualized_return']:.2%}, "
            f"Sharpe {base['sharpe_vs_cash']:.3f}"
        )
    ci = report["final_annualized_arithmetic_mean_gap_ci95"][
        "selected_minus_buy_hold"
    ]
    print(
        "Final annualized arithmetic mean active-minus-benchmark return: "
        f"{ci['point_estimate']:.2%} (block-bootstrap 95% CI "
        f"{ci['ci95_low']:.2%}, {ci['ci95_high']:.2%})"
    )
    print(f"Saved: {args.output}")
    print("Historical backtest only: no claim of verified real-money profitability.")


if __name__ == "__main__":
    main()
