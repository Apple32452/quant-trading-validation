"""Post-hoc factor alpha diagnostics of the EXISTING frozen SPY strategy.

This never searches for a new rule. The 2023-2026 SPY test was previously
inspected, so even a significant regression intercept would NOT provide new
independent evidence of alpha. Factor data are from official Dartmouth files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import io
import json
from math import erfc, sqrt
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from fetch_factor_data import ARCHIVES, FACTOR_SOURCE_ROOT
from fetch_market_data import EXPECTED_SHA256
from market_backtest import (
    CANDIDATES,
    FINAL_FIRST,
    REFERENCE_COST_BPS,
    TRAIN_FIRST,
    TRADING_DAYS,
    VALIDATION_FIRST,
    PriceHistory,
    load_adjusted_prices,
    period_mask,
    portfolio_returns,
    target_weights,
)


FROZEN_STRATEGY = "volatility_target_20"
FROZEN_RESEARCH_COMMIT = "dea5b9823947d2ad95280e16d0f2fa59d7599873"
EXPECTED_FIELDS = {
    "ff3": ("Mkt-RF", "SMB", "HML", "RF"),
    "ff5": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    "mom": ("Mom",),
}
HAC_LAGS = 5
REGRESSION_MODELS = (
    "SPY_market",
    "FF_market",
    "FF3",
    "FF5",
    "FF5_plus_momentum",
)


@dataclass(frozen=True)
class FactorArchive:
    name: str
    dates_to_factors: dict[date, dict[str, float]]
    sha256: str
    first_date: date
    last_date: date


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_french_archive(
    path: Path, name: str, *, verify_pinned: bool = False
) -> FactorArchive:
    """Parse the daily block of a zipped official factor CSV.

    The library places metadata before the header and a footer after the
    daily records. Factor values are expressed in PERCENT, and must be
    converted to decimal daily returns before the regressions.
    """
    if name not in ARCHIVES:
        raise ValueError(f"Unknown factor archive {name}")
    digest = _sha256(path)
    if verify_pinned and digest != ARCHIVES[name]["sha256"]:
        raise ValueError(
            f"{name} factor vintage changed: expected "
            f"{ARCHIVES[name]['sha256']}, got {digest}"
        )
    with ZipFile(path) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"{name}: expected exactly one CSV in the archive")
        content = zf.read(csv_members[0]).decode("utf-8-sig")
    rows = csv.reader(io.StringIO(content))
    required = EXPECTED_FIELDS[name]
    header: list[str] | None = None
    observations: dict[date, dict[str, float]] = {}
    for row in rows:
        cleaned = [field.strip() for field in row]
        if header is None:
            if cleaned and cleaned[0] == "" and set(required).issubset(cleaned[1:]):
                header = cleaned
            continue
        if not cleaned or not re.fullmatch(r"\d{8}", cleaned[0]):
            if observations:
                break
            continue
        current_date = date(
            int(cleaned[0][:4]), int(cleaned[0][4:6]), int(cleaned[0][6:])
        )
        if current_date in observations:
            raise ValueError(f"{name}: duplicate factor date {current_date}")
        if len(cleaned) != len(header):
            raise ValueError(f"{name}: wrong column count on {current_date}")
        values = {}
        for field in required:
            value = float(cleaned[header.index(field)])
            if not np.isfinite(value) or value in (-99.99, -999.0):
                raise ValueError(f"{name}: missing factor on {current_date}")
            values[field] = value / 100.0
        observations[current_date] = values
    if header is None or len(observations) < 1000:
        raise ValueError(f"{name}: no valid long daily factor block")
    ordered = list(observations)
    if ordered != sorted(ordered):
        raise ValueError(f"{name}: factor dates are not increasing")
    return FactorArchive(
        name=name,
        dates_to_factors=observations,
        sha256=digest,
        first_date=ordered[0],
        last_date=ordered[-1],
    )


def hac_regression(
    excess_daily_returns: np.ndarray,
    factor_daily_returns: np.ndarray,
    factor_names: tuple[str, ...],
    *,
    lags: int = HAC_LAGS,
) -> dict:
    """OLS intercept with Newey-West/Bartlett HAC covariance.

    Outputs annualized ARITHMETIC alpha and a nominal 95% normal-approximation
    interval. An intercept is model-dependent, not a guaranteed return.
    """
    y = np.asarray(excess_daily_returns, dtype=float)
    factors = np.asarray(factor_daily_returns, dtype=float)
    if y.ndim != 1 or factors.shape != (y.size, len(factor_names)):
        raise ValueError("Factor design must be (n_days, n_factors)")
    n = y.size
    k = len(factor_names) + 1
    if n < max(100, 10 * k) or lags < 0 or lags >= n:
        raise ValueError("Insufficient data or invalid HAC lag count")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(factors)):
        raise ValueError("Regression inputs must all be finite")

    X = np.column_stack((np.ones(n), factors))
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank != k:
        raise ValueError("Regressors are linearly dependent")
    residual = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    score = residual[:, None] * X
    meat = score.T @ score
    for lag in range(1, lags + 1):
        cross = score[lag:].T @ score[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (cross + cross.T)
    covariance = (n / (n - k)) * xtx_inv @ meat @ xtx_inv
    variance_alpha = float(covariance[0, 0])
    if variance_alpha <= 0 or not np.isfinite(variance_alpha):
        raise ValueError("Nonpositive/invalid HAC intercept variance")
    se_daily = sqrt(variance_alpha)
    z = float(beta[0] / se_daily)
    alpha_ann = float(beta[0] * TRADING_DAYS)
    se_ann = se_daily * TRADING_DAYS
    p_two_sided = erfc(abs(z) / sqrt(2.0))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1 - np.sum(residual**2) / total) if total > 0 else None
    return {
        "observations": n,
        "factor_names": list(factor_names),
        "betas": {name: float(coef) for name, coef in zip(factor_names, beta[1:])},
        "annualized_arithmetic_alpha": alpha_ann,
        "annualized_alpha_se_hac": float(se_ann),
        "annualized_alpha_ci95_low": float(alpha_ann - 1.9599639845 * se_ann),
        "annualized_alpha_ci95_high": float(alpha_ann + 1.9599639845 * se_ann),
        "z_normal_approximation": z,
        "p_two_sided_nominal": float(p_two_sided),
        "r_squared": r_squared,
        "hac_lags": lags,
        "uncertainty_note": (
            f"HAC lag-{lags} nominal normal interval, conditional on the fitted "
            "factor model; selection/revisions are not accounted for."
        ),
    }


def _aligned_arrays(
    dates: tuple[date, ...],
    masks: np.ndarray,
    net: np.ndarray,
    prices: np.ndarray,
    factors: dict[str, FactorArchive],
) -> dict[str, np.ndarray]:
    """Exact-date alignment; missing factor dates fail, never silently drop."""
    indices = np.flatnonzero(masks)
    if indices.size < 100:
        raise ValueError("Too few observations in selected chronological period")
    daily_asset = prices[1:] / prices[:-1] - 1.0
    rows: dict[str, list[dict[str, float]]] = {}
    for source in ("ff3", "ff5", "mom"):
        lookup = factors[source].dates_to_factors
        missing = [dates[idx + 1] for idx in indices if dates[idx + 1] not in lookup]
        if missing:
            raise ValueError(
                f"{source}: {len(missing)} missing dates, first={missing[0]}"
            )
        rows[source] = [lookup[dates[idx + 1]] for idx in indices]
    ff3, ff5, mom = rows["ff3"], rows["ff5"], rows["mom"]
    rf3 = np.asarray([r["RF"] for r in ff3], dtype=float)
    rf5 = np.asarray([r["RF"] for r in ff5], dtype=float)
    if not np.allclose(rf3, rf5, rtol=0, atol=1e-12):
        raise ValueError("Official FF3 and FF5 risk-free series disagree")
    return {
        "y": net[indices] - rf3,
        "spy": daily_asset[indices] - rf3,
        "market": np.asarray([r["Mkt-RF"] for r in ff3]),
        "ff3": np.asarray([[r[x] for x in ("Mkt-RF", "SMB", "HML")] for r in ff3]),
        "ff5": np.asarray(
            [[r[x] for x in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")] for r in ff5]
        ),
        "mom": np.asarray([r["Mom"] for r in mom]),
    }


def run_factor_diagnostic(
    history: PriceHistory, factors: dict[str, FactorArchive]
) -> dict:
    """Reconstruct the pre-frozen rule; do NOT optimize over candidates."""
    if set(factors) != {"ff3", "ff5", "mom"}:
        raise ValueError("Required FF3, FF5 and momentum archives")
    frozen_rule = next(x for x in CANDIDATES if x.name == FROZEN_STRATEGY)
    weights = target_weights(history.adjusted_close, frozen_rule)
    net, _ = portfolio_returns(
        history.adjusted_close,
        history.daily_cash_return,
        weights,
        cost_bps=REFERENCE_COST_BPS,
    )
    periods = {
        "training_2010_2018": period_mask(
            history.dates, TRAIN_FIRST, VALIDATION_FIRST
        ),
        "validation_2019_2022": period_mask(
            history.dates, VALIDATION_FIRST, FINAL_FIRST
        ),
        "already_seen_final_2023_2026": period_mask(
            history.dates, FINAL_FIRST, date(2100, 1, 1)
        ),
    }
    model_outputs = {}
    for period_name, mask in periods.items():
        a = _aligned_arrays(
            history.dates, mask, net, history.adjusted_close, factors
        )
        designs = {
            "SPY_market": (a["spy"].reshape(-1, 1), ("SPY_minus_RF",)),
            "FF_market": (a["market"].reshape(-1, 1), ("Mkt-RF",)),
            "FF3": (a["ff3"], ("Mkt-RF", "SMB", "HML")),
            "FF5": (a["ff5"], ("Mkt-RF", "SMB", "HML", "RMW", "CMA")),
            "FF5_plus_momentum": (
                np.column_stack((a["ff5"], a["mom"])),
                ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"),
            ),
        }
        results = {}
        for model_name in REGRESSION_MODELS:
            matrix, names = designs[model_name]
            entry = hac_regression(a["y"], matrix, names)
            entry["p_bonferroni_across_five_models"] = min(
                1.0, 5 * entry["p_two_sided_nominal"]
            )
            results[model_name] = entry
        model_outputs[period_name] = results

    return {
        "status": (
            "POST_HOC_DIAGNOSTIC_ON_ALREADY_SEEN_SPY_HOLDOUT_"
            "NOT_NEW_INDEPENDENT_ALPHA_EVIDENCE"
        ),
        "frozen_existing_research_commit": FROZEN_RESEARCH_COMMIT,
        "frozen_strategy": FROZEN_STRATEGY,
        "source": {
            "spy_adjusted_close_sha256": history.source_sha256,
            "spy_price_snapshot_matches_prior": history.source_sha256 == EXPECTED_SHA256,
            "spy_first_loaded": history.dates[0].isoformat(),
            "spy_last_loaded": history.dates[-1].isoformat(),
            "factor_official_site": (
                "https://mba.tuck.dartmouth.edu/pages/faculty/"
                "ken.french/data_library.html"
            ),
            "factors": {
                key: {
                    "archive_url": (
                        f"{FACTOR_SOURCE_ROOT}/{ARCHIVES[key]['upstream']}"
                    ),
                    "sha256": value.sha256,
                    "matches_2026_09_pinned_release": (
                        value.sha256 == ARCHIVES[key]["sha256"]
                    ),
                    "first_date": value.first_date.isoformat(),
                    "last_date": value.last_date.isoformat(),
                    "data_scale": "daily factor percent converted to decimal",
                }
                for key, value in factors.items()
            },
        },
        "method": {
            "regression_models": list(REGRESSION_MODELS),
            "standard_errors": "Newey-West HAC, Bartlett kernel, 5 daily lags",
            "alpha": "daily model intercept multiplied by 252; arithmetic",
            "rf": "official Fama-French daily RF for regression excess returns",
            "portfolio_cash": (
                "original frozen simulation used an approximate ^IRX 13-week "
                "yield; not identical to official one-month FF RF"
            ),
            "multiple_models": (
                "Bonferroni multiply two-sided p-values by five as a "
                "diagnostic only; 95% intervals are individually nominal"
            ),
            "warning": (
                "holdout and strategy are already known; significance here "
                "does not constitute an independent alpha discovery"
            ),
        },
        "regressions": model_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-csv", type=Path, default=Path("data/raw/daily_adjclose.csv")
    )
    parser.add_argument("--factors-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/factor_alpha_diagnostic.json"),
    )
    args = parser.parse_args()
    history = load_adjusted_prices(args.market_csv)
    if history.source_sha256 != EXPECTED_SHA256:
        raise ValueError("Refusing to mix a different SPY price vintage")
    factors = {
        name: load_french_archive(
            args.factors_dir / source["local"], name, verify_pinned=True
        )
        for name, source in ARCHIVES.items()
    }
    report = run_factor_diagnostic(history, factors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final = report["regressions"]["already_seen_final_2023_2026"]
    print("FROZEN strategy, POST-HOC alpha diagnostic; not independent validation")
    for name in REGRESSION_MODELS:
        entry = final[name]
        print(
            f"{name:18s}: annual alpha {entry['annualized_arithmetic_alpha']:+.2%} "
            f"(HAC 95% CI {entry['annualized_alpha_ci95_low']:+.2%}, "
            f"{entry['annualized_alpha_ci95_high']:+.2%}), "
            f"p={entry['p_two_sided_nominal']:.4g}"
        )
    print(f"Saved: {args.output}")
    print("These diagnostics cannot turn the already-inspected SPY holdout into fresh evidence.")


if __name__ == "__main__":
    main()
