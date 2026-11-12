"""Repeated-seed study of backtest selection bias on synthetic daily returns.

All candidate strategies are independent, IID Gaussian return streams, not
tradeable strategies or historical market observations. We select using the
training period only. The out-of-sample period is observed only after selection.

Under the global null, every candidate has zero expected excess return.
The planted-signal control adds a known positive mean to candidate 0, without
changing the independent Gaussian innovations or evaluating on the holdout.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from math import erfc, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np


ANNUAL_TRADING_DAYS = 252
NORMAL_95 = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class StudyConfig:
    repetitions: int = 500
    seed: int = 20261112
    training_days: int = 756
    holdout_days: int = 504
    candidate_counts: tuple[int, ...] = (1, 5, 10, 25, 50, 100)
    daily_volatility: float = 0.01
    planted_daily_alpha: float = 0.0012
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        if self.repetitions < 2:
            raise ValueError("At least two repetitions are needed for intervals")
        if min(self.training_days, self.holdout_days) < 30:
            raise ValueError("Each period must contain at least 30 observations")
        if not self.candidate_counts or any(c < 1 for c in self.candidate_counts):
            raise ValueError("Candidate counts must contain positive integers")
        if len(set(self.candidate_counts)) != len(self.candidate_counts):
            raise ValueError("Candidate counts must be unique")
        if self.daily_volatility <= 0 or not np.isfinite(self.daily_volatility):
            raise ValueError("Daily volatility must be positive and finite")
        if self.planted_daily_alpha <= 0 or not np.isfinite(self.planted_daily_alpha):
            raise ValueError("Planted mean must be positive and finite")
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must lie between zero and one")


@dataclass(frozen=True)
class StudyRow:
    candidates: int
    repetitions: int
    seed: int
    training_days: int
    holdout_days: int
    daily_volatility: float
    planted_daily_alpha: float
    significance_level: float
    null_mean_best_training_sharpe: float
    null_mean_selected_holdout_sharpe: float
    null_holdout_ci95_low: float
    null_holdout_ci95_high: float
    null_mean_optimism_gap: float
    null_gap_ci95_low: float
    null_gap_ci95_high: float
    null_naive_false_positive_rate: float
    null_bonferroni_false_positive_rate: float
    null_bonferroni_fpr_ci95_low: float
    null_bonferroni_fpr_ci95_high: float
    signal_correct_detection_rate: float
    signal_false_selection_rate: float
    signal_mean_selected_holdout_sharpe: float


def _sharpe_by_candidate(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return annualized sample Sharpe and sample standard deviations by column."""
    deviations = np.std(returns, axis=0, ddof=1)
    if np.any(deviations <= np.finfo(float).eps):
        raise ValueError("All simulated candidates need nonzero volatility")
    return (
        np.mean(returns, axis=0) / deviations * sqrt(ANNUAL_TRADING_DAYS),
        deviations,
    )


def one_sided_normal_tail_p(training_sharpe: float, training_days: int) -> float:
    """Approximate one-sided H0: mean <= 0 test, using a normal t-stat tail.

    This is a large-sample NORMAL approximation, not an exact Student-t test.
    The simulations use IID normal returns so approximation error is small at
    the default 756-day training horizon.
    """
    if training_days < 2 or not np.isfinite(training_sharpe):
        raise ValueError("Invalid Sharpe or training sample size")
    test_stat = training_sharpe * sqrt(training_days / ANNUAL_TRADING_DAYS)
    return 0.5 * erfc(test_stat / sqrt(2.0))


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    half_width = NORMAL_95 * float(np.std(arr, ddof=1)) / sqrt(arr.size)
    return mean, mean - half_width, mean + half_width


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a Bernoulli event rate."""
    p = successes / trials
    denominator = 1.0 + NORMAL_95**2 / trials
    center = (p + NORMAL_95**2 / (2 * trials)) / denominator
    half_width = (
        NORMAL_95
        * sqrt(p * (1.0 - p) / trials + NORMAL_95**2 / (4 * trials**2))
        / denominator
    )
    return center - half_width, center + half_width


def run_study(config: StudyConfig) -> list[StudyRow]:
    """Measure selection bias, global-null false positives, and signal detection.

    Candidate sets are nested within a repetition, and the same underlying
    Gaussian draws supply both the null and planted-signal experiments. This
    gives paired comparisons across candidate counts and experimental arms.
    """
    rng = np.random.default_rng(config.seed)
    counts = tuple(sorted(config.candidate_counts))
    maximum = max(counts)
    annual_factor = sqrt(ANNUAL_TRADING_DAYS)
    per_count: dict[int, dict[str, list[float]]] = {
        n: {
            "null_train": [],
            "null_holdout": [],
            "null_gap": [],
            "naive_selected": [],
            "bonferroni_selected": [],
            "signal_correct": [],
            "signal_false": [],
            "signal_holdout": [],
        }
        for n in counts
    }

    for _ in range(config.repetitions):
        # IID Gaussian synthetic "daily strategy returns"; chronological split.
        samples = rng.normal(
            0.0,
            config.daily_volatility,
            size=(config.training_days + config.holdout_days, maximum),
        )
        training = samples[: config.training_days]
        untouched_holdout = samples[config.training_days :]
        null_train_sharpes, train_stds = _sharpe_by_candidate(training)
        null_holdout_sharpes, holdout_stds = _sharpe_by_candidate(untouched_holdout)

        signal_train_sharpes = null_train_sharpes.copy()
        signal_holdout_sharpes = null_holdout_sharpes.copy()
        signal_train_sharpes[0] += (
            config.planted_daily_alpha / train_stds[0] * annual_factor
        )
        signal_holdout_sharpes[0] += (
            config.planted_daily_alpha / holdout_stds[0] * annual_factor
        )

        for n in counts:
            observations = per_count[n]
            # Crucial: neither winner index uses the holdout scores.
            null_winner = int(np.argmax(null_train_sharpes[:n]))
            train_score = float(null_train_sharpes[null_winner])
            holdout_score = float(null_holdout_sharpes[null_winner])
            null_p = one_sided_normal_tail_p(train_score, config.training_days)
            observations["null_train"].append(train_score)
            observations["null_holdout"].append(holdout_score)
            observations["null_gap"].append(train_score - holdout_score)
            observations["naive_selected"].append(
                float(null_p < config.significance_level)
            )
            observations["bonferroni_selected"].append(
                float(null_p < config.significance_level / n)
            )

            signal_winner = int(np.argmax(signal_train_sharpes[:n]))
            signal_p = one_sided_normal_tail_p(
                float(signal_train_sharpes[signal_winner]), config.training_days
            )
            passes_guard = signal_p < config.significance_level / n
            observations["signal_correct"].append(
                float(passes_guard and signal_winner == 0)
            )
            observations["signal_false"].append(
                float(passes_guard and signal_winner != 0)
            )
            # OOS score is recorded even if the training-only guard rejects it.
            observations["signal_holdout"].append(
                float(signal_holdout_sharpes[signal_winner])
            )

    rows = []
    for n in counts:
        observations = per_count[n]
        _, holdout_low, holdout_high = _mean_ci(observations["null_holdout"])
        gap_mean, gap_low, gap_high = _mean_ci(observations["null_gap"])
        false_positives = int(sum(observations["bonferroni_selected"]))
        fpr_low, fpr_high = _wilson_interval(false_positives, config.repetitions)
        rows.append(
            StudyRow(
                candidates=n,
                repetitions=config.repetitions,
                seed=config.seed,
                training_days=config.training_days,
                holdout_days=config.holdout_days,
                daily_volatility=config.daily_volatility,
                planted_daily_alpha=config.planted_daily_alpha,
                significance_level=config.significance_level,
                null_mean_best_training_sharpe=float(
                    np.mean(observations["null_train"])
                ),
                null_mean_selected_holdout_sharpe=float(
                    np.mean(observations["null_holdout"])
                ),
                null_holdout_ci95_low=holdout_low,
                null_holdout_ci95_high=holdout_high,
                null_mean_optimism_gap=gap_mean,
                null_gap_ci95_low=gap_low,
                null_gap_ci95_high=gap_high,
                null_naive_false_positive_rate=float(
                    np.mean(observations["naive_selected"])
                ),
                null_bonferroni_false_positive_rate=false_positives / config.repetitions,
                null_bonferroni_fpr_ci95_low=fpr_low,
                null_bonferroni_fpr_ci95_high=fpr_high,
                signal_correct_detection_rate=float(
                    np.mean(observations["signal_correct"])
                ),
                signal_false_selection_rate=float(
                    np.mean(observations["signal_false"])
                ),
                signal_mean_selected_holdout_sharpe=float(
                    np.mean(observations["signal_holdout"])
                ),
            )
        )
    return rows


def write_csv(rows: list[StudyRow], destination: Path) -> None:
    """Write all predeclared metrics and simulation parameters for reproduction."""
    if not rows:
        raise ValueError("No study results to write")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(StudyRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--training-days", type=int, default=756)
    parser.add_argument("--holdout-days", type=int, default=504)
    parser.add_argument("--candidate-counts", default="1,5,10,25,50,100")
    parser.add_argument("--daily-volatility", type=float, default=0.01)
    parser.add_argument("--planted-daily-alpha", type=float, default=0.0012)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = StudyConfig(
        repetitions=args.repetitions,
        seed=args.seed,
        training_days=args.training_days,
        holdout_days=args.holdout_days,
        candidate_counts=tuple(int(c.strip()) for c in args.candidate_counts.split(",")),
        daily_volatility=args.daily_volatility,
        planted_daily_alpha=args.planted_daily_alpha,
        significance_level=args.significance_level,
    )
    rows = run_study(config)
    print(
        "SYNTHETIC IID Gaussian returns, zero-edge null, paired planted signal;"
        f" repetitions={config.repetitions}, seed={config.seed}"
    )
    print(
        "Candidates | null train SR | null OOS SR | gap (95% CI)         "
        "| naive FPR | Bonf FPR | signal detection"
    )
    for row in rows:
        print(
            f"{row.candidates:10d} | {row.null_mean_best_training_sharpe:13.3f} "
            f"| {row.null_mean_selected_holdout_sharpe:11.3f} "
            f"| {row.null_mean_optimism_gap:5.3f} "
            f"({row.null_gap_ci95_low:.3f}, {row.null_gap_ci95_high:.3f}) "
            f"| {row.null_naive_false_positive_rate:9.1%} "
            f"| {row.null_bonferroni_false_positive_rate:8.1%} "
            f"| {row.signal_correct_detection_rate:15.1%}"
        )
    if args.output is not None:
        write_csv(rows, args.output)
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
