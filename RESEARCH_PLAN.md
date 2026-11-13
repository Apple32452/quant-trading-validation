# Research Roadmap: Selection Bias and Backtest Validation

## Research question

How much does searching across multiple candidate strategies inflate
in-sample risk-adjusted performance, and how well does an untouched
chronological holdout reveal the inflation?

The current project is an educational implementation, not evidence of a
superior trading strategy. It now includes synthetic multiple-comparison
experiments and a frozen historical SPY study. The latter has positive
modeled returns but underperformed buy-and-hold in the final holdout.
The follow-up Fama–French factor regressions also show no statistically
reliable positive alpha; these regressions are **post-hoc diagnostics**
on an already-seen SPY final interval.
The **pre-registered QQQ/IWM/EFA historical replication is now complete**:
the QQQ primary benchmark-outperformance and positive-alpha conditions
both failed; all three frozen rules underperformed SPY in modeled net CAGR.

## Completed baseline

- A testable annualized daily-return Sharpe implementation.
- Chronological train/holdout split and train-only rule selection.
- A deterministic demonstration using 100 independent zero-edge
  synthetic strategies.
- Finite-difference European call valuation checked against Black-Scholes.
- A paired, simplified blackjack stake-sizing simulation.
- A 500-replication zero-alpha experiment with nested candidate counts
  (1, 5, 10, 25, 50, 100), fixed 756/504-day split, Monte Carlo intervals,
  and a deterministic results CSV.
- An unadjusted versus Bonferroni-adjusted training-only one-sided
  normal-approximation comparison and a positive-signal detection control.
- Added automated tests and CI smoke coverage for the new experiment.
- A checksum-pinned ETF adjusted-close data download, strict date validation,
  holiday/yield-gap checks, causal next-close execution, modeled trading costs,
  and a 13-week T-bill cash proxy.
- Seven predeclared SPY strategy candidates, training-only selection on
  2010–2018, validation on 2019–2022, and a single final 2023–2026
  holdout with historical benchmarks and bootstrap uncertainty.
- Development-only trailing five-year walk-forward folds for 2018–2022;
  fixed-winner cost stress tests without strategy reselection.
- Official 2026-09 Kenneth French daily 3-factor, 5-factor, and momentum
  ZIP files with SHA-256 pins, strict exact-date joins, and percent-to-decimal
  conversion, without committing third-party datasets.
- Frozen-strategy regressions against SPY excess return and Fama–French
  market / FF3 / FF5 / FF5 + momentum, with Newey–West HAC uncertainty,
  model-specific betas, and no post-hoc strategy reselection.
- A prespecified QQQ/IWM/EFA and future prospective replication design
  in INDEPENDENT_REPLICATION_PROTOCOL.md, committed while neither
  experiment had been executed or examined.
- A one-time fixed-rule QQQ/IWM/EFA replication, with identical source hash,
  903 matched sessions, own-ETF and SPY benchmarks, 5-bp costs, 97.5%
  paired circular block-bootstrap intervals, FF5+Mom HAC(20) alpha intervals,
  secondary Holm adjustment, and 0/5/10/20-bp cost sensitivity.
- The full **failed primary** result, separate from the unmodified pre-run
  protocol, in REPLICATION_RESULTS.md and
  experiments/cross_asset_replication.json.

## Phase 1: Repeated synthetic overfitting study (core complete)

Completed: fixed configuration (training/holdout lengths, daily volatility,
candidate counts, seed), 500 independent repetitions under the global null,
Monte Carlo optimism/holdout intervals, unadjusted versus Bonferroni false
positive rates, and a positive-signal control with known ground truth.
Full numeric evidence and reproducibility command are in README.md and
experiments/overfitting_summary.csv.

Still planned:

1. Evaluate two or more independent chronological validation segments
   versus a single training backtest, without using final holdout outcomes
   for method selection.
2. Add heavier-tailed and autocorrelated null models, where a Gaussian
   normal approximation may be unreliable.
3. Compare a predeclared held-out threshold or nonparametric correction
   with Bonferroni; record detection power and false positive rates.

## Phase 2: A leakage-safe evaluation framework (prototype complete)

Completed for the first historical study: time-ordered input checks,
next-close execution lag, trading-cost and turnover calculations, a cash
yield approximation, walk-forward development, frozen final candidate,
provenance and reproducible outputs.

Still needed: real quote/fill cost calibration, data-license and corporate
action audit, capacity and liquidity models, more realistic financing and
market-impact assumptions, and stronger controls for missing data.

## Phase 3: Real historical market data (single-ETF first pass complete)

Initial data: a third-party mirror of yfinance adjusted-close history,
not an independently audited primary market feed. We did not redistribute
it. The SPY study records its exact snapshot hash, costs, cash yield
approximations, split boundaries, benchmark, Sharpe, return, drawdown,
turnover, bootstrap intervals, and the negative benchmark finding.

**Crucial next step:** the 2023–2026 SPY holdout has already been seen.
Do NOT retune new rules to improve that same final sample. A new frozen
prospective dataset, separately predesignated market/instrument, or
precommitted time-based replication is necessary for independent evidence.
Market impact, point-in-time corporate actions, and execution calibration
also remain necessary before any deployment consideration.

## Phase 3B: Factor-adjusted alpha check (post-hoc diagnostic complete)

The already-selected `volatility_target_20` rule was reconstructed with
its original delayed fills and 5-bp trading cost. Daily excess returns
were regressed against the SPY excess-return benchmark and official
French market, 3-factor, 5-factor, and 5-factor + momentum series.
In 903 final SPY sessions, the annualized arithmetic intercepts ranged
from **-1.10% to -1.89%**; every nominal 95% HAC interval included zero.
The result is not evidence of positive factor-adjusted alpha.

The precommitted design for an additional historical cross-asset attempt
and stronger prospective test is preserved in
[INDEPENDENT_REPLICATION_PROTOCOL.md](INDEPENDENT_REPLICATION_PROTOCOL.md).
The historical replication was **executed once** without retuning and did
not meet its QQQ primary criteria. Read the completed audit and negative
result in [REPLICATION_RESULTS.md](REPLICATION_RESULTS.md).

The prospective 252-session test has **not begun**. Future strategy
development is exploratory, requires a separate fresh test for any claimed
improvement, and must not reuse inspected 2023–2026 observations as an
independent holdout.

## Phase 4: Additional mathematical checks

- Black-Scholes: grid and timestep convergence across strikes, expiries,
  rates, and volatilities; Greeks versus analytical derivatives.
- Blackjack: bankroll drawdown and ruin sensitivity under alternative
  toy win-probability assumptions; separate bets from actual blackjack
  round dynamics if a casino-faithful simulator is desired.

## Reporting standard

Every result should identify whether it is from synthetic or real data.
Do not present in-sample improvements as out-of-sample alpha, and do not
claim profitability without costs, risk, and independent validation.
