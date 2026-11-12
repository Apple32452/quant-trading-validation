# Research Roadmap: Selection Bias and Backtest Validation

## Research question

How much does searching across multiple candidate strategies inflate
in-sample risk-adjusted performance, and how well does an untouched
chronological holdout reveal the inflation?

The current project is an educational implementation, not evidence of
profitable trading. We have now measured synthetic multiple-comparison
effects, but not any actual market predictive performance.

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

## Phase 2: A leakage-safe evaluation framework

- Accept timestamped return series and enforce chronological order.
- Ensure each signal uses only information available before the trade.
- Add turnover, fees, slippage, shorting restrictions, and risk-free
  financing where relevant.
- Use rolling or expanding walk-forward windows for development.
- Freeze the final strategy and hyperparameters before a single
  untouched final-period evaluation.
- Record data provenance, timestamps, seeds, parameters, and code version.

## Phase 3: Real historical market data

Use properly licensed/public historical data or user-supplied CSVs.
Validate splits, adjusted prices, missingness, and corporate actions.
Compare against buy-and-hold and a no-predictability baseline. Report
drawdown, volatility, Sharpe, turnover, and transaction-cost sensitivity.
Do not select a reported result by repeatedly inspecting the final holdout.

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
