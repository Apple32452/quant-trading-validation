# Research Roadmap: Selection Bias and Backtest Validation

## Research question

How much does searching across multiple candidate strategies inflate
in-sample risk-adjusted performance, and how well does an untouched
chronological holdout reveal the inflation?

The current project is an educational implementation, not evidence of
profitable trading. The next experiments should distinguish synthetic
selection effects from genuine predictive performance.

## Completed baseline

- A testable annualized daily-return Sharpe implementation.
- Chronological train/holdout split and train-only rule selection.
- A deterministic demonstration using 100 independent zero-edge
  synthetic strategies.
- Finite-difference European call valuation checked against Black-Scholes.
- A paired, simplified blackjack stake-sizing simulation.

## Phase 1: Repeated synthetic overfitting study

1. Pre-specify the experiment: training length, holdout length, candidate
   counts, noise scale, and random seed list before examining results.
2. Run at least 100 independent repetitions under zero expected alpha.
3. Vary candidate counts (for example 5, 10, 25, 50, 100) while keeping
   total sample length and split fraction fixed.
4. Report the training-to-holdout Sharpe gap, distribution of holdout
   Sharpe ratios, and uncertainty intervals across repetitions.
5. Compare selection on a single noisy backtest with using multiple
   independent validation periods.
6. Include a control with a planted, known weak signal to measure power,
   false selection, and generalization.

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
