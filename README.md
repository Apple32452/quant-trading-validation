# Quantitative Simulation and Backtest Validation

Three corrected, reproducible Python exercises:

1. **Blackjack Monte Carlo.** Simulates Hi-Lo true counts every five cards
   and compares adaptive ($50/$500) with fixed ($50) bets on the same outcomes.
   The corrected per-hand payoff is **outcome x bet**, not outcome plus bet.
2. **Black-Scholes PDE.** Evolves a European call payoff across a stock-price
   grid using explicit finite differences. Uses boundary values and a
   conservative stability check, then compares to the analytical call price.
3. **Backtest overfitting.** Annualizes the daily-return Sharpe ratio,
   selects candidate rules using a chronological training period only,
   and evaluates the chosen rule once on held-out returns. A separate
   synthetic-noise example illustrates in-sample selection bias.

## Repository contents

| Path | Purpose |
| --- | --- |
| quant_models.py | Three reproducible simulations and command-line demos |
| tests/test_quant_models.py | Automated checks of payoffs, pricing, Sharpe, and holdout isolation |
| requirements.txt | NumPy dependency |
| RESEARCH_PLAN.md | Roadmap for repeated-seed, leakage-safe, and real-data experiments |
| .github/workflows/ci.yml | GitHub Actions checks across Python 3.11, 3.12, and 3.13 |

The near-term research focus is **selection bias in risk-adjusted
backtesting**, with the option-pricing and card-counting examples serving
as separate numerical-methods demonstrations. See [the research roadmap](RESEARCH_PLAN.md)
for experiments that have not yet been conducted.

## Requirements and commands

Python 3.10+ with NumPy.

~~~bash
python3 -m pip install -r requirements.txt
python3 quant_models.py all
python3 -m unittest discover -s tests -v
~~~

For individual demos:

~~~bash
python3 quant_models.py blackjack --sessions 1000 --decks 430
python3 quant_models.py black-scholes
python3 quant_models.py backtest
~~~

## Scope and limitations

- **Blackjack is a toy simulation.** Its illustrative win probability is
  P(win) = clip(0.5 + (TC - 1) / 400, 0, 1). This is not estimated from
  actual blackjack rules and omits pushes, different payouts, basic
  strategy, doubling, splitting, and shoe penetration. It does not
  establish an advantage over the house.
- **Black-Scholes uses an idealized pricing model**, assuming a European
  call with no dividends and constant volatility and interest rates.
  Numerical agreement with the analytical formula checks the solver,
  not the empirical accuracy of this financial model.
- **Backtesting uses synthetic returns.** A rule must not use future
  information and should output net strategy returns accounting for
  fees, slippage, and financing when used with real prices. Repeatedly
  tuning against the same holdout invalidates a one-time test. For
  real research, use realistic market data, execution assumptions,
  walk-forward testing, and fresh final validation periods.

These examples make no claim about a real-money profit, verified market
alpha, or a demonstrated trading advantage.
