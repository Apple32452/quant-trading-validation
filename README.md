# Quantitative Simulation and Backtest Validation

Three corrected, reproducible Python exercises, plus a repeated-simulation
research study:

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
4. **Repeated multiple-testing experiment.** Runs 500 independent synthetic
   repetitions with nested candidate counts, estimates the training/holdout
   Sharpe gap and confidence intervals, compares an unadjusted one-sided
   training significance test with a Bonferroni guard, and includes a
   planted-positive-signal control.

## Repository contents

| Path | Purpose |
| --- | --- |
| quant_models.py | Three reproducible simulations and command-line demos |
| overfitting_study.py | Repeated-seed zero-alpha study, Bonferroni comparison, and planted-signal control |
| tests/test_quant_models.py | Automated checks of payoffs, pricing, Sharpe, and holdout isolation |
| tests/test_overfitting_study.py | Reproducibility, statistics, CSV, and inflation/guard tests |
| experiments/overfitting_summary.csv | Fixed-seed, 500-repetition summary with experimental settings and intervals |
| requirements.txt | NumPy dependency |
| RESEARCH_PLAN.md | Completed stage and roadmap for leakage-safe, real-data experiments |
| .github/workflows/ci.yml | GitHub Actions checks across Python 3.11, 3.12, and 3.13 |

The research focus is **selection bias in risk-adjusted backtesting**.
Option pricing and card counting serve as separate numerical-methods
demonstrations. See [the research roadmap](RESEARCH_PLAN.md) for the
next work needed before any claim about real financial returns.

## Requirements and commands

Python 3.10+ with NumPy.

~~~bash
python3 -m pip install -r requirements.txt
python3 quant_models.py all
python3 -m unittest discover -s tests -v
python3 overfitting_study.py --repetitions 500 --seed 20261112 --output experiments/overfitting_summary.csv
~~~

For individual demos:

~~~bash
python3 quant_models.py blackjack --sessions 1000 --decks 430
python3 quant_models.py black-scholes
python3 quant_models.py backtest
~~~

## Repeated synthetic overfitting study

**Setup (fixed before the reported run):** 500 independent Monte Carlo
replications; 756 training days and 504 untouched holdout days; 252 trading
days/year; independent IID Gaussian candidate returns with daily mean 0 and
daily standard deviation 1%. The number of candidates is nested within each
replication. The training period selects the single highest-Sharpe candidate.
We record its held-out Sharpe **after** selection.

A one-sided large-sample normal-tail test asks whether that winning candidate
has a positive training mean. The naive rule uses a 5% significance threshold.
Bonferroni uses 5% divided by the number of candidates. Both are computed
entirely from training data: neither chooses candidates by looking at holdout
performance. The p-values are **normal approximations**, not exact Student-t
tests. This experiment evaluates them under IID Gaussian simulation, not
under realistic serial correlation or volatility clustering.

| Candidates | Mean best training Sharpe | Mean selected holdout Sharpe | Mean optimism gap (95% CI) | Naive null false-positive rate | Bonferroni null false-positive rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.043 | -0.023 | -0.020 (-0.097, 0.057) | 3.8% | 3.8% |
| 5 | 0.638 | 0.001 | 0.637 (0.565, 0.709) | 20.2% | 4.2% |
| 10 | 0.897 | 0.009 | 0.887 (0.815, 0.960) | 44.4% | 4.4% |
| 25 | 1.127 | 0.046 | 1.081 (1.011, 1.150) | 72.0% | 3.8% |
| 50 | 1.282 | 0.037 | 1.245 (1.177, 1.313) | 93.2% | 4.2% |
| 100 | 1.443 | -0.054 | 1.497 (1.429, 1.566) | 99.2% | 4.6% |

The gap confidence intervals describe the Monte Carlo mean training-minus-
holdout Sharpe difference; they are not profitability forecasts. CSV output
also includes 95% Wilson confidence intervals for the corrected null
false-positive rates.

**Planted-signal control:** candidate 0 is given a known +0.12% expected
daily return, with all other candidates still zero-edge. With the Bonferroni
guard and training-only selection, the correct-signal detection rate
decreases from 94.0% with one candidate to 49.2% with 100 candidates.
This shows the **cost of stricter false-discovery control**: a safeguard can
reject some genuine signals. The control is artificially constructed and
does not demonstrate a tradeable forecasting signal.

To rerun the committed experiment exactly:

~~~bash
python3 overfitting_study.py \
  --repetitions 500 --seed 20261112 \
  --training-days 756 --holdout-days 504 \
  --candidate-counts 1,5,10,25,50,100 \
  --daily-volatility 0.01 --planted-daily-alpha 0.0012 \
  --significance-level 0.05 \
  --output experiments/overfitting_summary.csv
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
  walk-forward testing, and fresh final validation periods. Bonferroni
  addresses a predeclared set of tests; it cannot repair hidden strategy
  searches, data leakage, or multiple rounds of post-hoc tuning.

These examples make no claim about a real-money profit, verified market
alpha, or a demonstrated trading advantage.
