# Quantitative Simulation and Backtest Validation

Three corrected, reproducible Python exercises plus four validation studies:

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
5. **Real historical ETF experiment.** Uses third-party archived SPY adjusted
   closing prices, seven predeclared rules, chronological splits, delayed
   close execution, trading costs, and a frozen 2023–2026 holdout. The selected
   strategy earns positive historical returns but does NOT beat buy-and-hold.
6. **Post-hoc factor-adjusted alpha check.** Reconstructs the existing frozen
   strategy with unchanged trading costs; joins official Fama–French 3-factor,
   5-factor, and momentum daily series by exact date; computes regression
   intercepts with autocorrelation-robust uncertainty. No fitted model shows
   statistically reliable positive alpha. The SPY final sample was already
   examined, so this is **not fresh independent validation**.
7. **Pre-registered cross-asset replication.** Locks the QQQ primary test and
   IWM/EFA secondary tests in GitHub before reading outcomes, with 97.5%
   block-bootstrap and factor-HAC intervals, identical frozen rules and
   trading costs. **The primary outperformance/alpha conditions failed.**

## Repository contents

| Path | Purpose |
| --- | --- |
| quant_models.py | Three reproducible simulations and command-line demos |
| overfitting_study.py | Repeated-seed zero-alpha study, Bonferroni comparison, and planted-signal control |
| fetch_market_data.py | Downloads a pinned historical CSV and verifies SHA-256 outside Git |
| market_backtest.py | Historical backtesting, costs, execution timing, and walk-forward evaluation |
| fetch_factor_data.py | Downloads SHA-256-pinned official Dartmouth factor archives into ignored raw-data storage |
| factor_alpha.py | Post-hoc SPY market/CAPM, Fama–French 3/5-factor and momentum alpha diagnostics |
| cross_asset_replication.py | Fixed QQQ/IWM/EFA test with SPY and own-ETF benchmarks and predeclared uncertainty |
| prospective_paper.py | Forward-only SPY paper ledger with point-in-time input checks, NYSE calendar, audit chain, and final-only inference |
| collect_alpaca_spy.py | Read-only Alpaca SIP/raw SPY daily bar collector with required independently sourced corporate action/yield fields, hash-archived CSV, and no order API |
| tests/test_quant_models.py | Automated checks of payoffs, pricing, Sharpe, and holdout isolation |
| tests/test_overfitting_study.py | Reproducibility, statistics, CSV, and inflation/guard tests |
| tests/test_market_backtest.py | Price integrity, causal trades, cost accounting, and holdout isolation tests |
| tests/test_factor_alpha.py | Synthetic factor parsing, exact-date joining, fixed-rule, and HAC regression tests |
| tests/test_cross_asset_replication.py | Synthetic-only tests of pre-registered QQQ primary inference and benchmark costs |
| tests/test_prospective_paper.py | Synthetic-only tests of the 2026–2027 exchange clock, causal one-close delay, entry costs, and ledger integrity |
| tests/test_collect_alpaca_spy.py | Mocked-only tests for read-only Alpaca requests, time gates, explicit supplemental sources, private archives, and corruption checks |
| experiments/overfitting_summary.csv | Fixed-seed, 500-repetition summary with experimental settings and intervals |
| experiments/real_market_study.json | Frozen SPY market results, settings, cost sensitivity, and uncertainty |
| experiments/factor_alpha_diagnostic.json | Factor-adjusted SPY diagnostic, full model results, provenance, and limitations |
| experiments/cross_asset_replication.json | Once-run QQQ/IWM/EFA results, CIs, cost sensitivity, and preregistration links |
| requirements.txt | NumPy dependency |
| RESEARCH_PLAN.md | Completed stage and roadmap for leakage-safe, real-data experiments |
| INDEPENDENT_REPLICATION_PROTOCOL.md | Immutable *pre-run* QQQ/IWM/EFA and future prospective test specification |
| REPLICATION_RESULTS.md | Completed replication, failed primary claim, audit trail, caveats, and all outcomes |
| PROSPECTIVE_PAPER_RUNBOOK.md | Prepared prospective SPY paper experiment, required input feed, start gates, and operator commands |
| ALPACA_DATA_RUNBOOK.md | How to obtain paper market-data keys, supply dividends/splits/^IRX, and archive verified read-only SPY CSVs |
| .github/workflows/ci.yml | GitHub Actions checks across Python 3.11, 3.12, and 3.13 |

The research focus is **selection bias in risk-adjusted backtesting**.
Option pricing and card counting serve as separate numerical-methods
demonstrations. See [the research roadmap](RESEARCH_PLAN.md) for the
next work needed before any claim about a verified trading advantage.

## Requirements and commands

Python 3.10+ with NumPy.

~~~bash
python3 -m pip install -r requirements.txt
python3 quant_models.py all
python3 -m unittest discover -s tests -v
python3 overfitting_study.py --repetitions 500 --seed 20261112 --output experiments/overfitting_summary.csv
python3 fetch_market_data.py
python3 market_backtest.py --csv data/raw/daily_adjclose.csv --cost-bps 5
python3 fetch_factor_data.py
python3 factor_alpha.py
~~~

For individual demos:

~~~bash
python3 quant_models.py blackjack --sessions 1000 --decks 430
python3 quant_models.py black-scholes
python3 quant_models.py backtest
~~~

## Historical SPY study: frozen evaluation

**Data/provenance.** A [third-party ETF price archive](https://github.com/manisahni/marketdata)
claims its prices are yfinance dividend/split-adjusted closes; this is
neither an official exchange feed nor independently audited. The downloader
pins upstream revision 65a079329e379e104495f459e12e76a8f995585a and
checks SHA-256
361efb287116fc8200582c15fcd315bcce2e218fa1e50b416bacc5acebff37b9.
It does **not** commit or redistribute the 1.5 MB third-party CSV.

Our SPY study uses 4,427 closing prices from 2009-01-02 to 2026-08-10.
The source has one empty SPY row on **2026-05-25**, confirmed as a closed
market day by the [NYSE holiday calendar](https://www.nyse.com/trade/hours-calendars),
and a trailing empty price row; these are skipped. Two missing Treasury-bill
yields are carried forward only from previously known quotes (maximum five
sessions). Other internal missing prices cause an error. The cash-rate
proxy is the reported ^IRX 13-week discount yield divided by 100 and 252;
it approximates daily cash accrual but is **not an executable return**.

**Predeclared experimental protocol.**

1. Candidate family: long-or-cash momentum (20, 60, 120 prior days),
   long-or-cash mean reversion (3, 5, 10 prior days), and 20-day
   volatility targeting (10% annualized target, exposure capped at 1).
   No leverage, shorting, or hyperparameter tuning on final data.
2. A signal observed after close t can only be executed at close t+1,
   so its first eligible return is close t+1 to close t+2. This
   explicitly prevents same-close execution look-ahead.
3. Assumed trading cost: 5 basis points per one-way unit of portfolio
   weight changed (an illustrative sum of spreads/slippage/commissions).
   This cost has **not** been calibrated to actual execution reports.
4. Choose the sole winner by net Sharpe on **2010–2018 training only**.
   Report 2019–2022 validation separately, then test the frozen winner
   once on **2023–2026-08** against the same-ETF buy-and-hold benchmark.
5. Development-only walk-forward folds each train on the previous
   five calendar years and test in 2018, 2019, 2020, 2021, or 2022.
   Also inspect fixed-winner final costs at 0, 5, 10, and 20 bps,
   with **no strategy reselection**.

The training winner is **volatility_target_20**. All returns below are
historical modeled net returns; Sharpe subtracts the daily cash proxy.

| Metric | Volatility targeting | SPY buy-and-hold |
| --- | ---: | ---: |
| Training annualized return (2010–2018) | 8.16% | 11.64% |
| Validation annualized return (2019–2022) | 8.78% | 13.09% |
| **Final annualized return (2023–2026)** | **15.45%** | **23.23%** |
| Final Sharpe versus cash | 0.991 | 1.160 |
| Final maximum drawdown | -11.75% | -18.76% |
| Final annualized volatility | 10.51% | 15.11% |

The strategy reduced drawdown and volatility but **underperformed on
both risk-adjusted performance and absolute return**. The final 903-day
holdout's annualized *arithmetic* mean active-minus-benchmark return was
-7.11 percentage points. A paired 20-day circular block bootstrap (1,500
repetitions, seed 20261113) yielded a 95% interval of approximately
-12.45 to -2.42 points. It does **not** account for all search or data
quality uncertainty, or establish anything about future alpha.

With the chosen rule held fixed, estimated final net CAGR under 0, 5,
10, and 20 one-way bps was 15.76%, 15.45%, 15.14%, and 14.53%.
These are sensitivity diagnostics, not new candidate tests.

Reproduce the experiment:

~~~bash
python3 -m pip install -r requirements.txt
python3 fetch_market_data.py
python3 market_backtest.py \
  --csv data/raw/daily_adjclose.csv \
  --cost-bps 5 --output experiments/real_market_study.json
python3 -m unittest discover -s tests -v
~~~

The final SPY period has now been inspected. **It cannot be reused as
an independent final test for revised strategies or settings.** Further
research needs new prospective data or a separately held-out instrument.

## SPY factor-adjusted alpha: already-seen data diagnostic

The previously frozen `volatility_target_20` rule and original 5-bp cost
assumption remain unchanged. Its daily net returns are regressed on the
official Kenneth French daily risk-free return and publicly documented
factor portfolios. The download pins the exact 2026-09 CRSP-released
factor archives from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The archived ZIPs remain in `data/raw/`, outside version control. The
script verifies all three checksums, converts factor **percent** to decimal,
requires an exact session-date join, and refuses an SPY source-vintage change.

Each intercept is daily regression alpha multiplied by 252 trading days;
this is an *annualized arithmetic* rate, **not** compounded CAGR. Two-sided
nominal 95% normal-approximation intervals use Newey–West/Bartlett HAC
standard errors with five lags. The JSON also reports betas, R², five-model
Bonferroni diagnostic p-values, and source hashes. The dependent variable
subtracts the official French one-month bill proxy, while the frozen
portfolio's cash treatment uses an approximate ^IRX 13-week discount yield;
these cash proxies are not identical.

**Existing final period: 903 sessions, 2023–2026-08; post-hoc, not a new test.**

| Model | Annualized alpha | HAC nominal 95% interval | Two-sided nominal p |
| --- | ---: | ---: | ---: |
| SPY excess-return benchmark | -1.10% | [-4.53%, +2.33%] | 0.530 |
| French market factor | -1.19% | [-4.50%, +2.11%] | 0.479 |
| French 3-factor | -1.41% | [-4.84%, +2.01%] | 0.418 |
| French 5-factor | -1.86% | [-5.10%, +1.38%] | 0.260 |
| French 5-factor + momentum | -1.89% | [-5.14%, +1.36%] | 0.255 |

Every interval includes zero; none demonstrates positive alpha. The
strategy's SPY-excess-return beta was approximately 0.65, consistent with
its lower market exposure under volatility targeting. Regression alpha is
model-dependent; no causal edge, trading profit, or prospective
generalization has been demonstrated. Five-model multiple testing and
the previously seen holdout only strengthen this caution.

Reproduce without committing upstream raw factor archives:

~~~bash
python3 -m pip install -r requirements.txt
python3 fetch_market_data.py
python3 fetch_factor_data.py
python3 factor_alpha.py --output experiments/factor_alpha_diagnostic.json
python3 -m unittest discover -s tests -v
~~~

The [cross-asset and prospective protocol](INDEPENDENT_REPLICATION_PROTOCOL.md)
was published before reading any QQQ/IWM/EFA performance. The **cross-asset
historical test has now been completed**, with an unfavorable primary result;
see [the replication report](REPLICATION_RESULTS.md). The future prospective
test remains unstarted. Correlation with SPY makes this historical attempt
less independent than truly new, prospective observations.

## Frozen cross-asset replication: completed, primary claim rejected

**Audit:** Strategy, source vintage, symbols, dates, costs, factors,
bootstrap interval and seed, and QQQ co-primary success criteria were fixed
in the [pre-run protocol](INDEPENDENT_REPLICATION_PROTOCOL.md), GitHub
commit `733a75a`. The implementation and synthetic tests were committed
as `cc3919d` before the single historical test was executed. The result
uses 903 aligned sessions ending 2023-01-03 through 2026-08-10, with
5-bp trading costs and one-way test-entry charges for buy-and-hold.

| Frozen rule | Net CAGR | Own ETF buy/hold CAGR | SPY buy/hold CAGR | Net annual mean vs SPY, 97.5% interval | FF5+Mom annual alpha, 97.5% HAC interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| **QQQ (primary)** | **18.70%** | 32.77% | 23.22% | -4.27 pp [-11.97, +3.49] | +1.05% [-3.58%, +5.68%] |
| IWM (secondary) | 9.56% | 17.70% | 23.22% | -12.29 pp [-22.48, -2.09] | -2.40% [-5.84%, +1.04%] |
| EFA (secondary) | 10.54% | 18.55% | 23.22% | -11.42 pp [-23.14, +0.32] | -2.13% [-10.40%, +6.13%] |

The QQQ strategy underperformed SPY in modeled net CAGR and its paired
annualized *arithmetic* active-return interval crossed zero. Its positive
factor-alpha **point estimate is not statistically significant**; the
confidence interval also crosses zero (nominal two-sided p=0.610).
Therefore **both pre-registered primary success requirements failed**.
IWM/EFA also failed to show positive factor-adjusted alpha. All three
rules lagged their *own* ETF buy-and-hold benchmark in net CAGR.

Volatility targeting limited market exposure and generally reduced
drawdown; a smaller drawdown is not a proof of a tradable alpha signal.
QQQ's realized Sharpe versus approximate cash was 1.215 versus SPY's
1.159, but this did not override the pre-specified criteria. See
[REPLICATION_RESULTS.md](REPLICATION_RESULTS.md) for the complete
uncertainty estimates, cost sensitivities, and provenance. The existing
pre-run protocol is intentionally preserved unchanged as an audit record.

To reproduce the **already-inspected** output (not claim another independent
test), with locally downloaded and SHA-256-pinned inputs:

~~~bash
python3 -m pip install -r requirements.txt
python3 fetch_market_data.py
python3 fetch_factor_data.py
python3 cross_asset_replication.py --output experiments/cross_asset_replication.json
python3 -m unittest discover -s tests -v
~~~

This is a third-party, adjusted-close historical simulation. It does not
audit fill prices, financing, execution impact, corporate actions, or
capacity. Correlated historical ETFs cannot replace the *prospective*
test proposed in the protocol, which has **not started**.

## Prospective forward-only SPY paper simulation: code prepared

The next research stage now has an [operator runbook](PROSPECTIVE_PAPER_RUNBOOK.md)
and a tested `prospective_paper.py` implementation. It uses a frozen
20-day volatility-targeting rule, observed unadjusted closes with explicit
dividends/splits, approximate observed cash rates, and an append-only,
SHA-256-chained journal. NYSE 2026–2027 session/early-close checks prevent
silent backfills, and all returns are withheld from the status view until
the fixed endpoint of **252 realized prospective return intervals**.

**Prepared is not started:** the code will refuse initialization until
after the **2026-11-13** market close and before **2026-11-16** open. A
verified warmup CSV and daily point-in-time feed are still required;
none has been supplied, no ledger has been initialized, and **no real or
paper orders have been placed**. The manually maintained journal is kept
outside Git, in `data/prospective/`, and needs durable private backups.
The [read-only Alpaca collector](ALPACA_DATA_RUNBOOK.md) can download
unadjusted consolidated SIP daily stock bars and archive them with raw
responses, source references and SHA-256 checksums. It **requires explicit,
independently checked dividend, split, and ^IRX inputs** and never sends
orders or initializes the ledger automatically. No provider credentials or
actual snapshots have been supplied to this repository.
The official factor files currently pinned in this project stop before
the prospective period. Final factor-alpha testing will require the first
official release covering every later date, archived with its checksum.

This is a deliberately constrained *paper simulation* with assumed close
execution/costs, not a brokerage account or automatic trading bot.

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
- **The original backtest and repeated Monte Carlo study use synthetic
  returns.** The SPY experiment uses historical prices from an external
  unverified mirror with assumed, not actual, fills/costs. The close-only
  data cannot validate intraday liquidity, slippage, impact, capacity, or
  financing, and the modeled T-bill cash proxy is approximate.
- **The final historical test is no longer untouched.** Future strategy
  changes must use fresh independent evidence; repeatedly improving a
  backtest on the same final interval would invalidate the independence.
  Bonferroni cannot repair hidden search, leakage, or post-hoc changes.

These studies establish no real-money profit, verified market alpha, or
demonstrated advantage over SPY buy-and-hold.
