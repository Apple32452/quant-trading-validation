# Preregistered follow-up: Can a frozen rule beat SPY and exhibit alpha?

**Written before examining the QQQ, IWM, or EFA strategy or benchmark outcomes.**
Status: **DESIGNED, NOT EXECUTED**. This is a future analysis, not a positive
finding. The current SPY 2023–2026 outcomes and factor diagnostic are already
known and are **not** independent evidence.

## Hypothesis and restrictions

The sole candidate is `volatility_target_20` from the previous SPY experiment:
20 previous close-to-close returns estimate realized volatility using sample
standard deviation; target 10% annualized portfolio volatility; cap ETF weight
at 100%, with the residual invested in modeled cash. A signal computed after
close *t* is executed at the close of *t+1* and can earn its first ETF return
from *t+1* to *t+2*. No shorting, borrowing, reoptimization, changed dates,
changed features, stop-loss overlays, or rescoring candidate families.

Use the **same** 5-basis-point one-way cost per change of portfolio weight,
cash proxy (^IRX discount yield / 100 / 252), exchange-holiday handling, and
daily return definitions as `market_backtest.py`. Start the rule from the
full historical warm-up, not from the test boundary. Run all analyses in a
single pass and archive the results whether they are positive or negative.

## Historical cross-asset replication (not yet run)

1. **Primary instrument:** QQQ; **secondary instruments:** IWM and EFA.
   These ETF column names were checked from the input *header only*, with no
   price paths or performance inspected for the three new ETFs. All are
   present in the existing source schema. Do not substitute instruments
   after seeing results.
2. **Source vintage:** the same archived wide-price CSV as the frozen SPY
   analysis, identified by SHA-256
   `361efb287116fc8200582c15fcd315bcce2e218fa1e50b416bacc5acebff37b9`.
   Do not silently switch to another vendor/vintage. This is a third-party
   adjusted-close mirror, not independently audited executions.
3. **Test interval:** 2023-01-01 through 2026-08-10 inclusive. Keep the full
   SPY/QQQ/IWM/EFA **intersection of valid session dates**; missing internal
   prices, duplicate dates, or unexplained gaps stop the analysis. Do not
   silently replace/drop a ticker with unfavorable data quality.
4. **Benchmarks:** simultaneous SPY buy-and-hold (requested benchmark),
   each ETF's own buy-and-hold, and the same approximate cash accrual.
   SPY and ETF benchmark returns must use the *same* included sessions;
   charge the documented one-way 5-bp initial purchase to buy-and-hold.
5. **Primary inference (QQQ only):** two **co-primary** conditions must hold
   before using the phrase *historical positive risk-adjusted outperformance*:

   - QQQ-rule daily net minus SPY buy-and-hold daily net returns has a
     strictly positive annualized arithmetic mean; the lower endpoint of
     a two-sided 97.5% **paired circular block-bootstrap** interval is
     above zero (20-session blocks; 5,000 replicates; fixed seed 20261114).
   - QQQ-rule factor-adjusted excess return has a strictly positive
     intercept in a **single prespecified** regression on the official
     Fama–French five daily factors plus daily momentum. Its two-sided
     97.5% Newey–West HAC interval (20 daily lags) must have a lower
     endpoint above zero. The coefficient is the *arithmetic* annualization
     (daily intercept × 252), not a CAGR.

   Requiring both 97.5% intervals above zero is deliberately conservative
   when two outcome families are inspected. This does **not** guarantee
   a future trading edge.
6. **Descriptive secondary results:** repeat both checks for IWM and EFA,
   reporting the full outcomes regardless of significance; apply Holm
   correction across their two-sided factor-intercept p-values. Also report
   ETF-own-buy-and-hold active return for each ticker; net CAGR, annual
   volatility, Sharpe versus cash, max drawdown, turnover, and 0/5/10/20-bp
   frozen-rule cost sensitivity. Do not designate a different ETF the
   *primary winner* based on favorable results.
7. **Audit:** verify no look-ahead (signals known before executable close),
   no regression-day dropping, correct decimal conversion for factor data,
   correct cash/dividend handling, a stable 2026-09 Kenneth French archive
   snapshot, deterministic bootstrap seed, and no market-return leakage
   into model selection.

Interpretation: the ETFs share dates, macroeconomic exposures, and (especially
QQQ) underlying equity-market risk with SPY. Cross-asset replication is useful
but **not fully independent** of the already-seen SPY backtest. These historical
checks alone do not establish robust, economically tradable alpha.

## Stronger prospective replication (not yet started)

Prespecify the same SPY rule as above before any new prospective return is
observed. **Proposed** paper-trading start: the first market session on or
after **2026-11-16**; minimum duration: 252 executed trading sessions.
Record immutable daily decisions, source quotes, approximate fill/arrival
times, any actual fees/spreads, source-vintage hashes, and portfolio holdings
at every rebalance. Compute SPY buy-and-hold on identical timestamped data
with realistic costs. If no investable execution data are available, label
the exercise a close-to-close *paper simulation*, not paper fills.

Do not change the rule during the prospective period. After the full period,
report alpha intercept (FF5 + momentum), paired SPY active return with
block-bootstrap intervals, SPY and own benchmark differences, exposure,
turnover, drawdown, total costs, and execution limitations. A **positive
prospective** result with sound data/costs would provide materially stronger
evidence; a negative result must be published too.

## Separation from the completed SPY study

The original SPY study's selected strategy, test-period returns, bootstrap,
and factor regression were inspected before this protocol. Its factor alpha
is a **post-hoc diagnostic**, not a replication test. The previously
inspected SPY observations must never be presented as another untouched
holdout. Do not iterate on the new tests after inspecting their results:
any such changes become exploratory and need another independent test.
