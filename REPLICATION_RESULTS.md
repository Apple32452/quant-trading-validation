# Completed pre-registered historical cross-asset replication

**Outcome: THE PRIMARY SUCCESS CONDITIONS WERE NOT MET.** This report
describes a one-time historical test of the frozen `volatility_target_20`
strategy on QQQ (primary) and IWM/EFA (secondary); no positive alpha claim.

## Audit trail and separation of stages

1. Original SPY research and Fama–French diagnostic were previously examined.
2. The historical cross-asset plan was publicly frozen in GitHub commit
   [`733a75a`](https://github.com/Apple32452/quant-trading-validation/commit/733a75afe72826f4163cca002ed3b12ac2c53bf9),
   `INDEPENDENT_REPLICATION_PROTOCOL.md` (SHA-256
   `9a0a8a2dd62ecc86b84c688f315f691ed0d287c3351e690276a862fbdf204018`).
   This original record deliberately retains its *not yet executed* status.
3. The exact runner, synthetic tests, and CI workflow were committed **before
   inspecting the QQQ/IWM/EFA strategy returns** in
   [`cc3919d`](https://github.com/Apple32452/quant-trading-validation/commit/cc3919d28362dc5026942d5ce1e259436f2cad6a).
   GitHub Actions passed Python 3.11, 3.12, and 3.13.
4. All four input checksums matched the previously pinned archives. The
   once-only run was `python3 cross_asset_replication.py` and completed without
   changed candidate rules or rejected samples. The machine-readable report
   is `experiments/cross_asset_replication.json`.
5. **Post-run documentation-only correction:** the generic regression tool
   generated a textual note stating "lag-5" even when invoked with
   `lags=20`. The JSON correctly stored `hac_lags: 20` and computed the
   20-lag Newey–West variance. We corrected the *text label* to "lag-20";
   **no return, coefficient, standard error, interval, p-value, or success
   decision was modified or recomputed.**

## Fixed procedure

- Price snapshot: third-party yfinance adjusted-close mirror, SHA-256
  `361efb287116fc8200582c15fcd315bcce2e218fa1e50b416bacc5acebff37b9`.
  This data source is not an audited market-execution record.
- Same 903 return-end sessions from 2023-01-03 through 2026-08-10 for SPY,
  QQQ, IWM, and EFA. No gaps, dropped model dates, ticker substitutions,
  alternative test windows, or hyperparameter searches.
- 20-day realized-volatility targeting to 10% annual risk, exposure capped
  at 1; execute one close after the signal is known; 5-bp cost per unit
  traded. Both SPY and each ETF buy-and-hold pay a 5-bp test-entry cost.
- One predeclared FF5-plus-momentum factor-adjusted alpha intercept,
  with 20-lag Newey–West uncertainty and **two-sided 97.5%** intervals.
  The dependent variable subtracts official French RF, whereas strategy
  cash approximates ^IRX/100/252.
- Paired circular 20-session block bootstrap, 5,000 repetitions,
  seed 20261114; **two-sided 97.5%** interval for annualized *arithmetic*
  daily difference versus SPY. These intervals are not CIs for CAGR.

## Outcomes

These are **modeled historical net compound annual growth rates** (CAGRs),
not deployable profit forecasts. Risk-free factor regression intercepts
and active mean returns are *arithmetic annualizations*, not CAGR.

| Frozen ETF rule | Net CAGR | Own ETF buy/hold CAGR | SPY buy/hold CAGR | Net annual mean vs SPY, 97.5% CI | FF5+Mom annual alpha, 97.5% HAC CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| **QQQ (primary)** | **18.70%** | 32.77% | 23.22% | -4.27 pp [-11.97, +3.49] | +1.05% [-3.58%, +5.68%] |
| IWM (secondary) | 9.56% | 17.70% | 23.22% | -12.29 pp [-22.48, -2.09] | -2.40% [-5.84%, +1.04%] |
| EFA (secondary) | 10.54% | 18.55% | 23.22% | -11.42 pp [-23.14, +0.32] | -2.13% [-10.40%, +6.13%] |

Both co-primary QQQ requirements were unmet:

1. The paired SPY active-return interval did **not** have a positive lower
   endpoint; its point estimate was negative.
2. The FF5-plus-momentum alpha interval also crossed zero (nominal two-sided
   p = 0.610). A positive but non-significant point estimate is not proof
   of positive alpha.

The two secondary factor-intercept p-values after predeclared Holm adjustment
were **0.235 for IWM** and **0.563 for EFA**. Neither indicates positive
alpha. IWM's annualized active-return interval was entirely negative.

| Strategy | Volatility | Maximum drawdown | Sharpe vs approximate cash |
| --- | ---: | ---: | ---: |
| QQQ rule | 10.89% | -11.49% | 1.215 |
| IWM rule | 10.94% | -15.99% | 0.478 |
| EFA rule | 10.72% | -9.53% | 0.568 |
| SPY buy and hold | 15.11% | -18.76% | 1.159 |

Volatility targeting reduced market exposure and frequently reduced
drawdown. QQQ's risk-adjusted Sharpe was somewhat higher than SPY's, but
its net CAGR and ETF-own buy-and-hold return comparison were weaker, and
the preregistered alpha test did not support a reliable positive intercept.
No strategy was reselected after these outcomes.

**Frozen-rule net CAGR across fixed one-way cost sensitivities:**

| Cost | QQQ | IWM | EFA |
| ---: | ---: | ---: | ---: |
| 0 bps | 19.02% | 9.83% | 10.87% |
| 5 bps (primary) | 18.70% | 9.56% | 10.54% |
| 10 bps | 18.38% | 9.29% | 10.21% |
| 20 bps | 17.75% | 8.76% | 9.55% |

## Conclusion and next evidence needed

**The historical cross-asset experiment does not show a profitable alpha
advantage over SPY.** Positive modeled returns do not imply outperforming
the benchmark, and factor regression estimates are compatible with zero.

Because QQQ, IWM, EFA, and SPY share economic exposures and test dates,
this historical replication is not fully independent of the original SPY
analysis. The adjusted-close mirror does not provide executable fills or
independent corporate-action/cost verification. Further development should
be explicitly exploratory; this already-seen 2023–2026 data must not be
reused as a new independent holdout after changing a rule.

The **next stronger test** is the *still-unstarted* timestamped prospective
paper simulation defined in the original protocol, run without rule changes
and evaluated after at least 252 trading sessions. No live investments or
paper fills have been made by this project.
