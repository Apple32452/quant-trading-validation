# Prospective SPY paper simulation: prepared, NOT started

**Current state (2026-11-13 after the close): CODE READY, NOT STARTED.**
The private warmup preparation has 21 consecutive SPY closing prices and
21 explicit dividend/split entries, but only **20 of the required 21 ^IRX
cash-yield observations**. Friday's **2026-11-13 final ^IRX close** has not
been independently verified. No prospective session has been recorded, no
ledger initialized, no orders placed, and no profit claimed. The partially
filled warmup CSV and spreadsheet remain **outside this public repository**.
This separate, prospective stage implements the predeclared continuation of
the frozen `volatility_target_20` strategy. It does **not** re-analyze the
already-seen 2023–2026 SPY or QQQ/IWM/EFA outcomes to choose a new rule.

## Locked scope

- **Instrument:** SPY only; paper-only long/cash portfolio vs SPY buy-and-hold.
- **Start:** close of Monday **2026-11-16**, subject to the official NYSE
  schedule. The rule's first SPY interval is 2026-11-16 close to the next
  close. Only then can any modeled return be realized.
- **Prerequisite:** at least 21 **consecutive NYSE session closes through
  Friday 2026-11-13**, with a point-in-time price/dividend/split/cash-yield
  series, committed **before 2026-11-16 9:30 a.m. Eastern**. Without the
  warmup and a named provider, the simulation is NOT running.
- **Candidate:** the already-frozen 20-day, 10%-annual-volatility long/cash
  rule; no leverage, borrowing, shorts, stop-losses, or retuning.
- **Execution:** signal visible after the close of day *t* is executed at the
  close of the next session *t+1*, with return credited no earlier than the
  *t+1* to *t+2* interval. This uses modeled closing-price execution,
  **not real exchange order fills**.
- **Costs:** 5 bps for each unit of portfolio weight changed; a 5-bp
  test-entry purchase cost for SPY buy-and-hold; approximate daily cash
  credit of the prior session's annual ^IRX quote divided by 100 and 252.
- **End:** exactly **252 realized close-to-close intervals** after initial
  entry, requiring 253 observed prospective closing sessions. No mid-test
  performance analysis, candidate switching, early success declarations,
  endpoint extension, or repeated-final testing.
- **Primary decision:** two-sided **97.5%** interval strictly above zero
  for (1) net annualized arithmetic active return versus SPY using paired
  20-session circular block bootstrap (5,000 replicates, seed 20261114),
  **and** (2) annualized FF5-plus-momentum alpha using Newey–West HAC(20).
  Both must hold. Report all failures.

## Data that must be supplied

Provide a consistent, documented, time-stamped **point-in-time** data feed
for SPY's **unadjusted closing price**, ex-date cash dividends, share-split
ratios, and observed 13-week Treasury yield (^IRX) at each session close.
The tracker builds an **online dividend-reinvested index** from these
observations so later vendor revisions of historical adjusted closes cannot
silently alter past simulated signals.

This index is still an approximation: dividend reinvestment at ex-date is
not a guaranteed execution; the 5-bp cost and ^IRX proxy are not verified
broker fills or investable T-bill returns. `SplitRatio` means shares **after**
the split per share **before** the split. `DividendPerShare` means the cash
entitlement *per post-split share* on the ex-date. Use explicit `0` dividends
and `1` splits when none occurred, never missing entries. If the provider
cannot furnish these definitions reliably, **do not start**.

Warmup CSV **exact header** (all numeric data rows must be real, sourced
observations; the snippet below is only the schema):

```csv
Date,Close,DividendPerShare,SplitRatio,CashYieldPct
```

The warmup input is not committed to GitHub, and neither is the private
rolling ledger in `data/prospective/`. **Keep redundant private backups** of
the journal after every appended observation; a transient workspace is
not persistent storage. The hash chain detects accidental editing but is
not a trusted external timestamp or protection from an adversary who
rewrites every record. Publicly committing daily digest anchors (without
licensed quote data) would further strengthen the audit if desired.

## Operator commands

After the verified **2026-11-13** close and strictly **before** the NYSE
opens on 2026-11-16:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 prospective_paper.py init \
  --warmup /absolute/private/path/spy_warmup_through_2026-11-13.csv \
  --source-id 'REAL_PROVIDER_AND_DATASET_VERSION'
```

The initialization refuses a missing/late warmup, nonconsecutive sessions,
missing dividends/splits/yields, and an existing ledger. As of the status
above, **do not initialize**: a November 13 intraday ^IRX quotation or the
November 12 value is not evidence of the November 13 final close.

For every subsequent NYSE session, once its close is published and **before
the next exchange session opens**, append exactly one observation. Replace
the placeholders below with verified, actual values and an archived source
identifier (this is a template, **not** an instruction to enter fictitious
values):

```bash
python3 prospective_paper.py record \
  --date YYYY-MM-DD \
  --close ACTUAL_CLOSE \
  --dividend-per-share ACTUAL_EX_DATE_DIVIDEND_OR_ZERO \
  --split-ratio ACTUAL_SHARE_RATIO_OR_ONE \
  --cash-yield-pct ACTUAL_ANNUAL_IRX_QUOTE \
  --observed-at-utc 'ACTUAL_ISO_TIMESTAMP_WITH_UTC_OFFSET' \
  --source-ref 'PROVIDER_ARCHIVE_OR_DOCUMENT_REFERENCE'
```

**Never run placeholder commands.** Once ingested, the program validates
that it is the *next* exchange session and that the source observation was
timestamped after the session closed. Backfilling a missed day is explicitly
rejected as prospective evidence. The private append-only file is protected
with a record-level SHA-256 chain and file locking.

To audit the number of completed return sessions, the journal hash, and the
next delayed trading signal **without seeing interim returns**:

```bash
python3 prospective_paper.py status
```

The state and signal are for a hypothetical, fully funded paper portfolio.
No brokerage API is connected, and this program cannot submit an order.

At the fixed endpoint only, obtain the **first available official Kenneth
French daily** FF5 and momentum archives covering every prospective date,
immediately archive the downloaded ZIPs, record their checksums/vintage, and
generate the single final report. (The existing 2026-09 pinned factors do
**not** cover the new observation period.)

```bash
python3 prospective_paper.py finalize \
  --ff5-zip /absolute/private/path/official_ff5_daily.zip \
  --momentum-zip /absolute/private/path/official_mom_daily.zip \
  --output experiments/prospective_final.json
```

The finalizer will fail until 252 real prospective intervals and all matching
factor dates exist; its output path cannot overwrite an existing report.
The final report will include annualized net return, SPY benchmark return,
drawdown, volatility, turnover, Sharpe, paired active-return uncertainty,
FF5+momentum alpha uncertainty, and whether both primary criteria passed.
This is **prospective historical paper evidence**, not a realized investment
profit, audited execution, or a guarantee of alpha.

## Calendar and operational limitations

For 2026–2027, the runner uses published NYSE holidays and 1 p.m. early
closes, with a fail-closed guard on missing or reordered sessions. Reference:
[NYSE Holidays & Trading Hours](https://www.nyse.com/trade/hours-calendars).
Unscheduled market closures require separate documented review before any
continuation; do not silently overwrite missing sessions. The tracker is
**manual**: it is not an automatically scheduled data collector and has no
price-service credentials, persistent cloud runner, or live broker access.

**Next operator decision:** verify and archive the final 2026-11-13 ^IRX
quote, finish and independently check the full private warmup CSV, and choose
a durable private ledger destination before the Monday 2026-11-16 market
open. If the required inputs cannot be authenticated by the deadline, the
experiment must remain *not started*; a later launch would require a
documented preregistration amendment.

### Optional read-only Alpaca CSV preparation

`collect_alpaca_spy.py` now provides a **read-only**, date-gated Alpaca
SIP/raw bar downloader and archive validator. It requires separately verified
per-date dividend, split and ^IRX yield inputs, saves exact raw responses and
checksums outside Git, and **never** initializes this journal or submits an
order. See [ALPACA_DATA_RUNBOOK.md](ALPACA_DATA_RUNBOOK.md) for paper-key
setup, empty supplemental CSV templates, commands and caveats. This is only
a data preparation tool; the experiment has not yet started.
