# Alpaca read-only daily data collector: operator runbook

**State: code/tests prepared, NO data collected, NO prospective study started, NO orders.**

`collect_alpaca_spy.py` is a reproducible **read-only market-data client**
for the locked SPY prospective paper simulation. It only makes HTTPS **GET**
requests to Alpaca's `data.alpaca.markets` stock bars endpoint. It has no code
for submitting paper or live orders and never touches a brokerage trading
endpoint. It does not automatically append to the paper ledger.

Alpaca's standard Basic market-data access offers historical consolidated
**SIP** prices after a roughly **15-minute delay**; free **real-time** data
may be **IEX-only**, which is not the full consolidated close. This collector
requires `feed=sip`, `adjustment=raw`, and `timeframe=1Day`; there is **no
automatic IEX fallback**. It additionally waits **20 minutes after the
scheduled NYSE close** before making a request.

Official reference:
[Alpaca market data API](https://docs.alpaca.markets/us/docs/about-market-data-api),
[historical stock bars](https://docs.alpaca.markets/us/reference/stockbars),
[paper credentials](https://docs.alpaca.markets/us/docs/paper-trading).

## 1. Create Alpaca paper credentials (on your computer)

1. Open [Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading).
2. Sign up, select **Paper**, and generate a **paper key and secret**. A
   paid real-time subscription is not necessary for our after-close design
   if your account grants 15-minute-delayed historical SIP access.
3. Keep the credentials **outside Git, chat, screenshots, and exported CSVs**.
   Configure these two environment variables in your private terminal or
   credential manager:

   ```text
   ALPACA_API_KEY_ID
   ALPACA_API_SECRET_KEY
   ```

   A paper account's API credentials can still perform *paper* order actions
   in other programs; our script never sends such requests. Never provide
   live account keys for this collector.

4. Install project requirements and check offline validation:

   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m unittest discover -s tests -p test_collect_alpaca_spy.py -v
   ```

The tests use invented bars and local mock responses; they cannot verify your
real API entitlements. The actual collection command is the first authorized
read-only API call.

## 2. Provide independently checked non-price inputs

Alpaca's stock **bar** endpoint supplies OHLCV prices, **not** the three
other fields in the frozen study. Each session also requires:

- **DividendPerShare:** verified SPY cash dividend per share on **ex-date**;
  enter `0` only after checking there was no dividend.
- **SplitRatio:** number of new shares per old share; enter `1` only after
  checking there was no split.
- **CashYieldPct:** the actual observed annual **^IRX 13-week Treasury
  discount-yield quote**, written as a percentage (e.g., a 4% quote is `4`,
  *not* `0.04`). Do not silently substitute a different series.
- **CorporateActionSourceRef / CashYieldSourceRef:** references recording
  the genuine data source and observation/check date for each session.

Alpaca also offers a separate corporate-actions data API, but its own
documentation warns that corporate-action records can arrive late. Thus
this collector intentionally does **not** equate "nothing returned yet" to
"no dividend/split." Verify events against an independent issuer/corporate
action source and archive the reference; the references are operator
attestations, not a claim of independent audit.

Official reference: [Alpaca corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1).

Generate an **empty** private warmup supplement template:

```bash
python3 collect_alpaca_spy.py template --kind warmup \
  --output data/prospective/alpaca/warmup_aux.csv
```

It includes exactly 21 prospective *warmup* trading sessions through
**Friday, November 13, 2026**. Its columns are:

```csv
Date,DividendPerShare,SplitRatio,CashYieldPct,CorporateActionSourceRef,CashYieldSourceRef
```

**The generated numeric/source cells are intentionally empty.** Fill all six
fields with verified values, including explicit zeros/ones only where
appropriate; any missing date, source or value fails validation. None of
these private data files are tracked by Git.

## 3. After November 13's close, archive the warmup

After **4:20 p.m. Eastern on November 13**, and strictly before **9:30 a.m.
Eastern on Monday, November 16**, with your filled supplement and credentials:

```bash
python3 collect_alpaca_spy.py warmup \
  --supplements data/prospective/alpaca/warmup_aux.csv

python3 collect_alpaca_spy.py verify \
  --bundle data/prospective/alpaca/warmup_through_2026-11-13
```

The collector will create three **new, non-overwriting** private files:

```text
data/prospective/alpaca/warmup_through_2026-11-13.csv
data/prospective/alpaca/warmup_through_2026-11-13.alpaca_raw.json
data/prospective/alpaca/warmup_through_2026-11-13.manifest.json
```

The manifest has the exact source query, retrieval timestamp, source
references, and SHA-256 checksums of the raw API bytes, filled supplement
input, and output CSV. No subsequent API revisions can silently overwrite
these archived bytes. The `verify` command checks the raw JSON, CSV values,
dates, source identifiers and hashes.

**Only after reviewing and backing up those files** may the operator
explicitly initialize the *separate* frozen paper study:

```bash
python3 prospective_paper.py init \
  --warmup data/prospective/alpaca/warmup_through_2026-11-13.csv \
  --source-id 'alpaca_v2_stocks_bars_SPY_1Day_sip_raw + verified_auxiliary_sources'
```

Do **not** run `init` if the supplement was incomplete, source identities
cannot be authenticated, the time gate has passed, or the vendor response
isn't available. In those cases the study did **not start**; any revised
start requires an explicitly documented preregistration amendment.

## 4. Collect one later daily close

For example, for Monday **2026-11-16**, make the private supplement template,
fill the one row with sources available that day, and run after **4:20 p.m.
Eastern** and before Tuesday's market open:

```bash
python3 collect_alpaca_spy.py template --kind daily --date 2026-11-16 \
  --output data/prospective/alpaca/session_2026-11-16_aux.csv

python3 collect_alpaca_spy.py daily --date 2026-11-16 \
  --supplements data/prospective/alpaca/session_2026-11-16_aux.csv

python3 collect_alpaca_spy.py verify \
  --bundle data/prospective/alpaca/session_2026-11-16
```

This saves `session_2026-11-16.csv`, `.alpaca_raw.json` and
`.manifest.json`, and **does not mutate the experiment's ledger**. Review
the one-row CSV and manifest; if the actual experiment was initialized,
use the separate `prospective_paper.py record` command documented in
`PROSPECTIVE_PAPER_RUNBOOK.md` with the CSV's real numbers, the manifest's
`retrieved_at_utc` as the observation time, and a `--source-ref` pointing
to that manifest/checksum. Data can be fetched automatically by this command
once authoritative supplemental data are provided; the experiment does
**not** currently schedule unattended collection or execute orders.

## 5. Safety and provenance limitations

- `data/prospective/` and `.env*` are excluded from version control.
- Files are created private (`0600`) and are **never overwritten**. An
  interruption may leave a partial bundle; preserve it for incident review.
- The raw SIP daily bar is an **Alpaca vendor aggregate**, not independently
  audited official closing-auction evidence. Checking its checksum proves
  local integrity, not the economic correctness of its price or entitlement.
- Dividends, split ratios, and ^IRX yields are **not fetched** from Alpaca's
  stock bars and must be verified independently. If any data source is
  unsuitable, pause before claiming prospective data are valid.
- Keep multiple durable private backups of the source bundle and the
  separate append-only ledger. A hash alone is not a trusted timestamp.
- No backfilling or strategy modification is permitted after the preregistered
  paper start. Never describe a CSV simulation as real or broker-filled PnL.
