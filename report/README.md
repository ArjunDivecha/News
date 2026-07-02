# Unified Daily Market & Portfolio Report

**One command replaces the whole legacy pipeline** (Phase 0 holdings feed,
Step 4 market report, Phase 2 portfolio report):

```bash
python3 report/main.py
```

That's it. ~2.5 minutes later you have a PDF.

---

## What it does (in 10th-grade English)

1. **Holdings** - Pulls Schwab and IBKR positions automatically. Both
   brokers can run fully unattended:
   - **Schwab**: Playwright auto-auth (headless Chromium, TOTP) or
     interactive OAuth fallback
   - **IBKR**: Flex Web Service (token-based, no TWS) or TWS subprocess
     fallback
2. **Prices** - Downloads 1 year of daily prices for ~800 ETFs from
   Yahoo Finance in one batched call.
3. **Analytics** - Computes everything: returns, YTD, volatility, betas,
   factor exposures, per-position contributions (in bps), alpha vs what
   the portfolio's beta predicted, peer-group comparisons, streaks,
   percentile extremes.
4. **Report** - Sends ONE complete data package to Claude Opus (with
   extended thinking) which writes a single unified report covering the
   market, the portfolio, and - most importantly - the bridge between
   them.
5. **PDF** - Renders a clean light-mode PDF via PrinceXML and archives
   everything to a database so tomorrow's report remembers today's.

## Outputs

| File | What it is |
|---|---|
| `outputs/unified/Unified_Report_<date>.pdf` | The report |
| `outputs/unified/Unified_Report_<date>.md` | Same content, canonical text |
| `outputs/unified/Data_Package_<date>.md` | Exactly what the LLM saw (audit) |
| `data/report.db` | Prices, snapshots, report archive |
| `data/holdings.xlsx` | Latest live holdings snapshot |

## Command-line options

```bash
python3 report/main.py                   # full run
python3 report/main.py --no-llm          # data + analytics only (free, fast)
python3 report/main.py --non-interactive # cron mode: never prompts,
                                         # falls back to stale holdings
python3 report/main.py --date 2026-06-09 # as-of date for analytics
```

## Files in this package

| File | Purpose |
|---|---|
| `main.py` | Orchestrator - the only thing you run |
| `config.py` | Every path, constant, window, and model name |
| `data.py` | Batched yfinance download with loud coverage validation |
| `holdings.py` | Schwab + IBKR pulls, preflights, stale fallback |
| `notify.py` | Emails the report PDF (Mail.app or SMTP) |
| `schwab_auto.py` | Playwright headless OAuth (auto-refresh tokens) |
| `ibkr_flex.py` | IBKR Flex Web Service HTTP client (no TWS) |
| `ibkr_fetch.py` | IBKR TWS subprocess (fallback, `.venv-ibkr312`) |
| `run_daily.sh` | launchd wrapper — runs pipeline + emails |
| `analytics.py` | ALL financial math (pure functions, fully unit-tested) |
| `scenarios.py` | Scenario risk engine (episode-calibrated stress tests, crash beta, liquidity ladder) |
| `tag_analytics.py` | Tier-3 multi-label tag views (day-type, tilts, bridge, concentration) |
| `tags.py` | DeepSeek dynamic tagger (multi-label; cached in report.db; overrides) |
| `names.py` | Ticker -> full security name (Yahoo, cached in report.db) |
| `prompt.py` | Builds the LLM data package |
| `llm.py` | Claude **Fable 5** call (subscription CLI first; streaming API fallback with refusal-fallback to Opus 4.8) + truncation guard |
| `pdf.py` | Markdown -> HTML -> PDF (PrinceXML, light mode) + table validation |
| `db.py` | SQLite layer (WAL, idempotent upserts) |
| `prompts/system.md` | The system prompt (the report's "personality") |
| `build_universe.py` | One-time: builds `data/universe.xlsx` |
| `migrate_history.py` | One-time: seeded history from legacy databases |
| `etf_map.py` | Bloomberg index -> ETF mapping (from ETF migration) |

## Tests

```bash
python3 -m pytest tests/ -v     # 35 tests (financial math + report pipeline)
```

`test_analytics.py` locks down the exact bugs found in the legacy system review:
short positions contribute positively when prices fall, contributions are
in bps (no double scaling), missing prices can't poison portfolio totals,
YTD anchors to the prior year's last close, and the sum of contributions
ties out EXACTLY to dollar P&L over beginning-of-day gross.

`test_report_pipeline.py` locks down the 2026-06-20 review fixes (below):
the PDF table validator rejects a truncated report, the data package emits a
summary table + breadth + proxy/scope labels, `compute_bridge` backfills
tier-2 (and labels off-universe names "Portfolio-Specific"), and percentiles
are integer-valued with sparse-data masking.

## Tier-3 tag views (2026-07-01)

Multi-label tag analytics layered on top of the report — **purely additive and
gated** by `SETTINGS["enable_tag_views"]` (env `REPORT_ENABLE_TAG_VIEWS=0` to
revert to the exact prior report). Every holding is dynamically tagged
(`tags.py`, DeepSeek + manual overrides, cached in `report.db`); tags are
orthogonal axes (AssetClass / Region / Sector / Style / Strategy / Size /
Duration). `tag_analytics.py` computes, as pure functions:

- **Market — day type:** VIX Rule-of-16 noise gate, cross-sectional dispersion,
  two-dimensional breadth (macro/factor day vs stock-picker's day).
- **Market — leadership:** universe-**demeaned** tag tilts (n≥3), long-short
  style/region spreads, a Region×Sector grid (n≥5).
- **Portfolio — posture:** tag active tilts vs a **60/40 ACWI/TLT** benchmark
  (benchmark tags pinned in `tags.py`), the **bridge** (each big book tilt WITH
  or AGAINST today's tape), and **concentration** (1/HHI effective positions and
  effective tags per axis).

- **Portfolio — P&L & reconciliation:** per-axis tag P&L attribution (each
  position's contribution split equally within an axis, so tags sum to the day's
  P&L per axis), exposure-vs-realized-beta (flags hidden co-movement / inert
  themes), and an EM country-vs-style η² decomposition.
- **Household asset allocation** (`compute_asset_allocation`): ONE hierarchical
  table (live book + GMO) — **Equities** and **Bonds** each split into US /
  International / EM sub-rows, then **Alternatives** and **Cash**. Weights are
  net exposure; bucket returns are P&L over gross so a short signs correctly
  (IBKR shorts show as negative equity, its cash incl. short proceeds stays
  positive). Multi-asset and global funds are **looked through** to their
  underlying mix via `config.FUND_LOOKTHROUGH` (GMO Benchmark-Free → equity/
  bond/alt; the market-neutral GMO Equity Dislocation → Alternatives, not
  equity; global-equity funds → US/Int/EM), sourced from published composition
  files with an as-of date and cited in a footnote. There is no "Global"
  bucket; a global fund lacking look-through data is surfaced as Unclassified,
  never fabricated. Everything else is classified from tags (`classify_holding`).
  Off-broker holdings with no daily mark (e.g. the Baupost LP in
  `config.MANUAL_HOLDINGS`) are carried at a fixed value, appear as their own
  sub-portfolio sleeve, and — since they have no price — earn a **generic
  asset-class proxy return** (US eq=SPX, Intl=EAFE, EM=EM, US bonds=AGG, cash=0)
  applied to their policy distribution, so they still contribute a sensible
  return to the sleeve, the household total, and the allocation buckets.

Tags are correlated, so tilts are **never summed across axes** — each is an
independent excess-vs-benchmark or excess-vs-universe reading. The blocks are fed
to the LLM as `TIER-3 TAG VIEWS` and `HOUSEHOLD ASSET ALLOCATION` package
sections; `system.md` makes them **required, table-first** sections (Asset
Allocation; day-type & leadership tables in The Tape; tag-tilt / bridge / tag-P&L
tables in The Bridge & Positioning) whenever the package carries their data.
Locked down by `tests/test_tag_analytics.py` (25 tests, including a flag-OFF
byte-identity guarantee and the NO-n/a rule).

## Investor-flow restructure + scenario risk (2026-07-01)

The report now follows the owner's reading order — *what happened → how did my
money do → what should I know → what could hurt me → bottom line* — as six
sections: Executive Summary, The Tape, **My Money** (household line first, then
the look-through allocation table, sub-portfolios, and the live-book detail),
**Worth Knowing** (3-6 data-triggered bullets: extremes, streak flips, peer
divergences, artifacts the reader would misread), **Scenario Risk**, Bottom
Line. The Tier-1/Theme tables were cut from the rendered report (their data
still feeds the narrative); the WITH/AGAINST structural rows and artifact-beta
rows are no longer rendered daily.

**Scenario Risk** (`scenarios.py`) is the standing stress panel. Six scenarios,
each a documented, episode-calibrated shock vector applied to the household's
actual look-through slices with per-symbol overrides for concentrated names
(Vietnam CEF, GMO Beyond China, the growth short, the market-neutral Equity
Dislocation): US −40% (2008-09), Asia/EM crisis (1997-98), China/Taiwan event,
inflation +300bp (2022), USD +10% spike, tech/growth crash (2000-02). Output:
household $ and % impact, biggest hurts, biggest cushions, and every shock
assumption printed for audit (edit them in `report/scenarios.py`). Plus a
structural panel: **crash beta** (full-sample vs worst-decile-S&P-days beta of
the current-weights book) and a **liquidity ladder** (cash / ETF / daily-NAV /
CEF-with-discount-risk / LP-lockup). Shorts sign correctly (a growth short
GAINS in the tech-crash scenario); these are labeled first-order estimates,
never predictions. Locked down by `tests/test_scenarios.py`.

## Report-writing hardening (2026-06-20 review)

A multi-agent review found the 2026-06-18 report was **silently truncated** —
the LLM output hit the exact `max_tokens` ceiling (32000) mid-table, dropping
Risks & Watchlist and Bottom Line, and it was saved + emailed with no error.
Fixes:

- **Token budget + streaming** (`config.py`, `llm.py`): `max_tokens` raised to
  64000 (opus-4-8 allows 128000), `thinking_effort` stays `max`. The call now
  **streams** (`messages.stream` + `get_final_message`) so the multi-minute
  generation doesn't trip the SDK's non-streaming timeout guard.
- **Fail-loud truncation guard** (`llm.py`): a `stop_reason == "max_tokens"`
  raises a non-retryable `ReportTruncatedError` — a partial report is never
  saved or rendered (FAIL IS FAIL).
- **PDF table validation** (`pdf.py`): every markdown table's rows are checked
  against the header before rendering; a malformed/truncated table raises
  rather than producing a broken PDF. `smarty` dropped (no `--`->en-dash).
- **Data package** (`prompt.py`): portfolio summary is now a table; YTD is
  labelled a current-weights proxy; the sub-portfolio total is "HOUSEHOLD
  TOTAL" (incl. GMO) vs the live-only book; a PORTFOLIO BREADTH block and a
  factor-overlap caveat were added; prior exec summaries are stripped to prose
  and tagged with "N days ago".
- **Analytics** (`analytics.py`): `compute_bridge` backfills tier-2 peer labels
  from the full universe and reports breadth; percentiles are integer + masked
  when sparse.
- **System prompt** (`prompts/system.md`): requires the alpha caveat when the
  book's tilt diverges from SPY, a verbatim STALE-first-sentence rule, and a
  completeness rule (all seven sections, never truncate).

## No n/a + market-holiday fix (2026-06-22)

The 2026-06-22 report came out ~84% "n/a". Root cause: **2026-06-19 was
Juneteenth** (US markets closed) — only ~125/808 tickers printed, leaving a
sparse row in the price matrix. `daily_returns` uses `pct_change`, which
compares each row to the immediately-prior one, so every US ticker's 06-22
return was computed against 06-19's blanks → NaN, and the book "return" was a
garbage -8.33% from a single $5.50 stub.

- **Core fix** (`data.py` `filter_sparse_rows`, applied in `main.py` right after
  `load_prices`): drop date rows below 50% of peak coverage. On the real DB this
  drops exactly the 7 US market holidays (~125 tickers each); the thinnest real
  session is 790, so there's no false-positive risk. The "1d" return on a
  post-holiday day then correctly spans the last two *real* sessions (06-22 vs
  06-18). Verified: factor returns 15/15 real, book 1d -0.18% (sane).
- **NO-NA rule** (project rule in `AAA Backup/CLAUDE.md`): the report never shows
  "n/a". `_md_table` renders an undefined cell as `—`; a position that didn't
  print on the as-of date shows its **last-available value with a `*`** (stale),
  with a footnote. `analytics.py` carries forward last-available 1d/factor
  returns; `compute_market` fails loud if SPY (the benchmark) is ever absent.

## Names, not tickers (2026-06-20)

The report refers to every asset by its full name, not its ticker symbol
(`EWY` -> "iShares MSCI South Korea ETF", `INTC` -> "Intel Corporation").
`names.py` resolves tickers via Yahoo `longName`, cached per ticker in the
`security_names` table so Yahoo is hit once per symbol. The stored universe
names are unusable (Goldman-basket codenames like "GS Korea L PB H Profit"),
so Yahoo is the source of truth. The handful of holdings Yahoo can't resolve
(CUSIP cash lines, a few OTC/GMO funds) keep their raw ticker/ID. The system
prompt enforces names-only in both tables and prose.

## Conventions (uniform everywhere)

- Returns in **percent**, contributions in **basis points**
- Shorts: negative market value, negative weight; the sign math is
  automatic - nothing flips signs downstream
- Weights vs **gross** exposure; contributions use **beginning-of-day**
  weights so they tie to dollar P&L
- Missing data is **reported, never papered over** (fail-is-fail)

## Requirements

- Python 3.14 (`yfinance`, `pandas`, `anthropic>=0.109`, `schwabdev`,
  `python-dotenv`, `markdown`, `pytest`)
- PrinceXML (`brew install prince`) for PDFs
- `.env` at repo root: `ANTHROPIC_API_KEY`, `SCHWAB_APP_KEY`,
  `SCHWAB_APP_SECRET`, and **`IBKR_FLEX_TOKEN` + `IBKR_FLEX_QUERY_ID`**
  (see IBKR Flex setup below)
- TWS / IB Gateway: **no longer required** when Flex is configured.
  The `.venv-ibkr312` venv is kept as an optional TWS fallback.

### IBKR Flex Web Service setup (one-time, 5 minutes)

This is the primary IBKR path — no TWS login, no `.venv-ibkr312`, no
interactive prompts. Once set up, it just works.

1. **Log into Client Portal** at https://ndcdyn.interactivebrokers.com
2. **Performance & Reports → Flex Queries → Create New Query**
   - Name it "Daily Positions"
   - Add an **Open Positions** section (all default fields are fine)
   - Optionally add **Cash Transactions** to include cash balances
   - Save — note the **Query ID** (a number)
3. **Flex Web Service Configuration** (gear icon on the Flex Queries page)
   - Enable Flex Web Service → **Generate Token**
   - Copy the token and query ID into `.env`:
     ```
     IBKR_FLEX_TOKEN=<your-token-here>
     IBKR_FLEX_QUERY_ID=<your-query-id-here>
     ```

That's it. The token is permanent until you regenerate it; no expiration,
no re-auth, no TWS dependency. If Flex is NOT configured (env vars absent),
the system falls back to the TWS subprocess path automatically.

### Schwab auto-auth setup (optional, one-time)

Schwab requires OAuth every ~7 days. Two modes:

**Auto-auth (Playwright — unattended):**
1. Set three env vars in `.env`:
   ```
   SCHWAB_USERNAME=<your-schwab-login-id>
   SCHWAB_PASSWORD=<your-schwab-password>
   SCHWAB_TOTP_SECRET=<your-base32-totp-secret>
   ```
2. That's it. `report/main.py` will now refresh Schwab tokens
   automatically in headless Chromium — no browser, no prompts.

**Interactive (default fallback):**
If the three env vars above are left empty, the system falls back to
the original flow: opens a browser, you paste the redirect URL back.
This is the same behavior as before.

> **Getting your TOTP secret:** When you set up 2FA on Schwab, your
> authenticator app (Google Authenticator, Authy, 1Password) received a
> Base32 secret key — usually shown as a text string or QR code during
> setup. This is the value for `SCHWAB_TOTP_SECRET`. If you don't have
> 2FA enabled on Schwab, leave `SCHWAB_TOTP_SECRET` empty — auto-auth
> will skip the TOTP step.

## Daily history it maintains

- `prices` - adjusted closes for the full universe (self-healing: every
  run refetches a year)
- `portfolio_snapshots` / `portfolio_summary` - your book, day by day,
  including alpha vs beta-implied return
- `reports` - every report's executive summary is fed back into the next
  report's prompt, so the narrative has memory (it migrated the legacy
  reports' summaries too)

## Automated daily run (launchd — every weekday at 1:05 PM PT)

The report fires automatically after market close. No manual steps after
initial setup.

### One-time setup (2 minutes)

**1. Set your email in `.env`:**
```
REPORT_EMAIL_TO=<your-email@example.com>
```

That's it for zero-config. The automation uses your existing Mac Mail.app
account — no SMTP credentials needed. For SMTP (Gmail, Fastmail, etc.),
add the optional `SMTP_HOST`/`SMTP_USER`/`SMTP_PASS` variables.

**2. Load the launchd job:**
```bash
launchctl load ~/Library/LaunchAgents/com.news.daily-report.plist
```

**3. Verify it's scheduled:**
```bash
launchctl list | grep com.news.daily-report
```

### What happens every weekday at 1:05 PM PT

1. `launchd` fires `report/run_daily.sh`
2. The wrapper runs `python3 report/main.py --non-interactive`
3. Pipeline: holdings → prices → analytics → LLM report → PDF
4. `report/notify.py` emails you the PDF
5. Everything logged to `outputs/unified/daily_run.log`

### If something fails
- Check `outputs/unified/daily_run.log` for the full trace
- `launchctl error` shows launchd-level error codes
- The job won't retry automatically — if you miss a day, run
  `python3 report/main.py` manually

### Troubleshooting
```bash
# Stop the schedule
launchctl unload ~/Library/LaunchAgents/com.news.daily-report.plist

# Restart
launchctl load ~/Library/LaunchAgents/com.news.daily-report.plist

# Check if running
launchctl list com.news.daily-report

# Manual run (same as what launchd does)
./report/run_daily.sh

# Last 50 lines of the log
tail -50 outputs/unified/daily_run.log
```

VERSION: 1.0 (2026-06-09) - initial release on the `rearchitect` branch
