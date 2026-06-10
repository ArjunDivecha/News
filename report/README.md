# Unified Daily Market & Portfolio Report

**One command replaces the whole legacy pipeline** (Phase 0 holdings feed,
Step 4 market report, Phase 2 portfolio report):

```bash
python3 report/main.py
```

That's it. ~2.5 minutes later you have a PDF.

---

## What it does (in 10th-grade English)

1. **Holdings** - Logs into Schwab and Interactive Brokers and downloads
   every position in every account, live. If Schwab's weekly token has
   expired it walks you through re-authorizing. If TWS isn't running it
   launches it and waits for you to log in. If a broker still can't be
   reached, it uses the last saved snapshot and stamps the report STALE.
2. **Prices** - Downloads 1 year of daily prices for ~800 ETFs (the whole
   market universe plus everything you own) from Yahoo Finance in one
   batched call. Free, no Bloomberg terminal needed.
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
| `ibkr_fetch.py` | IBKR subprocess (runs under `.venv-ibkr312`, Python 3.12) |
| `analytics.py` | ALL financial math (pure functions, fully unit-tested) |
| `prompt.py` | Builds the LLM data package |
| `llm.py` | Claude Opus call with extended thinking + retries |
| `pdf.py` | Markdown -> HTML -> PDF (PrinceXML, light mode) |
| `db.py` | SQLite layer (WAL, idempotent upserts) |
| `prompts/system.md` | The system prompt (the report's "personality") |
| `build_universe.py` | One-time: builds `data/universe.xlsx` |
| `migrate_history.py` | One-time: seeded history from legacy databases |
| `etf_map.py` | Bloomberg index -> ETF mapping (from ETF migration) |

## Tests

```bash
python3 -m pytest tests/ -v     # 25 tests on the financial math
```

The tests lock down the exact bugs found in the legacy system review:
short positions contribute positively when prices fall, contributions are
in bps (no double scaling), missing prices can't poison portfolio totals,
YTD anchors to the prior year's last close, and the sum of contributions
ties out EXACTLY to dollar P&L over beginning-of-day gross.

## Conventions (uniform everywhere)

- Returns in **percent**, contributions in **basis points**
- Shorts: negative market value, negative weight; the sign math is
  automatic - nothing flips signs downstream
- Weights vs **gross** exposure; contributions use **beginning-of-day**
  weights so they tie to dollar P&L
- Missing data is **reported, never papered over** (fail-is-fail)

## Requirements

- Python 3.14 (`yfinance`, `pandas`, `anthropic`, `schwabdev`,
  `python-dotenv`, `markdown`, `pytest`)
- `.venv-ibkr312` venv with `ib_insync` (IBKR only works on Python 3.12)
- PrinceXML (`brew install prince`) for PDFs
- `.env` at repo root: `ANTHROPIC_API_KEY`, `SCHWAB_APP_KEY`,
  `SCHWAB_APP_SECRET`
- TWS or IB Gateway running and logged in (it will launch it if not)

## Daily history it maintains

- `prices` - adjusted closes for the full universe (self-healing: every
  run refetches a year)
- `portfolio_snapshots` / `portfolio_summary` - your book, day by day,
  including alpha vs beta-implied return
- `reports` - every report's executive summary is fed back into the next
  report's prompt, so the narrative has memory (it migrated the legacy
  reports' summaries too)

VERSION: 1.0 (2026-06-09) - initial release on the `rearchitect` branch
