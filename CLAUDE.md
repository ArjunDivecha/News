# CLAUDE.md — News (Unified Daily Market & Portfolio Report)

Operator's manual for coding agents. Global rules (light mode, doc headers,
`file://` links, FAIL-IS-FAIL, NO-`n/a`) live in `~/CLAUDE.md` and
`~/Dropbox/AAA Backup/CLAUDE.md` and are not repeated here. For deep dives,
read `openwiki/quickstart.md` and follow its links; `AGENTS.md` holds the
legacy universe-pipeline history.

## Purpose

One command (`python3 report/main.py`) pulls live Schwab + IBKR holdings,
downloads ~800 ETF closes from Yahoo Finance (no Bloomberg terminal), runs
unit-tested portfolio/market analytics, has Claude Fable 5 write a unified
market-and-portfolio report, and renders it to a PDF that is emailed daily by
launchd. Everything the model sees is a deterministic "data package"; the LLM
adds narrative, not numbers. There are two pipelines: the active daily report
in `report/`, and a dormant universe-construction chain (`Step 1-3/`,
`fine tuning/`) that only rebuilds `data/universe.xlsx`.

## Architecture map (load-bearing files, all under the repo root)

Repo root: `/Users/arjundivecha/Dropbox/AAA Backup/A Working/News`

- `report/main.py` — orchestrator; the only thing you run for a report.
- `report/config.py` — single source of paths, windows, model IDs, `POLICY`,
  `FUND_LOOKTHROUGH`, `MANUAL_HOLDINGS`, `ACCOUNT_NAMES`, `BENCHMARK`.
- `report/data.py` — batched yfinance download; `filter_sparse_rows` (holiday
  fix); `_ensure_fd_limit` (launchd fix); 90% coverage gate.
- `report/holdings.py` — Schwab + IBKR pulls, preflights, stale fallback.
- `report/analytics.py` — ALL financial math as pure functions
  (`compute_market`, `compute_portfolio`, `compute_subportfolios`,
  `compute_bridge`, `compute_policy_check`).
- `report/tags.py` + `report/tag_analytics.py` — DeepSeek multi-label tagger
  (cached in `report.db`) and tier-3 tag views + household allocation.
- `report/scenarios.py` — episode-calibrated stress engine.
- `report/prompt.py` — builds the LLM data package (exactly what the model reads).
- `report/llm.py` — Claude Fable 5 call: CLI subscription first, streaming API
  fallback (server-side refusal-fallback to Opus 4.8); truncation guard.
- `report/fable_desk.py` + `report/asado.py` — second Fable pass (live web +
  read-only ASADO country-signal snapshot).
- `report/pdf.py` — Markdown → HTML → PDF (PrinceXML) + table validation.
- `report/db.py` — SQLite layer (WAL, idempotent upserts).
- `report/prompts/system.md`, `report/prompts/fable_desk.md` — system prompts.
- `report/run_daily.sh`, `report/com.news.daily-report.plist` — launchd automation.
- `tests/` — pytest suite (5 files).

## Commands that work

- `python3 -m pytest tests/ -q` — **verified: 105 passed in ~1.5s** (2026-07-06).
  Note: `report/README.md` says "35 tests" and "25 tests" — both stale; actual is 105.
- `python3 report/main.py` — full daily run (unverified here: hits live Schwab/
  IBKR, Yahoo, and a metered/subscription LLM call — do not run casually).
- `python3 report/main.py --no-llm` — data + analytics only, no LLM/PDF
  (unverified here: still hits live brokers + Yahoo).
- `python3 report/main.py --non-interactive` — cron mode: never prompts, stale
  fallback (this is what launchd runs).
- `python3 report/main.py --date YYYY-MM-DD` — analytics as-of override.
- `python3 report/fable_desk.py --date YYYY-MM-DD` — standalone Desk run.
- `python3 report/build_universe.py` — rebuild `data/universe.xlsx` (rare).
- Contract validator (verified): `python3 /Users/arjundivecha/code/divecha/divecha/scripts/validate_contract.py --mode author <spec>`.
- Toolchain (verified present): Python 3.14.3, `prince` at `/opt/homebrew/bin/prince`, `claude` CLI on PATH.

## Data locations (absolute)

- Universe: `/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/universe.xlsx`
- History DB: `.../News/data/report.db` (~17 MB, WAL, gitignored)
- Live holdings snapshot: `.../News/data/holdings.xlsx` (gitignored, rewritten each run)
- GMO holdings (manual, gitignored): `.../News/GMO.xlsx`
- Reports/audit: `.../News/outputs/unified/{Unified_Report,Data_Package}_<date>.{pdf,md,html}`
- launchd log: `.../News/outputs/unified/daily_run.log`
- Secrets: `.../News/.env` (gitignored; see FLAGS below)
- ASADO warehouse (read-only, external repo): `/Users/arjundivecha/Dropbox/AAA Backup/A Working/ASADO/Data/asado.duckdb`
- Installed launchd job: `~/Library/LaunchAgents/com.news.daily-report.plist`

## Conventions & gotchas (repo-specific — these have bitten before)

- **launchd PATH**: launchd starts with a minimal PATH; `run_daily.sh` must set
  it. The Claude CLI lives in the nvm-managed node bin. If `daily_run.log` shows
  `Claude CLI: NOT FOUND`, the report **silently ran on metered API and skipped
  Fable's Desk** — there is no alert; check the log's first lines.
- **FD limit**: launchd's 256 soft maxfiles is exhausted by ~800 threaded
  yfinance downloads → SQLite `unable to open database file` → ~90% coverage
  loss. `data.py::_ensure_fd_limit` raises it to 8192. Do not remove it.
- **Market holidays**: `filter_sparse_rows` drops <50%-coverage date rows and
  must stay the single chokepoint right after `db.load_prices` in `main.py`.
  Removing it reintroduces the 2026-06-22 "84% n/a" flood.
- **SPY is mandatory**: `compute_market` raises if SPY is absent (no
  benchmark-less report). Do not wrap that in a try/except.
- **Additive sections fail quietly**: tag views, allocation, scenarios, and the
  policy check are wrapped in broad `try/except` in `main.py` (lines ~223, ~257,
  ~285) so a bug there ships a report *missing whole sections* with only a
  stderr traceback. Watch the console for `SKIPPED`.
- **Fable model rules**: thinking is always on — never send `thinking:disabled`
  or `budget_tokens`. `max_tokens=64000` (32000 truncated the 2026-06-18 report
  mid-table). Streaming is required.
- **Two Python envs**: main path = Homebrew `python3` 3.14; IBKR pull =
  `.venv-ibkr312` (3.12) Gateway subprocess. **IBKR Flex Web Service was
  REMOVED 2026-07-11 — do not re-add it.** Transient 1001 errors trip an
  account-level failed-attempts lockout (1025) that a new token does NOT
  clear and that locked Client Portal login itself. Live IBKR holdings
  require a logged-in IB Gateway/TWS session (Arjun logs in daily);
  otherwise the run uses the flagged stale snapshot.
- **Schwab**: refresh token expires every ~7 days; needs interactive OAuth
  unless the TOTP auto-auth env vars are set (`SCHWAB_USERNAME/PASSWORD/TOTP_SECRET`).
- **Off-broker positions** (Baupost LP, Anthropic/Perplexity private stakes)
  live in `config.MANUAL_HOLDINGS` — update by editing `config.py`, not Excel.
- **Never name Goldman Sachs or Bloomberg** in generated report text (categories/
  themes only) — enforced in `system.md`.

## Current state

- **Active / in production.** Runs daily via launchd at 1:05 PM PT; 105 tests pass.
- Report writer is Claude Fable 5 at `xhigh` effort via the Claude CLI
  subscription path. Fable moves to API pricing 2026-07-07 — the silent
  CLI→metered-API fallback (above) is now a real cost exposure, not just a
  degraded-quality one.
- Legacy universe pipeline (`Step 1-3/`, `fine tuning/`) is dormant; only needed
  to rebuild `data/universe.xlsx`. The old Phase 0/2/4 chain is in `archive/`.
- Known-fragile: unattended launchd runs have failed repeatedly (PATH, FD limit,
  coverage) — each fixed, but there is still **no automated alert on silent
  degradation** (see FABLE.md P1).
