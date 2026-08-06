---
type: "Reference"
title: "Operations Runbook"
description: "Practical instructions for running the report system: runtime requirements, environment variables, scheduling, PDF rendering, and troubleshooting."
---

# Operations Runbook

This page collects the practical instructions for running the report system safely.

## Core runtime requirements

From the root README and package docs, the production environment expects:

- Python 3.14 for the main report path,
- a separate Python 3.12 environment for the optional IBKR TWS fallback,
- `yfinance`, `pandas`, `anthropic`, `schwabdev`, `python-dotenv`, `markdown`, `pytest`,
- PrinceXML for PDF rendering,
- a root `.env` file with the required API and broker variables,
- active broker access for Schwab and/or IBKR depending on the configured path.

Fable's Desk (the optional second LLM pass) additionally requires:

- the Claude CLI on PATH (subscription auth — the Desk uses WebSearch/WebFetch, which only the CLI backend supports),
- `duckdb` (Python package) for the ASADO country-signal snapshot; degrades gracefully if missing,
- the ASADO DuckDB warehouse at the path configured in `config.PATHS["asado_db"]`; the Desk runs without it if unavailable,
- `REPORT_ENABLE_FABLE_DESK=1` (the default) to enable; set to `0` to skip the Desk entirely.

## How to run

The main daily command is:

```bash
python3 report/main.py
```

Useful variants documented in the package README:

- `python3 report/main.py --no-llm` — data and analytics only
- `python3 report/main.py --non-interactive` — cron / launchd mode
- `python3 report/main.py --date YYYY-MM-DD` — as-of override

The package README also documents `report/run_daily.sh` as the launchd wrapper used for scheduled runs.

## Broker connectivity

### Schwab
`report/holdings.py` supports two paths:

- Playwright auto-auth with username, password, and TOTP secret
- the original interactive fallback that opens a browser and waits for the redirect URL

The code checks the age of the Schwab refresh token and can warn or prompt when it is close to expiry.

### IBKR
`report/holdings.py` supports:

- IBKR Flex Web Service as the primary path,
- TWS / IB Gateway subprocess fallback when Flex is not configured.

The package README gives the one-time setup steps for Flex and explains that the token is effectively permanent until regenerated.

## Data and storage

The system writes its working state to `data/report.db` through `report/db.py`.

Important tables include:

- assets
- prices
- portfolio_snapshots
- portfolio_summary
- reports
- security_names
- desk_ideas — Fable's Desk idea journal (title, action, conviction, horizon, invalidation, status, grade), graded by the Desk itself on subsequent days

The database is used both for archival continuity and for caches such as names and tags.

### One-time history migration
`report/migrate_history.py` is a one-time tool that seeds `data/report.db` from the legacy `market_data.db` and `portfolio.db`: portfolio-level history (`portfolio_summary`) and prior report executive summaries (`reports`) for day-one prompt continuity. Legacy asset prices are deliberately not migrated — legacy daily prices are Bloomberg index levels while the new system tracks ETF prices, so mixing the two across the cutover boundary would corrupt every return; asset history is instead backfilled by a 1-year batched yfinance fetch on the first run. The script takes byte-for-byte backups of both legacy databases before reading. Run once at cutover: `python3 report/migrate_history.py`.

## PDF and report output

`report/pdf.py` renders the final Markdown into HTML and PDF using PrinceXML. The output directories are configured in `report/config.py` and the package README describes the main artifacts:

- `outputs/unified/Unified_Report_<date>.pdf`
- `outputs/unified/Unified_Report_<date>.md`
- `outputs/unified/Data_Package_<date>.md`
- `outputs/unified/Fable_Desk_<date>.md` (standalone Desk run via `python3 report/fable_desk.py --date <date>`)
- `data/report.db`
- `data/holdings.xlsx`

## Report delivery (email)

The pipeline does not email the report itself — that is the job of `report/notify.py`, a standalone delivery step invoked by `report/run_daily.sh` after the PDF is on disk. It is intentionally outside `report/main.py`'s run loop so a delivery failure can never sink a successful report run.

`notify.send_report()` supports two transports, chosen by environment:

- **Mail.app (default)** — `send_via_mail_app()` drives macOS Mail via AppleScript (`osascript`). Zero config beyond `REPORT_EMAIL_TO`; uses the already-configured Mac Mail account, so no SMTP credentials are needed.
- **SMTP (when `SMTP_HOST` is set)** — `send_via_smtp()` connects to an SMTP relay (STARTTLS on port 587 by default) with optional `SMTP_USER` / `SMTP_PASS` auth and `SMTP_FROM` for the envelope sender.

The email subject carries the report date and a `STALE` flag when holdings were stale (`--stale`).

Relevant environment variables (root `.env`):

- `REPORT_EMAIL_TO` — recipient address (required to send at all).
- `SMTP_HOST` / `SMTP_PORT` — switch from Mail.app to SMTP (omit to stay on Mail.app).
- `SMTP_USER` / `SMTP_PASS` / `SMTP_FROM` — optional SMTP credentials and sender.

Run directly:

```bash
python3 report/notify.py outputs/unified/Unified_Report_<date>.pdf --date <date> [--stale]
```

### Delivery in the scheduled path

`report/run_daily.sh` is the launchd wrapper (called every weekday at 1:05 PM PT). After running the pipeline non-interactively, it finds the day's PDF and calls `notify.py`. Its failure policy is deliberately lenient: a missing `REPORT_EMAIL_TO` or Mail.app failure logs loudly but exits `0` so launchd sees success and does not retry — the PDF is already on disk and the report is not lost. This is why the delivery step lives in the wrapper, not in `report.main.run()`.

## Failure modes to know

The codebase is explicit about a few operational failure classes:

- missing benchmark data should fail loudly,
- sparse holiday rows should be dropped before analytics,
- stale holdings should be labeled as stale,
- `n/a` should never appear in the report,
- a truncated LLM generation should abort the run rather than being archived.

These are not just implementation details; they are part of the operational contract and are enforced by tests.

## Useful source files

- `README.md`
- `report/README.md`
- `report/main.py`
- `report/config.py`
- `report/data.py`
- `report/holdings.py`
- `report/llm.py`
- `report/pdf.py`
- `report/db.py`
- `report/migrate_history.py`
- `report/build_universe.py`
- `report/fable_desk.py`
- `report/asado.py`
- `report/notify.py`
- `report/run_daily.sh`

## For future changes

Before changing runtime behavior, verify:

- how the command is expected to run unattended,
- whether a data-quality safeguard is being relied on by the prompt or tests,
- whether a change touches both the live-book and household totals,
- whether output filenames or locations are referenced in docs or automation.
