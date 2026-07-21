---
type: "Reference"
title: "Operations Runbook"
openwiki_generated: true
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

## PDF and report output

`report/pdf.py` renders the final Markdown into HTML and PDF using PrinceXML. The output directories are configured in `report/config.py` and the package README describes the main artifacts:

- `outputs/unified/Unified_Report_<date>.pdf`
- `outputs/unified/Unified_Report_<date>.md`
- `outputs/unified/Data_Package_<date>.md`
- `outputs/unified/Fable_Desk_<date>.md` (standalone Desk run via `python3 report/fable_desk.py --date <date>`)
- `data/report.db`
- `data/holdings.xlsx`

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
- `report/fable_desk.py`
- `report/asado.py`
- `report/run_daily.sh`

## For future changes

Before changing runtime behavior, verify:

- how the command is expected to run unattended,
- whether a data-quality safeguard is being relied on by the prompt or tests,
- whether a change touches both the live-book and household totals,
- whether output filenames or locations are referenced in docs or automation.
