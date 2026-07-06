# OpenWiki Quickstart

This repository contains a unified daily market and portfolio reporting system plus the upstream universe-construction work that feeds it. The active day-to-day product lives in `report/`: it pulls Schwab and IBKR holdings, downloads ETF prices from Yahoo Finance, runs portfolio and market analytics, optionally computes tier-3 tag views and scenario risk, asks Claude to write the report, and renders the result to PDF.

Start here if you are new to the repo:

- [Unified report architecture](architecture/report-system.md)
- [Market, portfolio, and bridge analytics](analytics/market-portfolio.md)
- [Tag views and household allocation](domain/tag-views-and-allocation.md)
- [Scenario risk engine](domain/scenario-risk.md)
- [Operations and runbook](operations/runbook.md)

## What this repository is for

At a high level, the codebase has two major responsibilities:

1. **Daily report generation** — the `report/` package is the current production path.
2. **Universe construction and classification** — the legacy Step 1-3 / fine-tuning work remains in the repo as an upstream data-preparation pipeline and supporting history.

The root `README.md` describes the split clearly: `report/` is the daily report system, while `Step 1 Data Collection/`, `Step 2 Data Processing - Final1000/`, `Step 3 Data Analysis/`, and `fine tuning/` build and refine the asset universe that the report consumes.

## Current report flow

The current report pipeline, as implemented in `report/main.py`, is:

1. Fetch live holdings from Schwab and IBKR.
2. Load the universe from `data/universe.xlsx` and fetch prices from Yahoo Finance.
3. Run market, portfolio, bridge, tag, scenario, and policy-check analytics.
4. Build the LLM data package and system prompt (including the Action Box mandate and Policy Check section when available).
5. Ask Claude to write the report text.
5b. Optionally run Fable's Desk — a second Claude pass with live web search and ASADO country signals, producing 0–4 judgment-based ideas appended after Bottom Line. Gated by `REPORT_ENABLE_FABLE_DESK`; additive-only.
6. Render Markdown to HTML and PDF, archive results in `data/report.db`, and keep continuity through stored summaries.

Key runtime guarantees come from the source and tests:

- Missing benchmark data fails loudly rather than producing a benchmark-less report.
- Sparse holiday rows are filtered before analytics so post-holiday returns do not collapse to `n/a`.
- Stale holdings are explicitly labeled rather than silently dropped.
- The report never emits literal `n/a`; undefined cells are rendered as em dashes.
- Truncated model output is treated as a hard failure.

## Major domains

### Report system
The main production code is under `report/`. Important files include:

- `main.py` — orchestration
- `config.py` — paths, constants, and look-through policy
- `data.py` — Yahoo Finance download and sparse-row filtering
- `holdings.py` — Schwab/IBKR connectivity and fallback logic
- `analytics.py` — market, portfolio, sub-portfolio, and bridge math
- `prompt.py` — builds the LLM data package
- `llm.py` — Claude CLI/API invocation and truncation safeguards
- `pdf.py` — Markdown to PDF rendering
- `tag_analytics.py` and `scenarios.py` — additive report sections
- `tags.py` — dynamic canonical tag resolution
- `fable_desk.py` — second LLM pass (Fable's Desk) with live web search and ASADO snapshot
- `asado.py` — read-only ASADO DuckDB snapshot (34-country value/momentum/FX/risk signals, GDELT news pulse, factor P&L) for Fable's Desk

### Universe construction
The upstream universe pipeline is still documented in the repo root README and `AGENTS.md`. It is the source of the `data/universe.xlsx` file that the report consumes. The historical steps remain relevant when changing tagging, classification, or source data.

### Tests
The current safety net is in `tests/`.

- `tests/test_analytics.py` covers core financial math and policy check.
- `tests/test_report_pipeline.py` protects report-package and bridge behavior.
- `tests/test_tag_analytics.py` protects tier-3 tag and allocation logic.
- `tests/test_scenarios.py` protects the stress engine.
- `tests/test_fable_desk.py` protects desk-json trailer parsing, desk idea persistence/grading, and policy-check math.

## Suggested reading order for contributors

1. Read this page.
2. Read [Unified report architecture](architecture/report-system.md).
3. Read [Market, portfolio, and bridge analytics](analytics/market-portfolio.md).
4. Read [Tag views and household allocation](domain/tag-views-and-allocation.md) if you are touching positioning or allocation views.
5. Read [Scenario risk engine](domain/scenario-risk.md) if you are touching stress tests or liquidity risk.
6. Read [Operations and runbook](operations/runbook.md) before changing runtime, auth, scheduling, or PDF behavior.

## Source evidence worth trusting

These were the highest-signal sources used to build the wiki:

- `README.md` and `report/README.md` for the repo-level and package-level narratives.
- `report/main.py`, `report/config.py`, `report/analytics.py`, `report/prompt.py`, `report/scenarios.py`, `report/tag_analytics.py`, and `report/tags.py` for actual behavior.
- `report/prompts/system.md` for the report-writing contract.
- `tests/` for the edge cases that must keep working.
- Recent git history, especially the July 2026 changes that introduced tag views, look-through allocation, manual sleeves, scenario risk, and the Fable writer upgrade.

## Notes for future agents

- The document set is intentionally small and focused. If you need deeper detail, prefer reading the source and tests linked from the section pages rather than expanding the wiki prematurely.
- `openwiki/_plan.md` is temporary and should not be kept.
- The root-level agent instructions should point at this quickstart; check `AGENTS.md` and `CLAUDE.md` if present.
