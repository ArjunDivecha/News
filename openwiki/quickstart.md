---
type: "Reference"
title: "OpenWiki Quickstart"
description: "Entry point for the repository wiki. Covers the daily market-and-portfolio report system, upstream universe construction, analytics, tag views, scenario risk, and operations."
---

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
- `notify.py` — emails the rendered PDF (Mail.app default, SMTP fallback); invoked by `run_daily.sh`, not by the main run loop (see [Operations and runbook](operations/runbook.md))

### Universe construction
The upstream universe pipeline is still documented in the repo root README and `AGENTS.md`. It is the source of the `data/universe.xlsx` file that the report consumes. The historical steps remain relevant when changing tagging, classification, or source data.

### Tests
The current safety net is in `tests/`.

- `tests/test_analytics.py` covers core financial math and policy check.
- `tests/test_report_pipeline.py` protects report-package and bridge behavior.
- `tests/test_tag_analytics.py` protects tier-3 tag and allocation logic.
- `tests/test_scenarios.py` protects the stress engine.
- `tests/test_fable_desk.py` protects desk-json trailer parsing, desk idea persistence/grading, and policy-check math.

## Task routing for changes

Start from the area you intend to change. The table maps each change category to its wiki page, source entry points, important symbols, focused tests, and the narrowest validation command.

| Change area / intent | Wiki page | Source entry points | Important symbols | Focused tests | Minimal validation |
|---|---|---|---|---|---|
| Pipeline orchestration, run stages, or stage ordering | [Unified report architecture](architecture/report-system.md) | `report/main.py` | `run()` | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Holdings fetch, broker fallback, or stale-snapshot handling | [Unified report architecture](architecture/report-system.md) | `report/holdings.py`, `report/config.py` | `MANUAL_HOLDINGS`, Schwab/IBKR fetch paths | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Price download, sparse-row filtering, or coverage gate | [Unified report architecture](architecture/report-system.md) | `report/data.py`, `report/config.py` | `fetch_prices()`, `filter_sparse_rows()`, `DataCoverageError`, `SETTINGS["min_coverage"]` | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Market, portfolio, bridge, or policy-check math | [Market, portfolio, and bridge analytics](analytics/market-portfolio.md) | `report/analytics.py`, `report/config.py` | `compute_market()`, `compute_portfolio()`, `compute_subportfolios()`, `compute_bridge()`, `compute_policy_check()`, `aggregate_holdings()` | `tests/test_analytics.py`, `tests/test_fable_desk.py` (policy check) | `python3 -m pytest tests/test_analytics.py tests/test_fable_desk.py -q` |
| Tag resolution, tier-3 views, or household allocation | [Tag views and household allocation](domain/tag-views-and-allocation.md) | `report/tags.py`, `report/tag_analytics.py`, `report/config.py` | `compute_asset_allocation()`, `compute_tag_tilts()`, manual overrides, `FUND_LOOKTHROUGH` | `tests/test_tag_analytics.py` | `python3 -m pytest tests/test_tag_analytics.py -q` |
| Scenario shocks, crash beta, or liquidity ladder | [Scenario risk engine](domain/scenario-risk.md) | `report/scenarios.py`, `report/config.py`, `report/prompt.py` | `compute_scenario_risk()`, `compute_crash_beta()`, `compute_liquidity_ladder()` | `tests/test_scenarios.py` | `python3 -m pytest tests/test_scenarios.py -q` |
| LLM data package or report-writing contract | [Unified report architecture](architecture/report-system.md) | `report/prompt.py`, `report/prompts/system.md` | `_render_policy_check()`, `_render_scenario_risk()`, `_render_tag_views()` | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Model invocation, truncation guard, or Fable fallback chain | [Unified report architecture](architecture/report-system.md) | `report/llm.py` | streaming fallback, `stop_reason == "refusal"`, Opus 4.8 fallback | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Fable's Desk judgment pass, idea persistence, or grading | [Unified report architecture](architecture/report-system.md) | `report/fable_desk.py`, `report/asado.py`, `report/db.py` | `enable_fable_desk`, `desk_ideas`, ASADO snapshot | `tests/test_fable_desk.py` | `python3 -m pytest tests/test_fable_desk.py -q` |
| PDF rendering, table validation, or output filenames | [Operations and runbook](operations/runbook.md) | `report/pdf.py`, `report/config.py` | table validation, output paths | `tests/test_report_pipeline.py` | `python3 -m pytest tests/test_report_pipeline.py -q` |
| Runtime env, scheduling, email delivery, or run_daily.sh | [Operations and runbook](operations/runbook.md) | `report/notify.py`, `report/run_daily.sh`, `report/config.py` | `send_report()`, `send_via_mail_app()`, `send_via_smtp()`, `REPORT_ENABLE_FABLE_DESK` | none (delivery is non-fatal wrapper) | `python3 report/notify.py --help` |

Whole-suite validation, run only when a change spans categories or before a release:

```bash
python3 -m pytest tests/ -q
```

Data-and-analytics-only smoke (no LLM call, no broker prompts):

```bash
python3 report/main.py --no-llm
```

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
