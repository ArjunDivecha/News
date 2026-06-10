# Archive - Legacy Reporting Chain

**Archived 2026-06-10.** Everything in this folder was replaced by the
unified daily report system in `report/` (one command: `python3 report/main.py`).
Kept for reference only - none of it is run anymore.

| Folder / file | What it was | Replaced by |
|---|---|---|
| `Phase 0 Portfolio Feed/` | Broker holdings sync (Schwab + IBKR) -> `Client.xlsx` | `report/holdings.py` (live pulls every run) |
| `Phase 2 Portfolio Reports/` | Portfolio analytics + LLM portfolio report | `report/analytics.py` + the unified report |
| `Step 4 Report Generation/` | Bloomberg market data + LLM market report | `report/data.py` (Yahoo Finance) + the unified report |
| `runphase0.py` / `runphase1.py` / `runphase2.py` | Pipeline runner scripts | `report/main.py` |
| `Bloomberg Import Data/` | Bloomberg data export (preserved per data-retention policy) | n/a |

The legacy SQLite databases inside (`market_data.db`, `portfolio.db`) were
migrated into `data/report.db` by `report/migrate_history.py`, with backups
in `data_backups/`. The paths in `report/config.py` still point here so the
migration script remains re-runnable.
