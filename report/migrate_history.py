#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: migrate_history.py
=============================================================================

INPUT FILES:
    - Step 4 Report Generation/database/market_data.db   (read-only)
    - Phase 2 Portfolio Reports/database/portfolio.db    (read-only)

OUTPUT FILES:
    - data/report.db
      Seeded with portfolio history (portfolio_summary) and prior report
      executive summaries (reports) for day-one continuity.
    - data_backups/<timestamp>/
      Byte-for-byte backups of BOTH legacy databases, taken before reading.

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    One-time migration. Legacy ASSET PRICES are deliberately NOT migrated:
    legacy daily_prices are Bloomberg index levels while the new system
    tracks ETF prices - mixing the two scales across the cutover boundary
    would corrupt every return computed over it. Asset history is instead
    backfilled by a 1-year batched yfinance fetch on first run.

    What IS migrated:
      1. portfolio_summary rows from portfolio.db (portfolio-level history)
      2. report executive summaries from BOTH legacy report archives
         (market reports + portfolio reports) for prompt continuity.

DEPENDENCIES:
    - pandas

USAGE:
    python report/migrate_history.py
=============================================================================
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, ROOT_DIR, ensure_dirs
import db


def backup_legacy_dbs() -> Path:
    """Back up both legacy databases before touching anything (project rule)."""
    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    backup_dir = ROOT_DIR / "data_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for key in ("legacy_market_db", "legacy_portfolio_db"):
        src = PATHS[key]
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
            print(f"  Backed up {src.name} -> {backup_dir}")
        else:
            print(f"  WARNING: {src} not found, skipping backup")
    return backup_dir


def migrate_portfolio_history() -> int:
    """Copy portfolio_summary history from portfolio.db."""
    src = PATHS["legacy_portfolio_db"]
    if not src.exists():
        print("  No legacy portfolio.db - skipping portfolio history")
        return 0

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    legacy = pd.read_sql_query("""
        SELECT date, total_market_value, gross_exposure, net_exposure,
               total_long_value, total_short_value, holding_count,
               portfolio_return_1d, portfolio_return_ytd, total_open_pnl
        FROM portfolio_summary
        WHERE portfolio_id = 'LIVE'
        ORDER BY date
    """, conn)
    conn.close()

    n = 0
    with db.connect() as new_conn:
        for _, r in legacy.iterrows():
            new_conn.execute("""
                INSERT INTO portfolio_summary
                    (date, total_value, gross_exposure, net_exposure, long_value,
                     short_value, n_positions, return_1d, return_ytd,
                     expected_return_1d, alpha_1d, total_open_pnl,
                     holdings_stale, holdings_as_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 0, NULL)
                ON CONFLICT(date) DO NOTHING
            """, (r["date"], r["total_market_value"], r["gross_exposure"],
                  r["net_exposure"], r["total_long_value"], r["total_short_value"],
                  r["holding_count"], r["portfolio_return_1d"],
                  r["portfolio_return_ytd"], r["total_open_pnl"]))
            n += 1
    return n


def _first_chars(text: str, limit: int = 700) -> str:
    """Compact a legacy report into a continuity summary snippet."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned[:limit]


def migrate_report_summaries() -> int:
    """Seed the reports archive with summaries from both legacy systems."""
    n = 0
    sources = [
        (PATHS["legacy_market_db"],
         "SELECT report_date AS date, content_md FROM reports "
         "WHERE report_type = 'daily' ORDER BY report_date"),
        (PATHS["legacy_portfolio_db"],
         "SELECT report_date AS date, content_md FROM portfolio_reports "
         "ORDER BY report_date"),
    ]
    merged: dict[str, str] = {}
    for src, query in sources:
        if not src.exists():
            continue
        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            legacy = pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"  WARNING: could not read reports from {src.name}: {e}")
            legacy = pd.DataFrame()
        finally:
            conn.close()
        for _, r in legacy.iterrows():
            snippet = _first_chars(r["content_md"])
            if not snippet:
                continue
            # Merge market + portfolio summaries for the same date
            existing = merged.get(r["date"], "")
            merged[r["date"]] = (existing + " || " + snippet) if existing else snippet

    with db.connect() as conn:
        for date, summary in sorted(merged.items()):
            conn.execute("""
                INSERT INTO reports (date, executive_summary, content_md, model)
                VALUES (?, ?, NULL, 'legacy-migration')
                ON CONFLICT(date) DO NOTHING
            """, (date, summary[:1500]))
            n += 1
    return n


def main():
    print("=" * 70)
    print("MIGRATE LEGACY HISTORY -> report.db")
    print("=" * 70)

    ensure_dirs()

    print("\n[1/4] Backing up legacy databases (read-only thereafter)...")
    backup_legacy_dbs()

    print("\n[2/4] Initializing report.db schema...")
    db.init_schema()

    print("\n[3/4] Migrating portfolio history...")
    n_port = migrate_portfolio_history()
    print(f"  Migrated {n_port} portfolio_summary rows")

    print("\n[4/4] Migrating report summaries for continuity...")
    n_rep = migrate_report_summaries()
    print(f"  Seeded {n_rep} report summaries")

    print("\nDone. Asset price history will be backfilled by the first run's "
          "1-year yfinance fetch.")


if __name__ == "__main__":
    main()
