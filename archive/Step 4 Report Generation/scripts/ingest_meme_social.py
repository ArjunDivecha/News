#!/usr/bin/env python3
"""
=============================================================================
MEME / SOCIAL SNAPSHOT INGESTION
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/search_based_meme_stocks.json
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/search_based_meme_stocks.json
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 3 Data Analysis/search_based_meme_stocks.json
  Description: MemeFinder JSON output with timestamp, search queries, and meme stock rows

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/database/market_data.db
  Description: SQLite database updated with meme/social snapshot tables

VERSION HISTORY:
v1.0.0 (2026-03-13): Initial ingestion script for MemeFinder snapshot data

PURPOSE:
Load the latest MemeFinder JSON snapshot into SQLite so daily reports can
query meme/social flow alongside the market data pipeline.
"""

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.db import ingest_meme_social_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest meme/social snapshot JSON into SQLite")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional path to a specific search_based_meme_stocks.json file",
    )
    args = parser.parse_args()

    result = ingest_meme_social_snapshot(args.file)

    print("=" * 70)
    print("MEME / SOCIAL SNAPSHOT INGESTION")
    print("=" * 70)

    status = result.get('status')
    if status == 'missing_file':
        print(f"\nStatus: {status}")
        print(f"Message: {result.get('message')}")
        return 0

    print(f"\nStatus: {status}")
    print(f"Snapshot timestamp: {result.get('snapshot_ts')}")
    print(f"Snapshot date: {result.get('snapshot_date')}")
    print(f"Source file: {result.get('source_file')}")
    print(f"Rows loaded: {result.get('rows_loaded')}")
    print(f"Universe matches: {result.get('matched_count')}")
    print(f"Outside universe: {result.get('unmatched_count')}")
    print(f"Total found: {result.get('total_found')}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
