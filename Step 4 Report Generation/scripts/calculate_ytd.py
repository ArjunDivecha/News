#!/usr/bin/env python3
"""
=============================================================================
CALCULATE HISTORICAL YTD RETURNS
=============================================================================

PURPOSE:
Calculate Year-to-Date returns for all historical dates by compounding
daily returns from the start of each year.

INPUT FILES:
- database/market_data.db (daily_prices table with return_1d)

OUTPUT FILES:
- database/market_data.db (updates return_ytd column)

VERSION: 1.0.0
CREATED: 2026-02-01

USAGE:
    python3 scripts/calculate_historical_ytd.py

=============================================================================
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db() -> sqlite3.Connection:
    """Get database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def calculate_ytd_for_ticker(cursor: sqlite3.Cursor, ticker: str) -> Dict[str, float]:
    """
    Calculate YTD returns for a single ticker.
    
    Returns:
        Dict mapping date -> YTD return
    """
    # Get all daily returns for this ticker, ordered by date
    cursor.execute("""
        SELECT date, return_1d
        FROM daily_prices
        WHERE ticker = ? AND return_1d IS NOT NULL
        ORDER BY date
    """, (ticker,))
    
    rows = cursor.fetchall()
    
    if not rows:
        return {}
    
    ytd_values = {}
    current_year = None
    cumulative_return = 1.0
    
    for row in rows:
        date_str = row['date']
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        return_1d = row['return_1d'] / 100.0  # Convert percentage to decimal
        
        # If new year, reset cumulative return
        if current_year is None or year != current_year:
            current_year = year
            cumulative_return = 1.0
        
        # Compound the daily return
        cumulative_return *= (1.0 + return_1d)
        
        # Calculate YTD as percentage
        ytd_pct = (cumulative_return - 1.0) * 100.0
        ytd_values[date_str] = ytd_pct
    
    return ytd_values

def update_ytd_values(conn: sqlite3.Connection, ytd_data: Dict[str, Dict[str, float]]):
    """
    Update return_ytd in database.
    
    Args:
        conn: Database connection
        ytd_data: Dict mapping ticker -> {date -> ytd_value}
    """
    cursor = conn.cursor()
    
    total_updates = 0
    
    for ticker, dates_values in ytd_data.items():
        for date_str, ytd_value in dates_values.items():
            cursor.execute("""
                UPDATE daily_prices
                SET return_ytd = ?
                WHERE ticker = ? AND date = ?
            """, (ytd_value, ticker, date_str))
            total_updates += 1
    
    conn.commit()
    return total_updates

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CALCULATING HISTORICAL YTD RETURNS")
    print("=" * 70)
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get all tickers
    cursor.execute("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    
    print(f"\nProcessing {len(tickers)} tickers...")
    
    ytd_data = {}
    processed = 0
    
    for ticker in tickers:
        ytd_values = calculate_ytd_for_ticker(cursor, ticker)
        if ytd_values:
            ytd_data[ticker] = ytd_values
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(tickers)} tickers...")
    
    print(f"\nCalculated YTD for {len(ytd_data)} tickers")
    print("Updating database...")
    
    total_updates = update_ytd_values(conn, ytd_data)
    
    print(f"\n✓ Updated {total_updates} records with YTD values")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE return_ytd IS NOT NULL")
    ytd_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM daily_prices")
    total_count = cursor.fetchone()[0]
    
    print(f"✓ Total YTD records: {ytd_count}/{total_count} ({ytd_count*100/total_count:.1f}%)")
    
    conn.close()
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
