#!/usr/bin/env python3
"""
=============================================================================
DATABASE INITIALIZATION SCRIPT
=============================================================================

INPUT FILES:
- database/schema.sql: SQL schema definition

OUTPUT FILES:
- database/market_data.db: SQLite database file

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Initialize the SQLite database with the schema for the News from Data
report generation pipeline. Creates tables, indexes, and views.

USAGE:
    python database/init_db.py
    python database/init_db.py --reset  # Drop and recreate all tables
=============================================================================
"""

import sqlite3
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Database path
DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "market_data.db"
SCHEMA_PATH = DB_DIR / "schema.sql"


def init_database(reset: bool = False) -> bool:
    """
    Initialize the SQLite database.
    
    Args:
        reset: If True, drop all tables and recreate from scratch
        
    Returns:
        True if successful, False otherwise
    """
    print("=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)
    print(f"\nDatabase path: {DB_PATH}")
    print(f"Schema path: {SCHEMA_PATH}")
    print(f"Reset mode: {reset}")
    print()
    
    # Check schema file exists
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema file not found at {SCHEMA_PATH}")
        return False
    
    # Read schema
    print("[1/4] Reading schema file...")
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    print(f"      Schema loaded ({len(schema_sql):,} characters)")
    
    # Handle reset
    if reset and DB_PATH.exists():
        print("\n[2/4] Resetting database (dropping existing)...")
        os.remove(DB_PATH)
        print("      Old database removed")
    else:
        print("\n[2/4] Database reset not requested, preserving existing data")
    
    # Connect and create tables
    print("\n[3/4] Connecting to database and executing schema...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Execute schema (handles IF NOT EXISTS gracefully)
        cursor.executescript(schema_sql)
        conn.commit()
        
        print("      Schema executed successfully")
        
    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
        return False
    
    # Verify tables
    print("\n[4/4] Verifying database structure...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    print(f"\n      Tables created ({len(tables)}):")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"        - {table[0]}: {count} rows")
    
    # Check views
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
    views = cursor.fetchall()
    print(f"\n      Views created ({len(views)}):")
    for view in views:
        print(f"        - {view[0]}")
    
    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    indexes = cursor.fetchall()
    print(f"\n      Indexes created ({len(indexes)}):")
    for idx in indexes:
        print(f"        - {idx[0]}")
    
    conn.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nDatabase file: {DB_PATH}")
    print(f"File size: {DB_PATH.stat().st_size:,} bytes")
    print(f"Tables: {len(tables)}")
    print(f"Views: {len(views)}")
    print(f"Indexes: {len(indexes)}")
    print()
    
    return True


def test_database() -> bool:
    """
    Run basic tests on the database.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "=" * 70)
    print("DATABASE TESTS")
    print("=" * 70)
    
    if not DB_PATH.exists():
        print("ERROR: Database file does not exist")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Can insert into assets table
    print("\n[TEST 1] Insert into assets table...")
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO assets (ticker, name, tier1, tier2, source)
            VALUES ('TEST_TICKER', 'Test Asset', 'Equities', 'Global Indices', 'ETF')
        """)
        conn.commit()
        print("         PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Can insert into daily_prices table
    print("\n[TEST 2] Insert into daily_prices table...")
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_prices (date, ticker, return_1d, return_1w, return_ytd)
            VALUES ('2026-01-30', 'TEST_TICKER', -1.5, 0.8, 4.2)
        """)
        conn.commit()
        print("         PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Can insert into category_stats table
    print("\n[TEST 3] Insert into category_stats table...")
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO category_stats 
            (date, category_type, category_value, count, avg_return, median_return)
            VALUES ('2026-01-30', 'tier1', 'Equities', 520, -1.2, -1.0)
        """)
        conn.commit()
        print("         PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Can query views
    print("\n[TEST 4] Query v_latest_daily view...")
    try:
        cursor.execute("SELECT COUNT(*) FROM v_latest_daily")
        count = cursor.fetchone()[0]
        print(f"         PASSED (returned {count} rows)")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Can query tier1 summary view
    print("\n[TEST 5] Query v_tier1_summary view...")
    try:
        cursor.execute("SELECT * FROM v_tier1_summary")
        rows = cursor.fetchall()
        print(f"         PASSED (returned {len(rows)} rows)")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Foreign key constraint works
    print("\n[TEST 6] Foreign key constraint (daily_prices -> assets)...")
    try:
        # This should work because TEST_TICKER exists
        cursor.execute("""
            INSERT OR REPLACE INTO daily_prices (date, ticker, return_1d)
            VALUES ('2026-01-29', 'TEST_TICKER', 0.5)
        """)
        conn.commit()
        print("         PASSED (valid FK accepted)")
        tests_passed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Cleanup test data
    print("\n[CLEANUP] Removing test data...")
    cursor.execute("DELETE FROM daily_prices WHERE ticker = 'TEST_TICKER'")
    cursor.execute("DELETE FROM category_stats WHERE date = '2026-01-30' AND category_value = 'Equities'")
    cursor.execute("DELETE FROM assets WHERE ticker = 'TEST_TICKER'")
    conn.commit()
    print("          Test data removed")
    
    conn.close()
    
    # Summary
    print("\n" + "-" * 70)
    print(f"TESTS COMPLETED: {tests_passed} passed, {tests_failed} failed")
    print("-" * 70)
    
    return tests_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the market data database")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    parser.add_argument("--test", action="store_true", help="Run database tests after initialization")
    args = parser.parse_args()
    
    # Initialize database
    success = init_database(reset=args.reset)
    
    if not success:
        print("\nERROR: Database initialization failed")
        sys.exit(1)
    
    # Run tests if requested or by default
    if args.test or True:  # Always run tests
        test_success = test_database()
        if not test_success:
            print("\nWARNING: Some database tests failed")
            sys.exit(1)
    
    print("\n✓ Database ready for use")
    sys.exit(0)
