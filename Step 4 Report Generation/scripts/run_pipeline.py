#!/usr/bin/env python3
"""
=============================================================================
UNIFIED PIPELINE RUNNER
=============================================================================

INPUT FILES:
- database/market_data.db (populated by bloomberg_backfill.py and bloomberg_daily.py)

OUTPUT FILES:
- outputs/daily/daily_wrap_YYYY-MM-DD_anthropic.md
- outputs/daily/daily_wrap_YYYY-MM-DD_anthropic.pdf

VERSION: 1.0.0
CREATED: 2026-01-31

PURPOSE:
Single entry point for running the daily report generation pipeline.
This script is designed to run on the Mac side AFTER Bloomberg data
has been fetched on the PC side.

WORKFLOW:
1. PC Side (Windows/Parallels):
   - Run bloomberg_backfill.py (one-time, ~90 days of history)
   - Run bloomberg_daily.py (daily, after market close)
   
2. Mac Side (this script):
   - Run run_pipeline.py to generate the report

USAGE:
    python3 scripts/run_pipeline.py                    # Latest trading day
    python3 scripts/run_pipeline.py --date 2026-01-31  # Specific date
    python3 scripts/run_pipeline.py --check            # Check data status only
    python3 scripts/run_pipeline.py --list-dates       # List all available dates

=============================================================================
"""

import sqlite3
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

# Minimum days of history required for full analysis
MIN_HISTORY_DAYS = 5
RECOMMENDED_HISTORY_DAYS = 60

# =============================================================================
# DATABASE CHECKS
# =============================================================================

def get_db() -> sqlite3.Connection:
    """Get database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}\nRun init_db.py first.")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def check_database_status() -> dict:
    """
    Check the current state of the database.
    
    Returns:
        Dict with status information
    """
    conn = get_db()
    cursor = conn.cursor()
    
    status = {}
    
    # Assets count
    cursor.execute("SELECT COUNT(*) FROM assets")
    status['assets_count'] = cursor.fetchone()[0]
    
    # Daily prices date range
    cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices")
    row = cursor.fetchone()
    status['prices_min_date'] = row[0]
    status['prices_max_date'] = row[1]
    status['prices_trading_days'] = row[2]
    
    # Factor returns date range
    cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM factor_returns")
    row = cursor.fetchone()
    status['factors_min_date'] = row[0]
    status['factors_max_date'] = row[1]
    status['factors_trading_days'] = row[2]
    
    # Category stats
    cursor.execute("SELECT COUNT(DISTINCT date) FROM category_stats")
    status['category_stats_days'] = cursor.fetchone()[0]
    
    # Correlations
    cursor.execute("SELECT COUNT(DISTINCT date) FROM asset_correlations")
    status['correlations_days'] = cursor.fetchone()[0]
    
    # Check for data gaps
    cursor.execute("""
        SELECT date FROM daily_prices 
        GROUP BY date 
        HAVING COUNT(*) < 100
        ORDER BY date
    """)
    status['sparse_dates'] = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    return status


def get_available_dates() -> List[str]:
    """Get list of dates with complete data."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, COUNT(*) as cnt 
        FROM daily_prices 
        GROUP BY date 
        HAVING cnt >= 100
        ORDER BY date DESC
    """)
    
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return dates


def find_best_date(target_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Find the best date to use for report generation.
    
    Args:
        target_date: Requested date (YYYY-MM-DD) or None for latest
        
    Returns:
        Tuple of (selected_date, reason)
    """
    available = get_available_dates()
    
    if not available:
        raise ValueError("No data available in database. Run bloomberg_backfill.py first.")
    
    if target_date is None:
        # Use latest available
        return available[0], "Using latest available data"
    
    # Check if target date is available
    if target_date in available:
        return target_date, "Requested date available"
    
    # Target date not available - find closest
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Check if weekend
    if target_dt.weekday() >= 5:
        # Walk back to Friday
        days_back = target_dt.weekday() - 4
        friday = target_dt - timedelta(days=days_back)
        friday_str = friday.strftime('%Y-%m-%d')
        if friday_str in available:
            return friday_str, f"Weekend detected, using Friday {friday_str}"
    
    # Find closest earlier date
    for date in available:
        if date <= target_date:
            return date, f"Requested {target_date} not available, using {date}"
    
    # Fall back to latest
    return available[0], f"No data for {target_date}, using latest: {available[0]}"


def check_data_quality(date: str) -> dict:
    """
    Check data quality for a specific date.
    
    Returns:
        Dict with quality metrics and warnings
    """
    conn = get_db()
    cursor = conn.cursor()
    
    quality = {'date': date, 'warnings': [], 'ready': True}
    
    # Check asset count for the date
    cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE date = ?", (date,))
    quality['asset_count'] = cursor.fetchone()[0]
    
    if quality['asset_count'] < 100:
        quality['warnings'].append(f"Low asset count: {quality['asset_count']}")
        quality['ready'] = False
    
    # Check for return data
    cursor.execute("""
        SELECT COUNT(*) FROM daily_prices 
        WHERE date = ? AND return_1d IS NOT NULL
    """, (date,))
    quality['return_count'] = cursor.fetchone()[0]
    
    if quality['return_count'] < quality['asset_count'] * 0.5:
        quality['warnings'].append(f"Missing return data: only {quality['return_count']}/{quality['asset_count']}")
    
    # Check factor returns
    cursor.execute("SELECT COUNT(*) FROM factor_returns WHERE date = ?", (date,))
    quality['factor_count'] = cursor.fetchone()[0]
    
    if quality['factor_count'] < 10:
        quality['warnings'].append(f"Low factor count: {quality['factor_count']}")
    
    # Check category stats
    cursor.execute("SELECT COUNT(*) FROM category_stats WHERE date = ?", (date,))
    quality['category_stats_count'] = cursor.fetchone()[0]
    
    if quality['category_stats_count'] == 0:
        quality['warnings'].append("No category stats computed for this date")
    
    # Check history depth (for percentiles, correlations)
    cursor.execute("""
        SELECT COUNT(DISTINCT date) FROM daily_prices WHERE date < ?
    """, (date,))
    quality['history_days'] = cursor.fetchone()[0]
    
    if quality['history_days'] < MIN_HISTORY_DAYS:
        quality['warnings'].append(f"Limited history: {quality['history_days']} days (min {MIN_HISTORY_DAYS})")
    elif quality['history_days'] < RECOMMENDED_HISTORY_DAYS:
        quality['warnings'].append(f"Partial history: {quality['history_days']} days (recommended {RECOMMENDED_HISTORY_DAYS})")
    
    # Check correlations
    cursor.execute("SELECT COUNT(*) FROM asset_correlations WHERE date = ?", (date,))
    quality['correlation_count'] = cursor.fetchone()[0]
    
    if quality['correlation_count'] == 0 and quality['history_days'] >= RECOMMENDED_HISTORY_DAYS:
        quality['warnings'].append("Correlations not computed (run 04_compute_correlations.py)")
    
    conn.close()
    
    return quality


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_correlations(date: str) -> bool:
    """Run correlation computation for a date."""
    import subprocess
    
    script_path = SCRIPT_DIR / "04_compute_correlations.py"
    if not script_path.exists():
        print(f"  WARNING: Correlation script not found: {script_path}")
        return False
    
    result = subprocess.run(
        ["python3", str(script_path), "--date", date],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  WARNING: Correlation computation failed: {result.stderr}")
        return False
    
    return True


def run_report_generation(date: str) -> bool:
    """Run report generation for a date."""
    import subprocess
    
    script_path = SCRIPT_DIR / "03_generate_daily_report.py"
    if not script_path.exists():
        print(f"  ERROR: Report script not found: {script_path}")
        return False
    
    result = subprocess.run(
        ["python3", str(script_path), "--date", date],
        cwd=str(PROJECT_DIR),
        capture_output=False  # Stream output to console
    )
    
    return result.returncode == 0


def run_pipeline(target_date: Optional[str] = None, skip_checks: bool = False) -> bool:
    """
    Run the complete pipeline for a date.
    
    Args:
        target_date: Date to generate report for (YYYY-MM-DD)
        skip_checks: Skip data quality checks
        
    Returns:
        True if successful
    """
    print("=" * 70)
    print("NEWS FROM DATA - DAILY REPORT PIPELINE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Step 1: Find best date
    print("\n[1/5] Selecting report date...")
    try:
        report_date, reason = find_best_date(target_date)
        print(f"      Date: {report_date}")
        print(f"      Reason: {reason}")
    except ValueError as e:
        print(f"      ERROR: {e}")
        return False
    
    # Step 2: Check database status
    print("\n[2/5] Checking database status...")
    try:
        status = check_database_status()
        print(f"      Assets: {status['assets_count']}")
        print(f"      Price data: {status['prices_min_date']} to {status['prices_max_date']} ({status['prices_trading_days']} days)")
        print(f"      Factor returns: {status['factors_trading_days']} days")
        print(f"      Category stats: {status['category_stats_days']} days")
        print(f"      Correlations: {status['correlations_days']} days")
    except Exception as e:
        print(f"      ERROR: {e}")
        return False
    
    # Step 3: Check data quality for selected date
    print("\n[3/5] Checking data quality for {report_date}...")
    quality = check_data_quality(report_date)
    print(f"      Assets with data: {quality['asset_count']}")
    print(f"      Assets with returns: {quality['return_count']}")
    print(f"      Factor returns: {quality['factor_count']}")
    print(f"      Category stats: {quality['category_stats_count']}")
    print(f"      Historical depth: {quality['history_days']} days")
    print(f"      Correlations: {quality['correlation_count']}")
    
    if quality['warnings']:
        print("\n      Warnings:")
        for warning in quality['warnings']:
            print(f"        - {warning}")
    
    if not quality['ready'] and not skip_checks:
        print("\n      ERROR: Data quality checks failed. Use --skip-checks to override.")
        return False
    
    # Step 4: Run correlations if needed and possible
    print("\n[4/5] Computing correlations...")
    if quality['history_days'] >= RECOMMENDED_HISTORY_DAYS and quality['correlation_count'] == 0:
        success = run_correlations(report_date)
        if success:
            print("      Correlations computed successfully")
        else:
            print("      Correlation computation skipped/failed (non-fatal)")
    elif quality['history_days'] < RECOMMENDED_HISTORY_DAYS:
        print(f"      Skipped: Need {RECOMMENDED_HISTORY_DAYS} days of history, have {quality['history_days']}")
    else:
        print("      Already computed")
    
    # Step 5: Generate report
    print("\n[5/5] Generating daily report...")
    success = run_report_generation(report_date)
    
    if success:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nReport generated for: {report_date}")
        print(f"Output directory: {PROJECT_DIR / 'outputs' / 'daily'}")
        return True
    else:
        print("\n      ERROR: Report generation failed")
        return False


# =============================================================================
# CLI
# =============================================================================

def print_status():
    """Print detailed database status."""
    print("=" * 70)
    print("DATABASE STATUS")
    print("=" * 70)
    
    try:
        status = check_database_status()
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    print(f"\nAssets: {status['assets_count']}")
    print(f"\nDaily Prices:")
    print(f"  Date range: {status['prices_min_date']} to {status['prices_max_date']}")
    print(f"  Trading days: {status['prices_trading_days']}")
    
    print(f"\nFactor Returns:")
    print(f"  Date range: {status['factors_min_date']} to {status['factors_max_date']}")
    print(f"  Trading days: {status['factors_trading_days']}")
    
    print(f"\nCategory Stats: {status['category_stats_days']} days")
    print(f"Correlations: {status['correlations_days']} days")
    
    if status['sparse_dates']:
        print(f"\nSparse dates (< 100 assets): {len(status['sparse_dates'])}")
        for d in status['sparse_dates'][:5]:
            print(f"  - {d}")
    
    # Readiness assessment
    print("\n" + "-" * 70)
    print("READINESS ASSESSMENT")
    print("-" * 70)
    
    if status['assets_count'] == 0:
        print("\n❌ Assets not loaded. Run 01_sync_static_data.py")
    elif status['assets_count'] < 900:
        print(f"\n⚠ Only {status['assets_count']} assets. Expected ~970")
    else:
        print(f"\n✓ Assets: {status['assets_count']}")
    
    if status['prices_trading_days'] == 0:
        print("❌ No price data. Run bloomberg_backfill.py on PC")
    elif status['prices_trading_days'] < 60:
        print(f"⚠ Only {status['prices_trading_days']} days of prices. Run bloomberg_backfill.py for full history")
    else:
        print(f"✓ Prices: {status['prices_trading_days']} trading days")
    
    if status['factors_trading_days'] == 0:
        print("❌ No factor returns. Run bloomberg_backfill.py on PC")
    else:
        print(f"✓ Factor returns: {status['factors_trading_days']} days")
    
    if status['category_stats_days'] < status['prices_trading_days']:
        print(f"⚠ Category stats incomplete: {status['category_stats_days']}/{status['prices_trading_days']} days")
    else:
        print(f"✓ Category stats: {status['category_stats_days']} days")
    
    if status['correlations_days'] == 0 and status['prices_trading_days'] >= 60:
        print("⚠ No correlations computed. Run 04_compute_correlations.py --backfill")
    elif status['correlations_days'] > 0:
        print(f"✓ Correlations: {status['correlations_days']} days")


def print_available_dates():
    """Print list of available dates."""
    print("=" * 70)
    print("AVAILABLE DATES")
    print("=" * 70)
    
    dates = get_available_dates()
    
    if not dates:
        print("\nNo data available. Run bloomberg_backfill.py on PC first.")
        return
    
    print(f"\nTotal: {len(dates)} trading days\n")
    
    # Group by month
    from collections import defaultdict
    by_month = defaultdict(list)
    for d in dates:
        month = d[:7]  # YYYY-MM
        by_month[month].append(d)
    
    for month in sorted(by_month.keys(), reverse=True):
        days = by_month[month]
        print(f"{month}: {len(days)} days ({days[-1]} to {days[0]})")


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for daily report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py                    # Generate report for latest date
  python3 run_pipeline.py --date 2026-01-31  # Specific date
  python3 run_pipeline.py --check            # Check database status
  python3 run_pipeline.py --list-dates       # List available dates
        """
    )
    parser.add_argument("--date", type=str, help="Date to generate report for (YYYY-MM-DD)")
    parser.add_argument("--check", action="store_true", help="Check database status only")
    parser.add_argument("--list-dates", action="store_true", help="List available dates")
    parser.add_argument("--skip-checks", action="store_true", help="Skip data quality checks")
    args = parser.parse_args()
    
    try:
        if args.check:
            print_status()
            return 0
        
        if args.list_dates:
            print_available_dates()
            return 0
        
        success = run_pipeline(target_date=args.date, skip_checks=args.skip_checks)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
