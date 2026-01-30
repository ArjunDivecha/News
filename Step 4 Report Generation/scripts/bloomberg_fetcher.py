#!/usr/bin/env python3
"""
=============================================================================
BLOOMBERG DATA FETCHER (Run on Windows/Parallels)
=============================================================================

INPUT FILES:
- database/market_data.db (reads ticker list from assets table)
  OR
- tickers.txt (fallback list of tickers, one per line)

OUTPUT FILES:
- bloomberg_data_YYYY-MM-DD.csv (written to shared Dropbox folder)

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Pull daily closing prices and returns from Bloomberg DAPI for all assets
in the database. This script runs on Windows (Parallels) where Bloomberg
Terminal is installed, and writes output to shared Dropbox folder.

USAGE (on Windows):
    python bloomberg_fetcher.py                    # Today's data
    python bloomberg_fetcher.py --date 2026-01-30  # Specific date
    python bloomberg_fetcher.py --test             # Test mode (no Bloomberg)

SETUP:
    1. Ensure Bloomberg Terminal is running and logged in
    2. Install blpapi: pip install blpapi
    3. Run this script from the shared Dropbox folder

=============================================================================
"""

import os
import sys
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

# Try to import blpapi (Bloomberg API)
try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    print("WARNING: blpapi not installed. Run: pip install blpapi")

# Try to import sqlite3 for reading tickers
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "database" / "market_data.db"
OUTPUT_DIR = PROJECT_DIR / "data"  # Shared Dropbox folder

# Bloomberg fields to fetch
PERFORMANCE_FIELDS = [
    'PX_LAST',           # Last price
    'PX_OPEN',           # Open price
    'PX_HIGH',           # High price
    'PX_LOW',            # Low price
    'CHG_PCT_1D',        # 1-day return %
    'CHG_PCT_1WK',       # 1-week return %
    'CHG_PCT_1M',        # 1-month return %
    'CHG_PCT_YTD',       # YTD return %
    'CHG_PCT_1YR',       # 1-year return %
    'VOLUME',            # Trading volume
    'VOLATILITY_30D',    # 30-day volatility
    'VOLATILITY_60D',    # 60-day volatility
]

# Factor tickers for beta attribution
FACTOR_TICKERS = {
    'SPX': 'SPX Index',
    'Russell2000': 'RTY Index',
    'Nasdaq100': 'NDX Index',
    'Value': 'IWD US Equity',
    'Growth': 'IWF US Equity',
    'EAFE': 'EFA US Equity',
    'EM': 'EEM US Equity',
    'HYCredit': 'HYG US Equity',
    'Treasuries': 'TLT US Equity',
    'TIPS': 'TIP US Equity',
    'Commodities': 'DJP US Equity',
    'Agriculture': 'DBA US Equity',
    'Crypto': 'BITO US Equity',
    'REIT_US': 'VNQ US Equity',
    'REIT_Global': 'VNQI US Equity',
}


def get_tickers_from_db() -> List[str]:
    """Read ticker list from SQLite database."""
    if not SQLITE_AVAILABLE:
        return []
    
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM assets ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers
    except Exception as e:
        print(f"Error reading database: {e}")
        return []


def get_tickers_from_file() -> List[str]:
    """Read ticker list from fallback text file."""
    ticker_file = SCRIPT_DIR / "tickers.txt"
    if ticker_file.exists():
        with open(ticker_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def get_tickers() -> List[str]:
    """Get ticker list from database or fallback file."""
    tickers = get_tickers_from_db()
    if not tickers:
        tickers = get_tickers_from_file()
    return tickers


def create_bloomberg_session() -> Optional[object]:
    """Create and start a Bloomberg API session."""
    if not BLPAPI_AVAILABLE:
        return None
    
    try:
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost("localhost")
        sessionOptions.setServerPort(8194)
        
        session = blpapi.Session(sessionOptions)
        
        if not session.start():
            print("ERROR: Failed to start Bloomberg session")
            return None
        
        if not session.openService("//blp/refdata"):
            print("ERROR: Failed to open //blp/refdata service")
            session.stop()
            return None
        
        print("Bloomberg session started successfully")
        return session
        
    except Exception as e:
        print(f"ERROR creating Bloomberg session: {e}")
        return None


def fetch_bloomberg_data(session, tickers: List[str], fields: List[str]) -> Dict[str, Dict]:
    """
    Fetch data from Bloomberg for given tickers and fields.
    
    Returns:
        Dict mapping ticker to dict of field values
    """
    if not BLPAPI_AVAILABLE or session is None:
        return {}
    
    results = {}
    refDataService = session.getService("//blp/refdata")
    
    # Process in batches of 100 (Bloomberg limit)
    batch_size = 100
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        print(f"  Fetching batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers)...")
        
        try:
            request = refDataService.createRequest("ReferenceDataRequest")
            
            for ticker in batch_tickers:
                request.append("securities", ticker)
            
            for field in fields:
                request.append("fields", field)
            
            session.sendRequest(request)
            
            # Process response
            while True:
                event = session.nextEvent(5000)  # 5 second timeout
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        securityDataArray = msg.getElement("securityData")
                        
                        for i in range(securityDataArray.numValues()):
                            securityData = securityDataArray.getValueAsElement(i)
                            ticker = securityData.getElementAsString("security")
                            
                            fieldData = securityData.getElement("fieldData")
                            
                            results[ticker] = {}
                            for field in fields:
                                try:
                                    if fieldData.hasElement(field):
                                        value = fieldData.getElement(field).getValue()
                                        results[ticker][field] = value
                                    else:
                                        results[ticker][field] = None
                                except:
                                    results[ticker][field] = None
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
                    
        except Exception as e:
            print(f"  ERROR in batch {batch_num + 1}: {e}")
    
    return results


def generate_mock_data(tickers: List[str]) -> Dict[str, Dict]:
    """Generate mock data for testing without Bloomberg."""
    import random
    
    print("  Generating mock data (test mode)...")
    
    results = {}
    for ticker in tickers:
        base_price = random.uniform(10, 500)
        daily_return = random.gauss(0, 1.5)
        
        results[ticker] = {
            'PX_LAST': round(base_price, 2),
            'PX_OPEN': round(base_price * (1 - daily_return/100), 2),
            'PX_HIGH': round(base_price * (1 + abs(daily_return)/100 + 0.5), 2),
            'PX_LOW': round(base_price * (1 - abs(daily_return)/100 - 0.3), 2),
            'CHG_PCT_1D': round(daily_return, 4),
            'CHG_PCT_1WK': round(daily_return * random.uniform(2, 5), 4),
            'CHG_PCT_1M': round(daily_return * random.uniform(8, 15), 4),
            'CHG_PCT_YTD': round(random.gauss(5, 15), 4),
            'CHG_PCT_1YR': round(random.gauss(8, 20), 4),
            'VOLUME': random.randint(100000, 50000000),
            'VOLATILITY_30D': round(random.uniform(10, 40), 2),
            'VOLATILITY_60D': round(random.uniform(10, 40), 2),
        }
    
    return results


def write_csv(data: Dict[str, Dict], date: str, output_dir: Path) -> str:
    """Write Bloomberg data to CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"bloomberg_data_{date}.csv"
    filepath = output_dir / filename
    
    # Get all unique fields
    all_fields = set()
    for ticker_data in data.values():
        all_fields.update(ticker_data.keys())
    
    fields = ['ticker'] + sorted(list(all_fields))
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for ticker, ticker_data in sorted(data.items()):
            row = {'ticker': ticker}
            row.update(ticker_data)
            writer.writerow(row)
    
    return str(filepath)


def fetch_factor_returns(session, date: str, test_mode: bool = False) -> Dict[str, float]:
    """Fetch factor returns for the given date."""
    if test_mode:
        import random
        return {name: round(random.gauss(0, 1.5), 4) for name in FACTOR_TICKERS.keys()}
    
    if session is None:
        return {}
    
    factor_tickers = list(FACTOR_TICKERS.values())
    data = fetch_bloomberg_data(session, factor_tickers, ['CHG_PCT_1D'])
    
    results = {}
    for name, ticker in FACTOR_TICKERS.items():
        if ticker in data and data[ticker].get('CHG_PCT_1D') is not None:
            results[name] = data[ticker]['CHG_PCT_1D']
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data for assets")
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help="Date to fetch (YYYY-MM-DD)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (generate mock data, no Bloomberg)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for CSV (default: data/ in project)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BLOOMBERG DATA FETCHER")
    print("=" * 70)
    print(f"\nDate: {args.date}")
    print(f"Test mode: {args.test}")
    print(f"blpapi available: {BLPAPI_AVAILABLE}")
    
    # Get tickers
    print("\n[1/4] Loading ticker list...")
    tickers = get_tickers()
    if not tickers:
        print("ERROR: No tickers found. Check database or create tickers.txt")
        return 1
    print(f"      Loaded {len(tickers)} tickers")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    print(f"      Output directory: {output_dir}")
    
    # Create Bloomberg session or use test mode
    session = None
    if not args.test:
        print("\n[2/4] Connecting to Bloomberg...")
        if not BLPAPI_AVAILABLE:
            print("      ERROR: blpapi not available. Use --test for mock data.")
            return 1
        session = create_bloomberg_session()
        if session is None:
            print("      ERROR: Could not connect to Bloomberg. Is Terminal running?")
            return 1
    else:
        print("\n[2/4] Test mode - skipping Bloomberg connection")
    
    # Fetch asset data
    print("\n[3/4] Fetching asset data...")
    if args.test:
        data = generate_mock_data(tickers)
    else:
        data = fetch_bloomberg_data(session, tickers, PERFORMANCE_FIELDS)
    print(f"      Fetched data for {len(data)} tickers")
    
    # Fetch factor returns
    print("\n[4/4] Fetching factor returns...")
    factor_returns = fetch_factor_returns(session, args.date, args.test)
    print(f"      Fetched {len(factor_returns)} factor returns")
    
    # Close session
    if session:
        session.stop()
        print("      Bloomberg session closed")
    
    # Write output
    print("\n" + "-" * 70)
    print("WRITING OUTPUT FILES")
    print("-" * 70)
    
    # Write asset data
    csv_path = write_csv(data, args.date, output_dir)
    print(f"\nAsset data: {csv_path}")
    print(f"  Rows: {len(data)}")
    
    # Write factor returns
    factor_path = output_dir / f"factor_returns_{args.date}.csv"
    with open(factor_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['factor_name', 'return_1d'])
        for name, ret in factor_returns.items():
            writer.writerow([name, ret])
    print(f"\nFactor returns: {factor_path}")
    print(f"  Factors: {len(factor_returns)}")
    
    print("\n" + "=" * 70)
    print("BLOOMBERG FETCH COMPLETE")
    print("=" * 70)
    print(f"\nFiles written to: {output_dir}")
    print(f"  - bloomberg_data_{args.date}.csv ({len(data)} assets)")
    print(f"  - factor_returns_{args.date}.csv ({len(factor_returns)} factors)")
    print("\nNext step: Run 04_load_bloomberg.py on Mac to load into database")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
