#!/usr/bin/env python3
"""
=============================================================================
BLOOMBERG HISTORICAL DATA BACKFILL
=============================================================================

INPUT FILES:
- /Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx
  Contains: 970 Bloomberg tickers with classifications

OUTPUT FILES:
- database/market_data.db (populates daily_prices, factor_returns, category_stats)
- logs/bloomberg_backfill_YYYYMMDD.log

VERSION: 1.0.0
CREATED: 2026-01-31

PURPOSE:
Backfill 90 days of historical data from Bloomberg for all 970 assets.
This is a ONE-TIME script to bootstrap the database with history.

PREREQUISITES:
- Bloomberg Terminal running and logged in
- blpapi Python package installed
- Run from Windows (Parallels) with Bloomberg access

USAGE:
    python bloomberg_backfill.py                    # Default: 90 days
    python bloomberg_backfill.py --days 60          # Custom days
    python bloomberg_backfill.py --test             # Test with 5 tickers only
    python bloomberg_backfill.py --resume           # Resume interrupted run

RUNTIME ESTIMATE:
- ~970 tickers × 90 days = ~87,000 data points
- Batched requests: ~50 tickers per batch
- Estimated time: 15-30 minutes

=============================================================================
"""

import sys
import os
import json
import time
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Bloomberg API
try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    print("ERROR: blpapi not installed")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - adjust for Windows if needed
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
NEWS_DIR = PROJECT_DIR.parent

# Input: Master asset list
MASTER_LIST_PATH = NEWS_DIR / "Step 2 Data Processing - Final1000" / "Final 1000 Asset Master List.xlsx"

# Output: Database
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

# Logs
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Bloomberg settings
BATCH_SIZE = 50  # Tickers per request (Bloomberg limits)
REQUEST_TIMEOUT = 30000  # 30 seconds
RATE_LIMIT_DELAY = 0.5  # Seconds between batches

# Fields to fetch
DAILY_FIELDS = [
    "PX_LAST",           # Last price
    "PX_OPEN",           # Open price
    "PX_HIGH",           # High price
    "PX_LOW",            # Low price
    "CHG_PCT_1D",        # 1-day return
    "VOLUME",            # Volume
    "VOLATILITY_30D",    # 30-day volatility
    "VOLATILITY_60D",    # 60-day volatility
    "RSI_14D",           # 14-day RSI
]

# Factor tickers for beta attribution
FACTOR_TICKERS = {
    "SPX Index": "SPX",
    "RTY Index": "Russell2000",
    "NDX Index": "Nasdaq100",
    "RAY Index": "Value",  # Russell 1000 Value
    "RAG Index": "Growth",  # Russell 1000 Growth
    "MXEA Index": "EAFE",
    "MXEF Index": "EM",
    "LF98TRUU Index": "HY_Credit",
    "LUATTRUU Index": "Treasuries",
    "LBUSTRUU Index": "TIPS",
    "BCOM Index": "Commodities",
    "BCOMAG Index": "Agriculture",
    "XBTUSD Curncy": "Crypto",
    "FNER Index": "REIT_US",
    "BWREAL Index": "REIT_Global",
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to file and console."""
    log_file = LOG_DIR / f"bloomberg_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def save_daily_prices(conn: sqlite3.Connection, data: List[Dict]):
    """Insert or update daily prices."""
    cursor = conn.cursor()
    
    for row in data:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_prices 
            (date, ticker, price, price_open, price_high, price_low,
             return_1d, volume, volatility_30d, volatility_60d, rsi_14)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['date'],
            row['ticker'],
            row.get('price'),
            row.get('price_open'),
            row.get('price_high'),
            row.get('price_low'),
            row.get('return_1d'),
            row.get('volume'),
            row.get('volatility_30d'),
            row.get('volatility_60d'),
            row.get('rsi_14'),
        ))
    
    conn.commit()

def save_factor_returns(conn: sqlite3.Connection, data: List[Dict]):
    """Insert or update factor returns."""
    cursor = conn.cursor()
    
    for row in data:
        cursor.execute("""
            INSERT OR REPLACE INTO factor_returns 
            (date, factor_name, return_1d)
            VALUES (?, ?, ?)
        """, (
            row['date'],
            row['factor_name'],
            row.get('return_1d'),
        ))
    
    conn.commit()

# =============================================================================
# BLOOMBERG SESSION
# =============================================================================

class BloombergSession:
    """Wrapper for Bloomberg API session."""
    
    def __init__(self):
        self.session = None
        self.refdata_service = None
    
    def connect(self) -> bool:
        """Connect to Bloomberg."""
        try:
            options = blpapi.SessionOptions()
            options.setServerHost("localhost")
            options.setServerPort(8194)
            
            self.session = blpapi.Session(options)
            
            if not self.session.start():
                logger.error("Failed to start Bloomberg session")
                return False
            
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open refdata service")
                return False
            
            self.refdata_service = self.session.getService("//blp/refdata")
            logger.info("Bloomberg session connected")
            return True
            
        except Exception as e:
            logger.error(f"Bloomberg connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Bloomberg."""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg session disconnected")
    
    def fetch_historical(self, tickers: List[str], fields: List[str],
                        start_date: datetime, end_date: datetime) -> Dict[str, Dict]:
        """
        Fetch historical data for multiple tickers.
        
        Returns:
            {ticker: {date: {field: value, ...}, ...}, ...}
        """
        if not self.session:
            raise RuntimeError("Not connected to Bloomberg")
        
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        
        self.session.sendRequest(request)
        
        results = {}
        
        while True:
            event = self.session.nextEvent(REQUEST_TIMEOUT)
            
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        ticker = security_data.getElementAsString("security")
                        
                        results[ticker] = {}
                        
                        if security_data.hasElement("fieldData"):
                            field_data_array = security_data.getElement("fieldData")
                            
                            for i in range(field_data_array.numValues()):
                                field_data = field_data_array.getValueAsElement(i)
                                
                                date_str = None
                                if field_data.hasElement("date"):
                                    date_val = field_data.getElement("date")
                                    date_str = str(date_val.getValue())
                                
                                if date_str:
                                    results[ticker][date_str] = {}
                                    
                                    for field in fields:
                                        if field_data.hasElement(field):
                                            try:
                                                val = field_data.getElement(field).getValue()
                                                results[ticker][date_str][field] = val
                                            except:
                                                pass
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return results

# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_tickers() -> List[str]:
    """Load tickers from master list."""
    if not MASTER_LIST_PATH.exists():
        raise FileNotFoundError(f"Master list not found: {MASTER_LIST_PATH}")
    
    df = pd.read_excel(MASTER_LIST_PATH)
    tickers = df['Bloomberg_Ticker'].dropna().unique().tolist()
    
    logger.info(f"Loaded {len(tickers)} tickers from master list")
    return tickers

def process_historical_data(raw_data: Dict[str, Dict]) -> List[Dict]:
    """Convert raw Bloomberg data to database format."""
    rows = []
    
    for ticker, dates_data in raw_data.items():
        for date_str, fields in dates_data.items():
            row = {
                'ticker': ticker,
                'date': date_str,
                'price': fields.get('PX_LAST'),
                'price_open': fields.get('PX_OPEN'),
                'price_high': fields.get('PX_HIGH'),
                'price_low': fields.get('PX_LOW'),
                'return_1d': fields.get('CHG_PCT_1D'),
                'volume': fields.get('VOLUME'),
                'volatility_30d': fields.get('VOLATILITY_30D'),
                'volatility_60d': fields.get('VOLATILITY_60D'),
                'rsi_14': fields.get('RSI_14D'),
            }
            rows.append(row)
    
    return rows

def compute_derived_metrics(conn: sqlite3.Connection):
    """Compute percentiles, z-scores, and streaks from historical data."""
    logger.info("Computing derived metrics...")
    
    cursor = conn.cursor()
    
    # Get all distinct dates
    cursor.execute("SELECT DISTINCT date FROM daily_prices ORDER BY date")
    dates = [row[0] for row in cursor.fetchall()]
    
    if len(dates) < 2:
        logger.warning("Not enough dates for derived metrics")
        return
    
    # Compute 60-day percentile for each (date, ticker)
    logger.info("Computing 60-day percentiles...")
    
    for i, current_date in enumerate(dates):
        if i < 5:  # Need at least 5 days of history
            continue
        
        # Get lookback window (up to 60 days)
        lookback_start = max(0, i - 60)
        lookback_dates = dates[lookback_start:i]
        
        if not lookback_dates:
            continue
        
        # For each ticker, compute percentile of current return vs lookback
        cursor.execute("""
            UPDATE daily_prices
            SET percentile_60d = (
                SELECT CAST(
                    (SELECT COUNT(*) FROM daily_prices dp2 
                     WHERE dp2.ticker = daily_prices.ticker 
                       AND dp2.date IN ({})
                       AND dp2.return_1d < daily_prices.return_1d
                    ) AS FLOAT
                ) / NULLIF(
                    (SELECT COUNT(*) FROM daily_prices dp3 
                     WHERE dp3.ticker = daily_prices.ticker 
                       AND dp3.date IN ({})
                       AND dp3.return_1d IS NOT NULL
                    ), 0
                ) * 100
            )
            WHERE date = ?
        """.format(
            ','.join(['?'] * len(lookback_dates)),
            ','.join(['?'] * len(lookback_dates))
        ), lookback_dates + lookback_dates + [current_date])
    
    conn.commit()
    
    # Compute z-scores (return / 60-day volatility)
    logger.info("Computing z-scores...")
    
    cursor.execute("""
        UPDATE daily_prices
        SET z_score_1d = return_1d / NULLIF(volatility_60d / SQRT(252), 0)
        WHERE return_1d IS NOT NULL AND volatility_60d IS NOT NULL
    """)
    
    conn.commit()
    logger.info("Derived metrics computed")

def compute_category_stats(conn: sqlite3.Connection):
    """Compute category statistics for each date, including 60-day percentiles."""
    logger.info("Computing category statistics...")
    
    cursor = conn.cursor()
    
    # Get assets with their categories
    assets_df = pd.read_sql_query("""
        SELECT ticker, tier1, tier2, tier3_tags FROM assets
    """, conn)
    
    # Get all daily prices
    prices_df = pd.read_sql_query("""
        SELECT date, ticker, return_1d FROM daily_prices WHERE return_1d IS NOT NULL
    """, conn)
    
    if prices_df.empty:
        logger.warning("No price data for category stats")
        return
    
    # Merge
    merged = prices_df.merge(assets_df, on='ticker', how='left')
    
    # Sort dates for percentile calculation
    all_dates = sorted(merged['date'].unique())
    
    # Build historical category returns for percentile calculation
    # {(category_type, category_value): {date: avg_return}}
    category_history = {}
    
    # First pass: compute basic stats and collect history
    for date in all_dates:
        day_data = merged[merged['date'] == date]
        
        # Tier-1 stats
        for tier1 in day_data['tier1'].dropna().unique():
            tier1_data = day_data[day_data['tier1'] == tier1]
            
            if len(tier1_data) >= 3:
                avg_ret = tier1_data['return_1d'].mean()
                
                # Store in history
                key = ('tier1', tier1)
                if key not in category_history:
                    category_history[key] = {}
                category_history[key][date] = avg_ret
                
                cursor.execute("""
                    INSERT OR REPLACE INTO category_stats
                    (date, category_type, category_value, count, avg_return, 
                     median_return, std_return, min_return, max_return)
                    VALUES (?, 'tier1', ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, tier1, len(tier1_data),
                    avg_ret,
                    tier1_data['return_1d'].median(),
                    tier1_data['return_1d'].std(),
                    tier1_data['return_1d'].min(),
                    tier1_data['return_1d'].max(),
                ))
        
        # Tier-2 stats
        for tier2 in day_data['tier2'].dropna().unique():
            tier2_data = day_data[day_data['tier2'] == tier2]
            
            if len(tier2_data) >= 3:
                avg_ret = tier2_data['return_1d'].mean()
                
                # Store in history
                key = ('tier2', tier2)
                if key not in category_history:
                    category_history[key] = {}
                category_history[key][date] = avg_ret
                
                cursor.execute("""
                    INSERT OR REPLACE INTO category_stats
                    (date, category_type, category_value, count, avg_return,
                     median_return, std_return, min_return, max_return)
                    VALUES (?, 'tier2', ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, tier2, len(tier2_data),
                    avg_ret,
                    tier2_data['return_1d'].median(),
                    tier2_data['return_1d'].std(),
                    tier2_data['return_1d'].min(),
                    tier2_data['return_1d'].max(),
                ))
    
    conn.commit()
    
    # Second pass: compute 60-day percentiles
    logger.info("Computing category 60-day percentiles...")
    
    for (cat_type, cat_value), history in category_history.items():
        sorted_dates = sorted(history.keys())
        
        for i, current_date in enumerate(sorted_dates):
            # Get lookback window (up to 60 days before current)
            lookback_start = max(0, i - 60)
            lookback_dates = sorted_dates[lookback_start:i]  # Exclude current
            
            if len(lookback_dates) < 5:  # Need at least 5 days of history
                continue
            
            current_return = history[current_date]
            lookback_returns = [history[d] for d in lookback_dates]
            
            # Calculate percentile: what % of historical returns are below current
            below_count = sum(1 for r in lookback_returns if r < current_return)
            percentile = (below_count / len(lookback_returns)) * 100
            
            cursor.execute("""
                UPDATE category_stats
                SET percentile_60d = ?
                WHERE date = ? AND category_type = ? AND category_value = ?
            """, (percentile, current_date, cat_type, cat_value))
    
    conn.commit()
    logger.info("Category statistics and percentiles computed")

def compute_streaks(conn: sqlite3.Connection):
    """Compute consecutive positive/negative day streaks for categories."""
    logger.info("Computing category streaks...")
    
    cursor = conn.cursor()
    
    # Get category stats ordered by date
    df = pd.read_sql_query("""
        SELECT date, category_type, category_value, avg_return
        FROM category_stats
        ORDER BY category_type, category_value, date
    """, conn)
    
    if df.empty:
        return
    
    # Compute streaks for each category
    for (cat_type, cat_value), group in df.groupby(['category_type', 'category_value']):
        group = group.sort_values('date').reset_index(drop=True)
        
        streaks = []
        current_streak = 0
        
        for i, row in group.iterrows():
            ret = row['avg_return']
            
            if i == 0:
                current_streak = 1 if ret > 0 else -1 if ret < 0 else 0
            else:
                prev_ret = group.iloc[i-1]['avg_return']
                
                if ret > 0 and prev_ret > 0:
                    current_streak = abs(current_streak) + 1
                elif ret < 0 and prev_ret < 0:
                    current_streak = -abs(current_streak) - 1
                else:
                    current_streak = 1 if ret > 0 else -1 if ret < 0 else 0
            
            streaks.append(current_streak)
        
        # Update database
        for i, (_, row) in enumerate(group.iterrows()):
            direction = 'positive' if streaks[i] > 0 else 'negative' if streaks[i] < 0 else 'neutral'
            
            cursor.execute("""
                UPDATE category_stats
                SET streak_days = ?, streak_direction = ?
                WHERE date = ? AND category_type = ? AND category_value = ?
            """, (streaks[i], direction, row['date'], cat_type, cat_value))
    
    conn.commit()
    logger.info("Category streaks computed")

# =============================================================================
# MAIN BACKFILL FUNCTION
# =============================================================================

def run_backfill(days: int = 90, test_mode: bool = False, resume: bool = False):
    """
    Run the historical data backfill.
    
    Args:
        days: Number of days of history to fetch
        test_mode: If True, only fetch 5 tickers
        resume: If True, skip tickers already in database
    """
    logger.info("=" * 70)
    logger.info("BLOOMBERG HISTORICAL DATA BACKFILL")
    logger.info("=" * 70)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra days for weekends/holidays
    
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Target: ~{days} trading days")
    
    # Load tickers
    tickers = load_tickers()
    
    if test_mode:
        tickers = tickers[:5]
        logger.info(f"TEST MODE: Using only {len(tickers)} tickers")
    
    # Check for resume
    conn = get_db()
    
    if resume:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM daily_prices")
        existing_tickers = {row[0] for row in cursor.fetchall()}
        
        tickers = [t for t in tickers if t not in existing_tickers]
        logger.info(f"RESUME MODE: {len(existing_tickers)} tickers already loaded, {len(tickers)} remaining")
    
    if not tickers:
        logger.info("No tickers to process")
        return
    
    # Connect to Bloomberg
    bbg = BloombergSession()
    if not bbg.connect():
        logger.error("Failed to connect to Bloomberg")
        return
    
    try:
        # Process in batches
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        total_rows = 0
        failed_tickers = []
        
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]
            
            logger.info(f"Batch {batch_num + 1}/{total_batches}: {len(batch_tickers)} tickers")
            
            try:
                # Fetch historical data
                raw_data = bbg.fetch_historical(
                    batch_tickers, DAILY_FIELDS, start_date, end_date
                )
                
                # Process and save
                rows = process_historical_data(raw_data)
                
                if rows:
                    save_daily_prices(conn, rows)
                    total_rows += len(rows)
                    logger.info(f"  Saved {len(rows)} rows (total: {total_rows})")
                else:
                    logger.warning(f"  No data returned for batch")
                
                # Track tickers with no data
                for ticker in batch_tickers:
                    if ticker not in raw_data or not raw_data[ticker]:
                        failed_tickers.append(ticker)
                
            except Exception as e:
                logger.error(f"  Batch error: {e}")
                failed_tickers.extend(batch_tickers)
            
            # Rate limit
            if batch_num < total_batches - 1:
                time.sleep(RATE_LIMIT_DELAY)
        
        # Fetch factor returns
        logger.info("\nFetching factor returns...")
        
        try:
            factor_raw = bbg.fetch_historical(
                list(FACTOR_TICKERS.keys()), 
                ["CHG_PCT_1D"], 
                start_date, 
                end_date
            )
            
            factor_rows = []
            for ticker, dates_data in factor_raw.items():
                factor_name = FACTOR_TICKERS.get(ticker, ticker)
                for date_str, fields in dates_data.items():
                    factor_rows.append({
                        'date': date_str,
                        'factor_name': factor_name,
                        'return_1d': fields.get('CHG_PCT_1D'),
                    })
            
            if factor_rows:
                save_factor_returns(conn, factor_rows)
                logger.info(f"Saved {len(factor_rows)} factor return rows")
                
        except Exception as e:
            logger.error(f"Factor returns error: {e}")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("BACKFILL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total rows saved: {total_rows}")
        logger.info(f"Failed tickers: {len(failed_tickers)}")
        
        if failed_tickers[:10]:
            logger.info(f"Sample failures: {failed_tickers[:10]}")
        
        # Compute derived metrics
        logger.info("\n" + "=" * 70)
        logger.info("COMPUTING DERIVED METRICS")
        logger.info("=" * 70)
        
        compute_derived_metrics(conn)
        compute_category_stats(conn)
        compute_streaks(conn)
        
        logger.info("\n" + "=" * 70)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 70)
        
    finally:
        bbg.disconnect()
        conn.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bloomberg Historical Data Backfill")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--test", action="store_true", help="Test mode (5 tickers only)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    args = parser.parse_args()
    
    try:
        run_backfill(days=args.days, test_mode=args.test, resume=args.resume)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
