#!/usr/bin/env python3
"""
=============================================================================
BLOOMBERG DAILY DATA REFRESH
=============================================================================

INPUT FILES:
- /Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx
- database/market_data.db (existing history)

OUTPUT FILES:
- database/market_data.db (updated with today's data)
- logs/bloomberg_daily_YYYYMMDD.log

VERSION: 1.0.0
CREATED: 2026-01-31

PURPOSE:
Fetch today's market data for all 970 assets and append to the database.
Run this script DAILY after market close to update the database.

PREREQUISITES:
- Bloomberg Terminal running and logged in
- blpapi Python package installed
- Historical backfill already completed (bloomberg_backfill.py)

USAGE:
    python bloomberg_daily.py                    # Fetch today's data
    python bloomberg_daily.py --date 2026-01-30  # Fetch specific date
    python bloomberg_daily.py --test             # Test with 5 tickers

RUNTIME ESTIMATE:
- ~970 tickers in ~20 batches
- Estimated time: 2-5 minutes

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
from typing import List, Dict, Optional
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

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
NEWS_DIR = PROJECT_DIR.parent

MASTER_LIST_PATH = NEWS_DIR / "Step 2 Data Processing - Final1000" / "Final 1000 Asset Master List.xlsx"
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 50
REQUEST_TIMEOUT = 30000
RATE_LIMIT_DELAY = 0.3

# Fields for daily refresh
DAILY_FIELDS = [
    "PX_LAST",
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "CHG_PCT_1D",
    "CHG_PCT_1W",
    "CHG_PCT_1M",
    "CHG_PCT_YTD",
    "VOLUME",
    "VOLATILITY_30D",
    "VOLATILITY_60D",
    "RSI_14D",
]

# Factor tickers
FACTOR_TICKERS = {
    "SPX Index": "SPX",
    "RTY Index": "Russell2000",
    "NDX Index": "Nasdaq100",
    "RAY Index": "Value",
    "RAG Index": "Growth",
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
# LOGGING
# =============================================================================

def setup_logging():
    log_file = LOG_DIR / f"bloomberg_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# DATABASE
# =============================================================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def save_daily_prices(conn: sqlite3.Connection, data: List[Dict], date: str):
    """Insert or update daily prices for a specific date."""
    cursor = conn.cursor()
    
    for row in data:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_prices 
            (date, ticker, price, price_open, price_high, price_low,
             return_1d, return_1w, return_1m, return_ytd,
             volume, volatility_30d, volatility_60d, rsi_14)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            row['ticker'],
            row.get('price'),
            row.get('price_open'),
            row.get('price_high'),
            row.get('price_low'),
            row.get('return_1d'),
            row.get('return_1w'),
            row.get('return_1m'),
            row.get('return_ytd'),
            row.get('volume'),
            row.get('volatility_30d'),
            row.get('volatility_60d'),
            row.get('rsi_14'),
        ))
    
    conn.commit()

def save_factor_returns(conn: sqlite3.Connection, data: List[Dict], date: str):
    cursor = conn.cursor()
    
    for row in data:
        cursor.execute("""
            INSERT OR REPLACE INTO factor_returns 
            (date, factor_name, return_1d)
            VALUES (?, ?, ?)
        """, (date, row['factor_name'], row.get('return_1d')))
    
    conn.commit()

def update_derived_metrics(conn: sqlite3.Connection, date: str):
    """Update percentiles and z-scores for today's data."""
    cursor = conn.cursor()
    
    # Get 60-day lookback dates
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices 
        WHERE date < ? 
        ORDER BY date DESC 
        LIMIT 60
    """, (date,))
    lookback_dates = [row[0] for row in cursor.fetchall()]
    
    if len(lookback_dates) < 5:
        logger.warning(f"Only {len(lookback_dates)} days of history for percentile calc")
        return
    
    # Update z-scores
    cursor.execute("""
        UPDATE daily_prices
        SET z_score_1d = return_1d / NULLIF(volatility_60d / SQRT(252), 0)
        WHERE date = ? AND return_1d IS NOT NULL AND volatility_60d IS NOT NULL
    """, (date,))
    
    conn.commit()
    logger.info("Updated derived metrics")

def update_category_stats(conn: sqlite3.Connection, date: str):
    """Update category statistics for today."""
    cursor = conn.cursor()
    
    # Get today's data with categories
    df = pd.read_sql_query("""
        SELECT dp.ticker, dp.return_1d, a.tier1, a.tier2
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ? AND dp.return_1d IS NOT NULL
    """, conn, params=[date])
    
    if df.empty:
        logger.warning("No data for category stats")
        return
    
    # Tier-1 stats
    for tier1 in df['tier1'].dropna().unique():
        tier1_data = df[df['tier1'] == tier1]
        
        if len(tier1_data) >= 3:
            cursor.execute("""
                INSERT OR REPLACE INTO category_stats
                (date, category_type, category_value, count, avg_return,
                 median_return, std_return, min_return, max_return)
                VALUES (?, 'tier1', ?, ?, ?, ?, ?, ?, ?)
            """, (
                date, tier1, len(tier1_data),
                tier1_data['return_1d'].mean(),
                tier1_data['return_1d'].median(),
                tier1_data['return_1d'].std(),
                tier1_data['return_1d'].min(),
                tier1_data['return_1d'].max(),
            ))
    
    # Tier-2 stats
    for tier2 in df['tier2'].dropna().unique():
        tier2_data = df[df['tier2'] == tier2]
        
        if len(tier2_data) >= 3:
            cursor.execute("""
                INSERT OR REPLACE INTO category_stats
                (date, category_type, category_value, count, avg_return,
                 median_return, std_return, min_return, max_return)
                VALUES (?, 'tier2', ?, ?, ?, ?, ?, ?, ?)
            """, (
                date, tier2, len(tier2_data),
                tier2_data['return_1d'].mean(),
                tier2_data['return_1d'].median(),
                tier2_data['return_1d'].std(),
                tier2_data['return_1d'].min(),
                tier2_data['return_1d'].max(),
            ))
    
    conn.commit()
    logger.info("Updated category statistics")

def update_streaks(conn: sqlite3.Connection, date: str):
    """Update streak counts for today's categories."""
    cursor = conn.cursor()
    
    # Get yesterday's date
    cursor.execute("""
        SELECT MAX(date) FROM category_stats WHERE date < ?
    """, (date,))
    result = cursor.fetchone()
    prev_date = result[0] if result else None
    
    if not prev_date:
        logger.info("No previous date for streak calculation")
        return
    
    # Update streaks based on yesterday
    cursor.execute("""
        UPDATE category_stats
        SET streak_days = (
            SELECT CASE
                WHEN category_stats.avg_return > 0 AND prev.avg_return > 0 THEN ABS(prev.streak_days) + 1
                WHEN category_stats.avg_return < 0 AND prev.avg_return < 0 THEN -ABS(prev.streak_days) - 1
                WHEN category_stats.avg_return > 0 THEN 1
                WHEN category_stats.avg_return < 0 THEN -1
                ELSE 0
            END
            FROM category_stats prev
            WHERE prev.date = ?
              AND prev.category_type = category_stats.category_type
              AND prev.category_value = category_stats.category_value
        ),
        streak_direction = CASE
            WHEN avg_return > 0 THEN 'positive'
            WHEN avg_return < 0 THEN 'negative'
            ELSE 'neutral'
        END
        WHERE date = ?
    """, (prev_date, date))
    
    conn.commit()
    logger.info("Updated category streaks")

# =============================================================================
# BLOOMBERG
# =============================================================================

class BloombergSession:
    def __init__(self):
        self.session = None
        self.refdata_service = None
    
    def connect(self) -> bool:
        try:
            options = blpapi.SessionOptions()
            options.setServerHost("localhost")
            options.setServerPort(8194)
            
            self.session = blpapi.Session(options)
            
            if not self.session.start():
                logger.error("Failed to start session")
                return False
            
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open refdata service")
                return False
            
            self.refdata_service = self.session.getService("//blp/refdata")
            logger.info("Bloomberg connected")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        if self.session:
            self.session.stop()
            logger.info("Bloomberg disconnected")
    
    def fetch_reference_data(self, tickers: List[str], fields: List[str]) -> Dict[str, Dict]:
        """Fetch current reference data (BDP equivalent)."""
        if not self.session:
            raise RuntimeError("Not connected")
        
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        self.session.sendRequest(request)
        
        results = {}
        
        while True:
            event = self.session.nextEvent(REQUEST_TIMEOUT)
            
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        
                        for i in range(security_data.numValues()):
                            security = security_data.getValueAsElement(i)
                            ticker = security.getElementAsString("security")
                            
                            results[ticker] = {}
                            
                            if security.hasElement("fieldData"):
                                field_data = security.getElement("fieldData")
                                
                                for field in fields:
                                    if field_data.hasElement(field):
                                        try:
                                            val = field_data.getElement(field).getValue()
                                            results[ticker][field] = val
                                        except:
                                            pass
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return results

# =============================================================================
# MAIN
# =============================================================================

def load_tickers() -> List[str]:
    if not MASTER_LIST_PATH.exists():
        raise FileNotFoundError(f"Master list not found: {MASTER_LIST_PATH}")
    
    df = pd.read_excel(MASTER_LIST_PATH)
    return df['Bloomberg_Ticker'].dropna().unique().tolist()

def run_daily_refresh(date: str = None, test_mode: bool = False):
    """Run daily data refresh."""
    
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info("=" * 70)
    logger.info("BLOOMBERG DAILY DATA REFRESH")
    logger.info("=" * 70)
    logger.info(f"Date: {date}")
    
    # Load tickers
    tickers = load_tickers()
    
    if test_mode:
        tickers = tickers[:5]
        logger.info(f"TEST MODE: {len(tickers)} tickers")
    else:
        logger.info(f"Tickers: {len(tickers)}")
    
    # Connect
    bbg = BloombergSession()
    if not bbg.connect():
        return False
    
    conn = get_db()
    
    try:
        # Fetch data in batches
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        all_data = []
        
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]
            
            logger.info(f"Batch {batch_num + 1}/{total_batches}")
            
            try:
                raw_data = bbg.fetch_reference_data(batch_tickers, DAILY_FIELDS)
                
                for ticker, fields in raw_data.items():
                    row = {
                        'ticker': ticker,
                        'price': fields.get('PX_LAST'),
                        'price_open': fields.get('PX_OPEN'),
                        'price_high': fields.get('PX_HIGH'),
                        'price_low': fields.get('PX_LOW'),
                        'return_1d': fields.get('CHG_PCT_1D'),
                        'return_1w': fields.get('CHG_PCT_1W'),
                        'return_1m': fields.get('CHG_PCT_1M'),
                        'return_ytd': fields.get('CHG_PCT_YTD'),
                        'volume': fields.get('VOLUME'),
                        'volatility_30d': fields.get('VOLATILITY_30D'),
                        'volatility_60d': fields.get('VOLATILITY_60D'),
                        'rsi_14': fields.get('RSI_14D'),
                    }
                    all_data.append(row)
                    
            except Exception as e:
                logger.error(f"Batch error: {e}")
            
            if batch_num < total_batches - 1:
                time.sleep(RATE_LIMIT_DELAY)
        
        # Save to database
        if all_data:
            save_daily_prices(conn, all_data, date)
            logger.info(f"Saved {len(all_data)} rows")
        
        # Fetch factor returns
        logger.info("Fetching factor returns...")
        
        try:
            factor_raw = bbg.fetch_reference_data(
                list(FACTOR_TICKERS.keys()),
                ["CHG_PCT_1D"]
            )
            
            factor_data = []
            for ticker, fields in factor_raw.items():
                factor_name = FACTOR_TICKERS.get(ticker, ticker)
                factor_data.append({
                    'factor_name': factor_name,
                    'return_1d': fields.get('CHG_PCT_1D'),
                })
            
            if factor_data:
                save_factor_returns(conn, factor_data, date)
                logger.info(f"Saved {len(factor_data)} factor returns")
                
        except Exception as e:
            logger.error(f"Factor error: {e}")
        
        # Update derived metrics
        logger.info("Updating derived metrics...")
        update_derived_metrics(conn, date)
        update_category_stats(conn, date)
        update_streaks(conn, date)
        
        # Compute correlations for today
        logger.info("Computing rolling correlations...")
        try:
            from subprocess import run, PIPE
            result = run(
                ["python3", str(SCRIPT_DIR / "04_compute_correlations.py"), "--date", date],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Correlations computed successfully")
            else:
                logger.warning(f"Correlation computation warning: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")
        
        logger.info("=" * 70)
        logger.info("DAILY REFRESH COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    finally:
        bbg.disconnect()
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Bloomberg Daily Data Refresh")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD)")
    parser.add_argument("--test", action="store_true", help="Test mode")
    args = parser.parse_args()
    
    try:
        success = run_daily_refresh(date=args.date, test_mode=args.test)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
