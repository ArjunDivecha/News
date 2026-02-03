#!/usr/bin/env python3
"""
=============================================================================
ROLLING CORRELATION COMPUTATION
=============================================================================

INPUT FILES:
- database/market_data.db (daily_prices, factor_returns)

OUTPUT FILES:
- database/market_data.db (asset_correlations table)

VERSION: 1.0.0
CREATED: 2026-01-31

PURPOSE:
Compute rolling 60-day correlations between each asset and 15 factor indices.
Store results for use in regime detection and factor attribution analysis.

USAGE:
    python scripts/04_compute_correlations.py                    # Latest date only
    python scripts/04_compute_correlations.py --backfill         # All historical dates
    python scripts/04_compute_correlations.py --date 2026-01-30  # Specific date
    python scripts/04_compute_correlations.py --test             # Test with 10 tickers

RUNTIME ESTIMATE:
- ~955 tickers x 87 dates = ~83,000 correlation sets
- Backfill: 5-10 minutes (parallelized)
- Daily: <30 seconds

=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

# Logging
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Factor names (must match factor_returns table)
FACTORS = [
    'SPX', 'Russell2000', 'Nasdaq100', 'Value', 'Growth',
    'EAFE', 'EM', 'HY_Credit', 'Treasuries', 'TIPS',
    'Commodities', 'Agriculture', 'Crypto', 'REIT_US', 'REIT_Global'
]

# Column names in database
FACTOR_COLUMNS = {
    'SPX': 'corr_spx',
    'Russell2000': 'corr_russell2000',
    'Nasdaq100': 'corr_nasdaq100',
    'Value': 'corr_value',
    'Growth': 'corr_growth',
    'EAFE': 'corr_eafe',
    'EM': 'corr_em',
    'HY_Credit': 'corr_hy_credit',
    'Treasuries': 'corr_treasuries',
    'TIPS': 'corr_tips',
    'Commodities': 'corr_commodities',
    'Agriculture': 'corr_agriculture',
    'Crypto': 'corr_crypto',
    'REIT_US': 'corr_reit_us',
    'REIT_Global': 'corr_reit_global',
}

LOOKBACK_DAYS = 60
MIN_DATA_POINTS = 20  # Minimum days needed to compute correlation
REGIME_CHANGE_THRESHOLD = 0.3  # Correlation change threshold
REGIME_LOOKBACK_DAYS = 5  # Days to look back for regime change detection

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_all_dates() -> List[str]:
    """Get all dates with data, sorted ascending."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices 
        WHERE return_1d IS NOT NULL
        ORDER BY date ASC
    """)
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_all_tickers() -> List[str]:
    """Get all tickers with data."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM daily_prices")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


def get_factor_returns_df() -> pd.DataFrame:
    """Load all factor returns into a DataFrame indexed by date."""
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT date, factor_name, return_1d
        FROM factor_returns
        WHERE return_1d IS NOT NULL
        ORDER BY date
    """, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to wide format: date as index, factors as columns
    df_wide = df.pivot(index='date', columns='factor_name', values='return_1d')
    return df_wide


def get_ticker_returns_df() -> pd.DataFrame:
    """Load all ticker returns into a DataFrame."""
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT date, ticker, return_1d
        FROM daily_prices
        WHERE return_1d IS NOT NULL
        ORDER BY date, ticker
    """, conn)
    conn.close()
    
    return df


def save_correlations(records: List[Dict]):
    """Save correlation records to database."""
    if not records:
        return
    
    conn = get_db()
    cursor = conn.cursor()
    
    for rec in records:
        cursor.execute("""
            INSERT OR REPLACE INTO asset_correlations
            (date, ticker, corr_spx, corr_russell2000, corr_nasdaq100, corr_value,
             corr_growth, corr_eafe, corr_em, corr_hy_credit, corr_treasuries,
             corr_tips, corr_commodities, corr_agriculture, corr_crypto,
             corr_reit_us, corr_reit_global, r_squared_best, best_factor, regime_change)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec['date'],
            rec['ticker'],
            rec.get('corr_spx'),
            rec.get('corr_russell2000'),
            rec.get('corr_nasdaq100'),
            rec.get('corr_value'),
            rec.get('corr_growth'),
            rec.get('corr_eafe'),
            rec.get('corr_em'),
            rec.get('corr_hy_credit'),
            rec.get('corr_treasuries'),
            rec.get('corr_tips'),
            rec.get('corr_commodities'),
            rec.get('corr_agriculture'),
            rec.get('corr_crypto'),
            rec.get('corr_reit_us'),
            rec.get('corr_reit_global'),
            rec.get('r_squared_best'),
            rec.get('best_factor'),
            rec.get('regime_change', 0),
        ))
    
    conn.commit()
    conn.close()


def get_prior_correlations(ticker: str, date: str, days_back: int = 5) -> Optional[Dict]:
    """Get correlations from N days ago for regime change detection."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM asset_correlations
        WHERE ticker = ? AND date < ?
        ORDER BY date DESC
        LIMIT 1 OFFSET ?
    """, (ticker, date, days_back - 1))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


# =============================================================================
# CORRELATION COMPUTATION
# =============================================================================

def compute_correlations_for_date(
    date: str,
    ticker_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates_list: List[str],
    prior_correlations: Dict[str, Dict] = None
) -> List[Dict]:
    """
    Compute correlations for all tickers on a specific date.
    
    Args:
        date: Target date
        ticker_returns: DataFrame with columns [date, ticker, return_1d]
        factor_returns: DataFrame indexed by date with factor columns
        dates_list: All dates in ascending order
        prior_correlations: Dict of {ticker: prior_correlation_dict} for regime detection
        
    Returns:
        List of correlation records
    """
    # Find lookback window
    date_idx = dates_list.index(date) if date in dates_list else -1
    if date_idx < MIN_DATA_POINTS:
        return []
    
    start_idx = max(0, date_idx - LOOKBACK_DAYS)
    lookback_dates = dates_list[start_idx:date_idx + 1]
    
    if len(lookback_dates) < MIN_DATA_POINTS:
        return []
    
    # Filter factor returns to lookback window
    factor_window = factor_returns[factor_returns.index.isin(lookback_dates)]
    
    if len(factor_window) < MIN_DATA_POINTS:
        return []
    
    # Get ticker returns for this window
    ticker_window = ticker_returns[
        (ticker_returns['date'].isin(lookback_dates))
    ]
    
    records = []
    
    # Get unique tickers for this date
    tickers_on_date = ticker_returns[ticker_returns['date'] == date]['ticker'].unique()
    
    for ticker in tickers_on_date:
        ticker_data = ticker_window[ticker_window['ticker'] == ticker].set_index('date')['return_1d']
        
        if len(ticker_data) < MIN_DATA_POINTS:
            continue
        
        # Align with factor dates
        common_dates = ticker_data.index.intersection(factor_window.index)
        
        if len(common_dates) < MIN_DATA_POINTS:
            continue
        
        ticker_aligned = ticker_data.loc[common_dates]
        factors_aligned = factor_window.loc[common_dates]
        
        # Compute correlations
        correlations = {}
        for factor in FACTORS:
            if factor in factors_aligned.columns:
                factor_series = factors_aligned[factor].dropna()
                ticker_subset = ticker_aligned.loc[ticker_aligned.index.intersection(factor_series.index)]
                factor_subset = factor_series.loc[factor_series.index.intersection(ticker_aligned.index)]
                
                if len(ticker_subset) >= MIN_DATA_POINTS and len(factor_subset) >= MIN_DATA_POINTS:
                    try:
                        corr = ticker_subset.corr(factor_subset)
                        if not pd.isna(corr):
                            correlations[FACTOR_COLUMNS[factor]] = round(corr, 4)
                    except:
                        pass
        
        if not correlations:
            continue
        
        # Find best factor
        best_factor = None
        best_r_squared = 0
        for factor, col in FACTOR_COLUMNS.items():
            if col in correlations and correlations[col] is not None:
                r_sq = correlations[col] ** 2
                if r_sq > best_r_squared:
                    best_r_squared = r_sq
                    best_factor = factor
        
        # Detect regime change
        regime_change = 0
        if prior_correlations and ticker in prior_correlations:
            prior = prior_correlations[ticker]
            for factor, col in FACTOR_COLUMNS.items():
                if col in correlations and prior.get(col) is not None and correlations[col] is not None:
                    if abs(correlations[col] - prior[col]) > REGIME_CHANGE_THRESHOLD:
                        regime_change = 1
                        break
        
        record = {
            'date': date,
            'ticker': ticker,
            'r_squared_best': round(best_r_squared, 4) if best_r_squared else None,
            'best_factor': best_factor,
            'regime_change': regime_change,
            **correlations
        }
        
        records.append(record)
    
    return records


def compute_correlations_batch(
    dates: List[str],
    ticker_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    all_dates: List[str]
) -> List[Dict]:
    """Compute correlations for a batch of dates."""
    all_records = []
    prior_correlations = {}
    
    for i, date in enumerate(dates):
        records = compute_correlations_for_date(
            date, ticker_returns, factor_returns, all_dates, prior_correlations
        )
        all_records.extend(records)
        
        # Update prior correlations for next iteration
        for rec in records:
            prior_correlations[rec['ticker']] = rec
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(dates)} dates...")
    
    return all_records


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_backfill(test_mode: bool = False):
    """Backfill correlations for all historical dates."""
    logger.info("=" * 70)
    logger.info("CORRELATION BACKFILL")
    logger.info("=" * 70)
    
    # Load all data
    logger.info("\n[1/4] Loading data...")
    all_dates = get_all_dates()
    ticker_returns = get_ticker_returns_df()
    factor_returns = get_factor_returns_df()
    
    logger.info(f"  Total dates: {len(all_dates)}")
    logger.info(f"  Total ticker-date rows: {len(ticker_returns)}")
    logger.info(f"  Factor returns shape: {factor_returns.shape}")
    
    if test_mode:
        # Limit to last 10 days and 50 tickers
        test_dates = all_dates[-10:]
        test_tickers = get_all_tickers()[:50]
        ticker_returns = ticker_returns[
            (ticker_returns['date'].isin(test_dates)) &
            (ticker_returns['ticker'].isin(test_tickers))
        ]
        all_dates = test_dates
        logger.info(f"  TEST MODE: {len(test_dates)} dates, {len(test_tickers)} tickers")
    
    # Filter dates that have enough history
    valid_dates = [d for i, d in enumerate(all_dates) if i >= MIN_DATA_POINTS]
    logger.info(f"  Valid dates (>={MIN_DATA_POINTS} days history): {len(valid_dates)}")
    
    # Compute correlations
    logger.info("\n[2/4] Computing correlations...")
    start_time = datetime.now()
    
    all_records = compute_correlations_batch(
        valid_dates, ticker_returns, factor_returns, all_dates
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"  Computed {len(all_records)} correlation records in {elapsed:.1f}s")
    
    # Save to database
    logger.info("\n[3/4] Saving to database...")
    
    # Save in batches
    batch_size = 1000
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]
        save_correlations(batch)
        logger.info(f"  Saved {min(i + batch_size, len(all_records))}/{len(all_records)} records")
    
    # Summary
    logger.info("\n[4/4] Summary...")
    conn = get_db()
    
    summary = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT date) as unique_dates,
            COUNT(DISTINCT ticker) as unique_tickers,
            SUM(regime_change) as regime_changes,
            AVG(r_squared_best) as avg_r_squared
        FROM asset_correlations
    """, conn)
    
    best_factors = pd.read_sql_query("""
        SELECT best_factor, COUNT(*) as count
        FROM asset_correlations
        WHERE best_factor IS NOT NULL
        GROUP BY best_factor
        ORDER BY count DESC
    """, conn)
    
    conn.close()
    
    logger.info(f"  Total records: {summary.iloc[0]['total_records']}")
    logger.info(f"  Unique dates: {summary.iloc[0]['unique_dates']}")
    logger.info(f"  Unique tickers: {summary.iloc[0]['unique_tickers']}")
    logger.info(f"  Regime changes detected: {summary.iloc[0]['regime_changes']}")
    logger.info(f"  Average R-squared (best factor): {summary.iloc[0]['avg_r_squared']:.3f}")
    
    logger.info("\n  Best factor distribution:")
    for _, row in best_factors.head(10).iterrows():
        logger.info(f"    {row['best_factor']}: {row['count']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)


def run_daily(date: str = None):
    """Compute correlations for a single date (daily update)."""
    if date is None:
        # Get latest date
        all_dates = get_all_dates()
        if not all_dates:
            logger.error("No dates found in database")
            return
        date = all_dates[-1]
    
    logger.info("=" * 70)
    logger.info(f"DAILY CORRELATION UPDATE FOR {date}")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[1/3] Loading data...")
    all_dates = get_all_dates()
    ticker_returns = get_ticker_returns_df()
    factor_returns = get_factor_returns_df()
    
    # Get prior correlations for regime detection
    conn = get_db()
    prior_df = pd.read_sql_query(f"""
        SELECT * FROM asset_correlations
        WHERE date = (
            SELECT date FROM asset_correlations 
            WHERE date < '{date}'
            ORDER BY date DESC
            LIMIT 1 OFFSET {REGIME_LOOKBACK_DAYS - 1}
        )
    """, conn)
    conn.close()
    
    prior_correlations = {}
    for _, row in prior_df.iterrows():
        prior_correlations[row['ticker']] = dict(row)
    
    logger.info(f"  Prior correlations loaded: {len(prior_correlations)} tickers")
    
    # Compute
    logger.info("\n[2/3] Computing correlations...")
    records = compute_correlations_for_date(
        date, ticker_returns, factor_returns, all_dates, prior_correlations
    )
    
    logger.info(f"  Computed {len(records)} correlation records")
    
    # Save
    logger.info("\n[3/3] Saving to database...")
    save_correlations(records)
    
    # Summary
    regime_changes = sum(1 for r in records if r.get('regime_change', 0) == 1)
    logger.info(f"  Regime changes detected: {regime_changes}")
    
    if regime_changes > 0:
        logger.info("  Tickers with regime changes:")
        for r in records:
            if r.get('regime_change', 0) == 1:
                logger.info(f"    {r['ticker']}: best factor = {r['best_factor']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DAILY UPDATE COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute rolling correlations")
    parser.add_argument("--backfill", action="store_true", help="Backfill all historical dates")
    parser.add_argument("--date", type=str, help="Compute for specific date")
    parser.add_argument("--test", action="store_true", help="Test mode (limited data)")
    args = parser.parse_args()
    
    try:
        if args.backfill:
            run_backfill(test_mode=args.test)
        else:
            run_daily(date=args.date)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
