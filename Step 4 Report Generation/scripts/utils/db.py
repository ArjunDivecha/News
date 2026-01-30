#!/usr/bin/env python3
"""
=============================================================================
DATABASE UTILITIES
=============================================================================

PURPOSE:
Common database operations for the report generation pipeline.

USAGE:
    from utils.db import get_db, get_assets, get_latest_prices, save_daily_prices
=============================================================================
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "database" / "market_data.db"


def get_db() -> sqlite3.Connection:
    """Get database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_assets(tier1: Optional[str] = None, tier2: Optional[str] = None) -> pd.DataFrame:
    """
    Get assets from database with optional filtering.
    
    Args:
        tier1: Filter by Tier-1 category
        tier2: Filter by Tier-2 category
        
    Returns:
        DataFrame with asset data
    """
    conn = get_db()
    
    query = "SELECT * FROM assets WHERE 1=1"
    params = []
    
    if tier1:
        query += " AND tier1 = ?"
        params.append(tier1)
    if tier2:
        query += " AND tier2 = ?"
        params.append(tier2)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df


def get_tickers() -> List[str]:
    """Get list of all tickers in database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM assets ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


def get_latest_prices(date: Optional[str] = None) -> pd.DataFrame:
    """
    Get latest daily prices, optionally for a specific date.
    
    Args:
        date: Date string (YYYY-MM-DD). If None, gets most recent date.
        
    Returns:
        DataFrame with price data joined with asset info
    """
    conn = get_db()
    
    if date is None:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        result = cursor.fetchone()
        date = result[0] if result[0] else None
    
    if date is None:
        conn.close()
        return pd.DataFrame()
    
    query = """
        SELECT 
            dp.*,
            a.name, a.tier1, a.tier2, a.tier3_tags, a.source,
            a.beta_spx, a.beta_russell2000, a.beta_nasdaq100,
            a.beta_eafe, a.beta_em
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ?
    """
    
    df = pd.read_sql_query(query, conn, params=[date])
    conn.close()
    
    return df


def save_daily_prices(df: pd.DataFrame, date: str) -> int:
    """
    Save daily price data to database.
    
    Args:
        df: DataFrame with columns matching daily_prices schema
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted/updated
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Ensure date column
    df = df.copy()
    df['date'] = date
    
    # Required columns
    required = ['date', 'ticker']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must have columns: {required}")
    
    # Get valid columns from schema
    cursor.execute("PRAGMA table_info(daily_prices)")
    valid_columns = [row[1] for row in cursor.fetchall()]
    
    # Filter to valid columns
    cols = [c for c in df.columns if c in valid_columns]
    df = df[cols]
    
    # Insert or replace
    placeholders = ','.join(['?' for _ in cols])
    cols_str = ','.join(cols)
    
    count = 0
    for _, row in df.iterrows():
        values = [None if pd.isna(v) else v for v in row.values]
        cursor.execute(
            f"INSERT OR REPLACE INTO daily_prices ({cols_str}) VALUES ({placeholders})",
            values
        )
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_category_stats(stats: List[Dict], date: str) -> int:
    """
    Save category statistics to database.
    
    Args:
        stats: List of dicts with category stats
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted
    """
    conn = get_db()
    cursor = conn.cursor()
    
    count = 0
    for stat in stats:
        cursor.execute("""
            INSERT OR REPLACE INTO category_stats 
            (date, category_type, category_value, count, avg_return, median_return,
             std_return, min_return, max_return, best_ticker, best_return,
             worst_ticker, worst_return, percentile_60d, streak_days, streak_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            stat.get('category_type'),
            stat.get('category_value'),
            stat.get('count'),
            stat.get('avg_return'),
            stat.get('median_return'),
            stat.get('std_return'),
            stat.get('min_return'),
            stat.get('max_return'),
            stat.get('best_ticker'),
            stat.get('best_return'),
            stat.get('worst_ticker'),
            stat.get('worst_return'),
            stat.get('percentile_60d'),
            stat.get('streak_days'),
            stat.get('streak_direction')
        ))
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_factor_returns(factors: Dict[str, float], date: str) -> int:
    """
    Save factor returns to database.
    
    Args:
        factors: Dict mapping factor name to return
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted
    """
    conn = get_db()
    cursor = conn.cursor()
    
    count = 0
    for factor_name, return_1d in factors.items():
        cursor.execute("""
            INSERT OR REPLACE INTO factor_returns (date, factor_name, return_1d)
            VALUES (?, ?, ?)
        """, (date, factor_name, return_1d))
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_report(report_id: str, report_type: str, report_date: str,
                content_md: str, model_name: str, **kwargs) -> None:
    """
    Save generated report to database.
    
    Args:
        report_id: Unique identifier
        report_type: 'daily' or 'flash'
        report_date: Date of the report
        content_md: Markdown content
        model_name: LLM model used
        **kwargs: Additional fields (pdf_path, tokens_input, etc.)
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO reports 
        (report_id, report_type, report_date, generated_at, content_md,
         model_name, pdf_path, tokens_input, tokens_output, generation_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        report_type,
        report_date,
        datetime.now().isoformat(),
        content_md,
        model_name,
        kwargs.get('pdf_path'),
        kwargs.get('tokens_input'),
        kwargs.get('tokens_output'),
        kwargs.get('generation_time_ms')
    ))
    
    conn.commit()
    conn.close()


def get_historical_stats(category_type: str, category_value: str, 
                         days: int = 60) -> pd.DataFrame:
    """
    Get historical category stats for pattern analysis.
    
    Args:
        category_type: 'tier1', 'tier2', or 'tier3_tag'
        category_value: The category value to look up
        days: Number of days of history
        
    Returns:
        DataFrame with historical stats
    """
    conn = get_db()
    
    query = """
        SELECT * FROM category_stats
        WHERE category_type = ? AND category_value = ?
        ORDER BY date DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=[category_type, category_value, days])
    conn.close()
    
    return df


def get_tier1_distribution() -> Dict[str, int]:
    """Get count of assets by Tier-1 category."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT tier1, COUNT(*) FROM assets GROUP BY tier1")
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result


def get_tier2_distribution() -> Dict[str, int]:
    """Get count of assets by Tier-2 category."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT tier2, COUNT(*) FROM assets GROUP BY tier2 ORDER BY COUNT(*) DESC")
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result
