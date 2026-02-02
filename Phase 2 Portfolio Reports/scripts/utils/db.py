#!/usr/bin/env python3
"""
=============================================================================
DATABASE UTILITIES - Phase 2 Portfolio Reports
=============================================================================

Common database operations for the portfolio report generation pipeline.
Uses a SEPARATE database from Phase 1 (portfolio.db vs market_data.db).

USAGE:
    from utils.db import get_db, get_portfolio, get_holdings, save_holdings
=============================================================================
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Database paths
PHASE2_DB_PATH = Path(__file__).parent.parent.parent / "database" / "portfolio.db"
PHASE1_DB_PATH = Path(__file__).parent.parent.parent.parent / "Step 4 Report Generation" / "database" / "market_data.db"


def get_db() -> sqlite3.Connection:
    """Get Phase 2 portfolio database connection."""
    conn = sqlite3.connect(PHASE2_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_phase1_db() -> sqlite3.Connection:
    """Get Phase 1 market data database connection (read-only)."""
    if not PHASE1_DB_PATH.exists():
        raise FileNotFoundError(f"Phase 1 database not found: {PHASE1_DB_PATH}")
    conn = sqlite3.connect(f"file:{PHASE1_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# PORTFOLIO OPERATIONS
# =============================================================================

def create_portfolio(portfolio_id: str, portfolio_name: str, 
                     client_name: str = None) -> None:
    """Create a new portfolio."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO portfolios (portfolio_id, portfolio_name, client_name, updated_at)
        VALUES (?, ?, ?, datetime('now'))
    """, (portfolio_id, portfolio_name, client_name))
    conn.commit()
    conn.close()


def get_portfolio(portfolio_id: str) -> Optional[dict]:
    """Get portfolio by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_portfolios() -> List[dict]:
    """List all portfolios."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM portfolios WHERE is_active = 1 ORDER BY portfolio_name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# =============================================================================
# HOLDINGS OPERATIONS
# =============================================================================

def save_holdings(portfolio_id: str, holdings: List[dict]) -> int:
    """
    Save portfolio holdings to database.
    
    Args:
        portfolio_id: Portfolio identifier
        holdings: List of holding dicts with required fields
        
    Returns:
        Number of holdings saved
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Clear existing holdings for this portfolio
    cursor.execute("DELETE FROM portfolio_holdings WHERE portfolio_id = ?", (portfolio_id,))
    
    count = 0
    for h in holdings:
        cursor.execute("""
            INSERT INTO portfolio_holdings (
                portfolio_id, symbol, position_type, quantity, market_value, avg_price, open_pnl,
                yf_ticker, security_name, security_type, yf_sector, yf_industry, yf_category,
                country, currency, tier1, tier2, tier3_tags, final1000_ticker,
                classification_source, resolution_status, resolution_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            portfolio_id,
            h.get('symbol'),
            h.get('position_type', 'LONG'),
            h.get('quantity'),
            h.get('market_value'),
            h.get('avg_price'),
            h.get('open_pnl'),
            h.get('yf_ticker'),
            h.get('security_name'),
            h.get('security_type'),
            h.get('yf_sector'),
            h.get('yf_industry'),
            h.get('yf_category'),
            h.get('country'),
            h.get('currency'),
            h.get('tier1'),
            h.get('tier2'),
            json.dumps(h['tier3_tags']) if h.get('tier3_tags') else None,
            h.get('final1000_ticker'),
            h.get('classification_source'),
            h.get('resolution_status', 'pending'),
            h.get('resolution_error'),
        ))
        count += 1
    
    conn.commit()
    conn.close()
    return count


def get_holdings(portfolio_id: str, resolved_only: bool = True) -> pd.DataFrame:
    """
    Get holdings for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        resolved_only: If True, only return resolved holdings
        
    Returns:
        DataFrame with holdings
    """
    conn = get_db()
    
    query = "SELECT * FROM portfolio_holdings WHERE portfolio_id = ?"
    params = [portfolio_id]
    
    if resolved_only:
        query += " AND resolution_status = 'resolved'"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Parse tier3_tags JSON
    if 'tier3_tags' in df.columns:
        df['tier3_tags'] = df['tier3_tags'].apply(
            lambda x: json.loads(x) if x else []
        )
    
    return df


def get_holdings_summary(portfolio_id: str) -> dict:
    """Get summary stats for portfolio holdings."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_count,
            SUM(CASE WHEN resolution_status = 'resolved' THEN 1 ELSE 0 END) as resolved_count,
            SUM(CASE WHEN resolution_status = 'failed' THEN 1 ELSE 0 END) as failed_count,
            SUM(CASE WHEN position_type = 'LONG' THEN 1 ELSE 0 END) as long_count,
            SUM(CASE WHEN position_type = 'SHORT' THEN 1 ELSE 0 END) as short_count,
            SUM(CASE WHEN position_type = 'LONG' THEN market_value ELSE 0 END) as long_value,
            SUM(CASE WHEN position_type = 'SHORT' THEN market_value ELSE 0 END) as short_value
        FROM portfolio_holdings 
        WHERE portfolio_id = ?
    """, (portfolio_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else {}


# =============================================================================
# PHASE 1 LOOKUPS
# =============================================================================

def lookup_final1000(ticker: str) -> Optional[dict]:
    """
    Look up a ticker in Phase 1's Final 1000 assets table.
    
    Returns:
        Dict with tier1, tier2, tier3_tags if found, else None
    """
    try:
        conn = get_phase1_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, name, tier1, tier2, tier3_tags 
            FROM assets 
            WHERE ticker = ?
        """, (ticker,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'ticker': row['ticker'],
                'name': row['name'],
                'tier1': row['tier1'],
                'tier2': row['tier2'],
                'tier3_tags': json.loads(row['tier3_tags']) if row['tier3_tags'] else [],
            }
        return None
    except Exception as e:
        print(f"WARNING: Could not lookup {ticker} in Final 1000: {e}")
        return None


def get_phase1_market_data(date: str) -> Optional[pd.DataFrame]:
    """
    Get Phase 1 market data for a specific date.
    
    Returns:
        DataFrame with category stats and factor returns
    """
    try:
        conn = get_phase1_db()
        
        # Get category stats
        category_df = pd.read_sql_query("""
            SELECT * FROM category_stats WHERE date = ?
        """, conn, params=[date])
        
        # Get factor returns
        factor_df = pd.read_sql_query("""
            SELECT * FROM factor_returns WHERE date = ?
        """, conn, params=[date])
        
        conn.close()
        
        return {
            'category_stats': category_df,
            'factor_returns': factor_df,
        }
    except Exception as e:
        print(f"WARNING: Could not get Phase 1 data for {date}: {e}")
        return None


# =============================================================================
# DAILY DATA OPERATIONS
# =============================================================================

def save_daily_snapshot(portfolio_id: str, date: str, data: List[dict]) -> int:
    """Save daily portfolio snapshot."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Clear existing for this date/portfolio
    cursor.execute("""
        DELETE FROM portfolio_daily WHERE date = ? AND portfolio_id = ?
    """, (date, portfolio_id))
    
    count = 0
    for d in data:
        cursor.execute("""
            INSERT INTO portfolio_daily (
                date, portfolio_id, holding_id, symbol, position_type,
                quantity, price, market_value_usd, weight,
                avg_price, cost_basis, open_pnl, open_pnl_pct, daily_pnl,
                return_1d, return_ytd, contribution_1d, fetch_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, portfolio_id, d.get('holding_id'), d.get('symbol'), d.get('position_type'),
            d.get('quantity'), d.get('price'), d.get('market_value_usd'), d.get('weight'),
            d.get('avg_price'), d.get('cost_basis'), d.get('open_pnl'), d.get('open_pnl_pct'),
            d.get('daily_pnl'), d.get('return_1d'), d.get('return_ytd'), 
            d.get('contribution_1d'), d.get('fetch_status'),
        ))
        count += 1
    
    conn.commit()
    conn.close()
    return count


def get_daily_snapshot(portfolio_id: str, date: str) -> pd.DataFrame:
    """Get daily portfolio snapshot."""
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT d.*, h.tier1, h.tier2, h.tier3_tags, h.security_name, h.country
        FROM portfolio_daily d
        JOIN portfolio_holdings h ON d.holding_id = h.id
        WHERE d.date = ? AND d.portfolio_id = ?
    """, conn, params=[date, portfolio_id])
    conn.close()
    return df


def save_aggregates(portfolio_id: str, date: str, aggregates: List[dict]) -> int:
    """Save portfolio aggregates."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Clear existing
    cursor.execute("""
        DELETE FROM portfolio_aggregates WHERE date = ? AND portfolio_id = ?
    """, (date, portfolio_id))
    
    count = 0
    for a in aggregates:
        cursor.execute("""
            INSERT INTO portfolio_aggregates (
                date, portfolio_id, dimension_type, dimension_value,
                holding_count, long_count, short_count,
                total_weight, long_weight, short_weight, total_value_usd,
                weighted_return_1d, contribution_1d
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, portfolio_id, a.get('dimension_type'), a.get('dimension_value'),
            a.get('holding_count'), a.get('long_count'), a.get('short_count'),
            a.get('total_weight'), a.get('long_weight'), a.get('short_weight'),
            a.get('total_value_usd'), a.get('weighted_return_1d'), a.get('contribution_1d'),
        ))
        count += 1
    
    conn.commit()
    conn.close()
    return count


def get_aggregates(portfolio_id: str, date: str) -> pd.DataFrame:
    """Get portfolio aggregates for a date."""
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT * FROM portfolio_aggregates 
        WHERE date = ? AND portfolio_id = ?
    """, conn, params=[date, portfolio_id])
    conn.close()
    return df


def save_portfolio_summary(portfolio_id: str, date: str, summary: dict) -> None:
    """Save portfolio summary."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO portfolio_summary (
            date, portfolio_id, total_market_value, total_long_value, total_short_value,
            net_exposure, gross_exposure, holding_count, long_count, short_count,
            portfolio_return_1d, portfolio_return_ytd, total_open_pnl, daily_pnl,
            top_contributors, top_detractors
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date, portfolio_id,
        summary.get('total_market_value'),
        summary.get('total_long_value'),
        summary.get('total_short_value'),
        summary.get('net_exposure'),
        summary.get('gross_exposure'),
        summary.get('holding_count'),
        summary.get('long_count'),
        summary.get('short_count'),
        summary.get('portfolio_return_1d'),
        summary.get('portfolio_return_ytd'),
        summary.get('total_open_pnl'),
        summary.get('daily_pnl'),
        json.dumps(summary.get('top_contributors', [])),
        json.dumps(summary.get('top_detractors', [])),
    ))
    
    conn.commit()
    conn.close()


def get_portfolio_summary(portfolio_id: str, date: str) -> Optional[dict]:
    """Get portfolio summary for a date."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM portfolio_summary WHERE date = ? AND portfolio_id = ?
    """, (date, portfolio_id))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        result['top_contributors'] = json.loads(result['top_contributors']) if result['top_contributors'] else []
        result['top_detractors'] = json.loads(result['top_detractors']) if result['top_detractors'] else []
        return result
    return None


# =============================================================================
# REPORT OPERATIONS
# =============================================================================

def save_report(report_id: str, portfolio_id: str, report_date: str,
                content_md: str, model_name: str, **kwargs) -> None:
    """Save generated report to database."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO portfolio_reports 
        (report_id, portfolio_id, report_date, generated_at, content_md,
         model_name, pdf_path, tokens_input, tokens_output, generation_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        portfolio_id,
        report_date,
        datetime.now().isoformat(),
        content_md,
        model_name,
        kwargs.get('pdf_path'),
        kwargs.get('tokens_input'),
        kwargs.get('tokens_output'),
        kwargs.get('generation_time_ms'),
    ))
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Test database operations
    print("Testing database utilities...")
    
    # Test Phase 1 connection
    try:
        conn = get_phase1_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM assets")
        print(f"Phase 1 assets count: {cursor.fetchone()[0]}")
        conn.close()
    except Exception as e:
        print(f"Phase 1 DB error: {e}")
    
    # Test Final 1000 lookup
    result = lookup_final1000("SPY")
    print(f"SPY lookup: {result}")
    
    result = lookup_final1000("EWZ")
    print(f"EWZ lookup: {result}")
    
    print("Database utilities test complete.")
