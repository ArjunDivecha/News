#!/usr/bin/env python3
"""
=============================================================================
LOAD BLOOMBERG DATA FROM CSV (Run on Mac)
=============================================================================

INPUT FILES:
- data/bloomberg_data_YYYY-MM-DD.csv (from bloomberg_fetcher.py)
- data/factor_returns_YYYY-MM-DD.csv (from bloomberg_fetcher.py)

OUTPUT:
- Updates daily_prices table in database/market_data.db
- Updates factor_returns table in database/market_data.db
- Computes and saves category_stats

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Load Bloomberg data from CSV files (written by bloomberg_fetcher.py on Windows)
into the SQLite database. Computes derived metrics and category statistics.

USAGE:
    python scripts/04_load_bloomberg.py                    # Today's data
    python scripts/04_load_bloomberg.py --date 2026-01-30  # Specific date

=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.db import (get_db, save_daily_prices, save_category_stats, 
                      save_factor_returns, get_assets)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = PROJECT_DIR / "database" / "market_data.db"

# Column mapping: Bloomberg field -> database column
FIELD_MAP = {
    'ticker': 'ticker',
    'PX_LAST': 'price',
    'PX_OPEN': 'price_open',
    'PX_HIGH': 'price_high',
    'PX_LOW': 'price_low',
    'CHG_PCT_1D': 'return_1d',
    'CHG_PCT_1WK': 'return_1w',
    'CHG_PCT_1M': 'return_1m',
    'CHG_PCT_YTD': 'return_ytd',
    'CHG_PCT_1YR': 'return_1y',
    'VOLUME': 'volume',
    'VOLATILITY_30D': 'volatility_30d',
    'VOLATILITY_60D': 'volatility_60d',
}


def load_bloomberg_csv(date: str) -> pd.DataFrame:
    """Load Bloomberg data from CSV file."""
    csv_path = DATA_DIR / f"bloomberg_data_{date}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Bloomberg data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} rows from {csv_path.name}")
    
    # Rename columns
    df = df.rename(columns=FIELD_MAP)
    
    return df


def load_factor_returns_csv(date: str) -> Dict[str, float]:
    """Load factor returns from CSV file."""
    csv_path = DATA_DIR / f"factor_returns_{date}.csv"
    
    if not csv_path.exists():
        print(f"      WARNING: Factor returns not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    returns = dict(zip(df['factor_name'], df['return_1d']))
    print(f"      Loaded {len(returns)} factor returns from {csv_path.name}")
    
    return returns


def compute_derived_metrics(df: pd.DataFrame, factor_returns: Dict[str, float],
                           assets: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived metrics like z-scores and alpha.
    
    Adds:
    - z_score_1d: return / volatility
    - beta_predicted_return: sum of beta * factor_return
    - alpha_1d: actual - predicted
    """
    df = df.copy()
    
    # Z-score (if we have volatility)
    if 'volatility_60d' in df.columns:
        df['z_score_1d'] = df.apply(
            lambda row: row['return_1d'] / row['volatility_60d'] * 100 
            if pd.notna(row.get('volatility_60d')) and row.get('volatility_60d', 0) > 0
            else None,
            axis=1
        )
    
    # Beta-predicted return and alpha
    if factor_returns and not assets.empty:
        # Create ticker -> beta mapping
        beta_cols = [c for c in assets.columns if c.startswith('beta_')]
        
        # Map beta columns to factor names
        beta_to_factor = {
            'beta_spx': 'SPX',
            'beta_russell2000': 'Russell2000',
            'beta_nasdaq100': 'Nasdaq100',
            'beta_russell_value': 'Value',
            'beta_russell_growth': 'Growth',
            'beta_eafe': 'EAFE',
            'beta_em': 'EM',
            'beta_hy_credit': 'HYCredit',
            'beta_treasuries': 'Treasuries',
            'beta_tips': 'TIPS',
            'beta_commodity': 'Commodities',
            'beta_agriculture': 'Agriculture',
            'beta_crypto': 'Crypto',
            'beta_reit_us': 'REIT_US',
            'beta_reit_global': 'REIT_Global',
        }
        
        # Compute predicted return for each asset
        predicted = {}
        for _, asset in assets.iterrows():
            ticker = asset['ticker']
            pred_return = 0.0
            
            for beta_col, factor_name in beta_to_factor.items():
                if beta_col in asset and pd.notna(asset[beta_col]) and factor_name in factor_returns:
                    try:
                        pred_return += float(asset[beta_col]) * factor_returns[factor_name]
                    except (TypeError, ValueError):
                        pass
            
            predicted[ticker] = pred_return
        
        # Add to dataframe
        df['beta_predicted_return'] = df['ticker'].map(predicted)
        df['alpha_1d'] = df['return_1d'] - df['beta_predicted_return']
    
    return df


def compute_category_stats(df: pd.DataFrame, assets: pd.DataFrame, date: str) -> List[Dict]:
    """Compute category-level statistics."""
    stats = []
    
    # Merge with asset data to get categories
    merged = df.merge(assets[['ticker', 'tier1', 'tier2']], on='ticker', how='left')
    
    # Tier-1 stats
    for tier1, group in merged.groupby('tier1'):
        if pd.isna(tier1):
            continue
        returns = group['return_1d'].dropna()
        if len(returns) == 0:
            continue
            
        stats.append({
            'category_type': 'tier1',
            'category_value': tier1,
            'count': len(group),
            'avg_return': round(returns.mean(), 4),
            'median_return': round(returns.median(), 4),
            'std_return': round(returns.std(), 4) if len(returns) > 1 else 0,
            'min_return': round(returns.min(), 4),
            'max_return': round(returns.max(), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(returns) > 0 else None,
            'best_return': round(returns.max(), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(returns) > 0 else None,
            'worst_return': round(returns.min(), 4),
        })
    
    # Tier-2 stats
    for tier2, group in merged.groupby('tier2'):
        if pd.isna(tier2):
            continue
        returns = group['return_1d'].dropna()
        if len(returns) == 0:
            continue
            
        stats.append({
            'category_type': 'tier2',
            'category_value': tier2,
            'count': len(group),
            'avg_return': round(returns.mean(), 4),
            'median_return': round(returns.median(), 4),
            'std_return': round(returns.std(), 4) if len(returns) > 1 else 0,
            'min_return': round(returns.min(), 4),
            'max_return': round(returns.max(), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(returns) > 0 else None,
            'best_return': round(returns.max(), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(returns) > 0 else None,
            'worst_return': round(returns.min(), 4),
        })
    
    return stats


def load_bloomberg_data(date: str, verbose: bool = True) -> Dict:
    """
    Load Bloomberg data from CSV into SQLite database.
    
    Args:
        date: Date string (YYYY-MM-DD)
        verbose: Print progress
        
    Returns:
        Dict with load statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING BLOOMBERG DATA FOR {date}")
        print(f"{'='*70}")
    
    # Load CSV data
    if verbose:
        print("\n[1/5] Loading CSV files...")
    
    try:
        df = load_bloomberg_csv(date)
    except FileNotFoundError as e:
        print(f"      ERROR: {e}")
        return {'error': str(e)}
    
    factor_returns = load_factor_returns_csv(date)
    
    # Load asset data for beta calculations
    if verbose:
        print("\n[2/5] Loading asset data...")
    assets = get_assets()
    if verbose:
        print(f"      Loaded {len(assets)} assets with classifications")
    
    # Compute derived metrics
    if verbose:
        print("\n[3/5] Computing derived metrics...")
    df = compute_derived_metrics(df, factor_returns, assets)
    
    has_z = df['z_score_1d'].notna().sum() if 'z_score_1d' in df.columns else 0
    has_alpha = df['alpha_1d'].notna().sum() if 'alpha_1d' in df.columns else 0
    if verbose:
        print(f"      Z-scores computed: {has_z}")
        print(f"      Alpha computed: {has_alpha}")
    
    # Save to database
    if verbose:
        print("\n[4/5] Saving to database...")
    
    prices_saved = save_daily_prices(df, date)
    if verbose:
        print(f"      Saved {prices_saved} price records")
    
    factors_saved = save_factor_returns(factor_returns, date)
    if verbose:
        print(f"      Saved {factors_saved} factor returns")
    
    # Compute and save category stats
    if verbose:
        print("\n[5/5] Computing category statistics...")
    
    stats = compute_category_stats(df, assets, date)
    stats_saved = save_category_stats(stats, date)
    if verbose:
        print(f"      Saved {stats_saved} category stats")
    
    # Summary
    result = {
        'date': date,
        'prices_saved': prices_saved,
        'factors_saved': factors_saved,
        'stats_saved': stats_saved,
        'z_scores_computed': has_z,
        'alpha_computed': has_alpha,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("BLOOMBERG DATA LOAD COMPLETE")
        print(f"{'='*70}")
        print(f"\nDate: {date}")
        print(f"Prices: {prices_saved}")
        print(f"Factors: {factors_saved}")
        print(f"Category stats: {stats_saved}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Load Bloomberg data from CSV")
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help="Date to load (YYYY-MM-DD)")
    args = parser.parse_args()
    
    try:
        result = load_bloomberg_data(args.date, verbose=True)
        
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            return 1
        
        print("\n✓ Bloomberg data loaded successfully")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
