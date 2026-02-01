#!/usr/bin/env python3
"""
=============================================================================
LOAD BLOOMBERG DATA FROM EXCEL (Run on Mac)
=============================================================================

INPUT FILES:
- data/bloomberg_data_YYYY-MM-DD.xlsx (exported from Bloomberg Excel)
  OR
- data/Bloomberg_Data_Template.xlsx (with values, not formulas)

OUTPUT:
- Updates daily_prices table in database/market_data.db
- Updates factor_returns table in database/market_data.db
- Computes and saves category_stats

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Load Bloomberg data from Excel files (exported from Bloomberg Excel Add-in)
into the SQLite database. This is an alternative to the CSV/API approach.

USAGE:
    python scripts/05_load_excel.py                         # Auto-detect file
    python scripts/05_load_excel.py --date 2026-01-30       # Specific date
    python scripts/05_load_excel.py --file data/myfile.xlsx # Specific file

=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
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

# Column mapping: Excel column -> database column
COLUMN_MAP = {
    'Ticker': 'ticker',
    'Name': 'name',
    'Tier1': 'tier1',
    'Tier2': 'tier2',
    'Last_Price': 'price',
    'Chg_1D': 'return_1d',
    'Chg_5D': 'return_5d',
    'Chg_1M': 'return_1m',
    'Chg_YTD': 'return_ytd',
    'Chg_1Y': 'return_1y',
    'RSI_14': 'rsi_14',
    'Vol_30D': 'volatility_30d',
    'Vol_240D': 'volatility_240d',
}


def find_excel_file(date: Optional[str] = None) -> Optional[Path]:
    """Find the Bloomberg Excel file to load."""
    if date:
        # Try date-specific file first
        candidates = [
            DATA_DIR / f"bloomberg_data_{date}.xlsx",
            DATA_DIR / f"Bloomberg_Data_{date}.xlsx",
        ]
        for path in candidates:
            if path.exists():
                return path
    
    # Try template file (if it has values)
    template = DATA_DIR / "Bloomberg_Data_Template.xlsx"
    if template.exists():
        return template
    
    # Find any xlsx file in data folder
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))
    if xlsx_files:
        # Return most recent
        return max(xlsx_files, key=lambda p: p.stat().st_mtime)
    
    return None


def load_asset_data(filepath: Path) -> pd.DataFrame:
    """Load asset data from Excel file."""
    print(f"      Loading from: {filepath.name}")
    
    # Try to read Asset_Data sheet
    try:
        df = pd.read_excel(filepath, sheet_name="Asset_Data")
    except:
        # Fall back to first sheet
        df = pd.read_excel(filepath, sheet_name=0)
    
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"      Columns: {list(df.columns)}")
    
    # Rename columns
    df = df.rename(columns=COLUMN_MAP)
    
    # Check for formula errors (Bloomberg returns #N/A for missing data)
    error_values = ['#N/A', '#N/A N/A', '#N/A Invalid Field', '#N/A Field Not Applicable', 
                    '#NAME?', '#REF!', '#VALUE!', '#DIV/0!', '#NULL!', '#NUM!']
    
    for col in df.columns:
        if df[col].dtype == object:
            # Replace error strings with NaN
            df[col] = df[col].replace(error_values, np.nan)
            # Also catch any string starting with #N/A
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.startswith('#') else x)
    
    # Convert numeric columns to float
    numeric_cols = ['price', 'return_1d', 'return_5d', 'return_1m', 'return_ytd', 'return_1y',
                    'rsi_14', 'volatility_30d', 'volatility_240d']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_factor_data(filepath: Path) -> Dict[str, float]:
    """Load factor returns from Excel file."""
    try:
        df = pd.read_excel(filepath, sheet_name="Factor_Returns")
    except:
        print("      WARNING: Factor_Returns sheet not found")
        return {}
    
    # Extract factor returns
    factors = {}
    for _, row in df.iterrows():
        factor_name = row.get('Factor_Name')
        return_1d = row.get('Chg_1D')
        
        if pd.notna(factor_name) and pd.notna(return_1d):
            try:
                factors[factor_name] = float(return_1d)
            except (TypeError, ValueError):
                pass
    
    print(f"      Loaded {len(factors)} factor returns")
    return factors


def compute_derived_metrics(df: pd.DataFrame, factor_returns: Dict[str, float],
                           assets: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics like z-scores and alpha."""
    df = df.copy()
    
    # Z-score
    if 'volatility_60d' in df.columns and 'return_1d' in df.columns:
        df['z_score_1d'] = df.apply(
            lambda row: row['return_1d'] / row['volatility_60d'] * 100 
            if pd.notna(row.get('volatility_60d')) and row.get('volatility_60d', 0) > 0
            else None,
            axis=1
        )
    
    # Beta-predicted return and alpha
    if factor_returns and not assets.empty:
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
        
        df['beta_predicted_return'] = df['ticker'].map(predicted)
        if 'return_1d' in df.columns:
            df['alpha_1d'] = df['return_1d'] - df['beta_predicted_return']
    
    return df


def compute_category_stats(df: pd.DataFrame, assets: pd.DataFrame, date: str) -> List[Dict]:
    """Compute category-level statistics."""
    stats = []
    
    # Check if tier columns are already in df (from Excel)
    has_tier1_in_df = 'tier1' in df.columns
    has_tier2_in_df = 'tier2' in df.columns
    
    # Merge with asset data to get categories (only if not already present)
    if has_tier1_in_df and has_tier2_in_df:
        # Excel already has tier columns, use them directly
        merged = df.copy()
    else:
        # Need to get tier columns from assets table
        merged = df.merge(assets[['ticker', 'tier1', 'tier2']], on='ticker', how='left', suffixes=('', '_db'))
        # If Excel had partial tier data, prefer it over database
        if has_tier1_in_df:
            merged['tier1'] = merged['tier1'].fillna(merged.get('tier1_db', merged['tier1']))
        if has_tier2_in_df:
            merged['tier2'] = merged['tier2'].fillna(merged.get('tier2_db', merged['tier2']))
    
    if 'return_1d' not in merged.columns:
        print("      WARNING: No return_1d column, skipping category stats")
        return stats
    
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
            'avg_return': round(float(returns.mean()), 4),
            'median_return': round(float(returns.median()), 4),
            'std_return': round(float(returns.std()), 4) if len(returns) > 1 else 0,
            'min_return': round(float(returns.min()), 4),
            'max_return': round(float(returns.max()), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(returns) > 0 else None,
            'best_return': round(float(returns.max()), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(returns) > 0 else None,
            'worst_return': round(float(returns.min()), 4),
            'percentile_60d': None,
            'streak_days': None,
            'streak_direction': None,
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
            'avg_return': round(float(returns.mean()), 4),
            'median_return': round(float(returns.median()), 4),
            'std_return': round(float(returns.std()), 4) if len(returns) > 1 else 0,
            'min_return': round(float(returns.min()), 4),
            'max_return': round(float(returns.max()), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(returns) > 0 else None,
            'best_return': round(float(returns.max()), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(returns) > 0 else None,
            'worst_return': round(float(returns.min()), 4),
            'percentile_60d': None,
            'streak_days': None,
            'streak_direction': None,
        })
    
    return stats


def load_excel_data(date: str, filepath: Optional[Path] = None, 
                   verbose: bool = True) -> Dict:
    """
    Load Bloomberg data from Excel into SQLite database.
    
    Args:
        date: Date string (YYYY-MM-DD)
        filepath: Optional specific file path
        verbose: Print progress
        
    Returns:
        Dict with load statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING BLOOMBERG EXCEL DATA FOR {date}")
        print(f"{'='*70}")
    
    # Find file
    if verbose:
        print("\n[1/5] Finding Excel file...")
    
    if filepath:
        excel_path = Path(filepath)
    else:
        excel_path = find_excel_file(date)
    
    if excel_path is None or not excel_path.exists():
        error = f"No Excel file found for date {date}"
        print(f"      ERROR: {error}")
        return {'error': error}
    
    if verbose:
        print(f"      Found: {excel_path.name}")
    
    # Load data
    if verbose:
        print("\n[2/5] Loading Excel data...")
    
    df = load_asset_data(excel_path)
    factor_returns = load_factor_data(excel_path)
    
    # Load asset data for categories and betas
    if verbose:
        print("\n[3/5] Loading asset metadata...")
    assets = get_assets()
    if verbose:
        print(f"      Loaded {len(assets)} assets with classifications")
    
    # Compute derived metrics
    if verbose:
        print("\n[4/5] Computing derived metrics...")
    df = compute_derived_metrics(df, factor_returns, assets)
    
    has_z = df['z_score_1d'].notna().sum() if 'z_score_1d' in df.columns else 0
    has_alpha = df['alpha_1d'].notna().sum() if 'alpha_1d' in df.columns else 0
    if verbose:
        print(f"      Z-scores computed: {has_z}")
        print(f"      Alpha computed: {has_alpha}")
    
    # Save to database
    if verbose:
        print("\n[5/5] Saving to database...")
    
    prices_saved = save_daily_prices(df, date)
    if verbose:
        print(f"      Saved {prices_saved} price records")
    
    factors_saved = save_factor_returns(factor_returns, date) if factor_returns else 0
    if verbose:
        print(f"      Saved {factors_saved} factor returns")
    
    # Compute and save category stats
    stats = compute_category_stats(df, assets, date)
    stats_saved = save_category_stats(stats, date) if stats else 0
    if verbose:
        print(f"      Saved {stats_saved} category stats")
    
    # Summary
    result = {
        'date': date,
        'file': str(excel_path),
        'prices_saved': prices_saved,
        'factors_saved': factors_saved,
        'stats_saved': stats_saved,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("EXCEL DATA LOAD COMPLETE")
        print(f"{'='*70}")
        print(f"\nDate: {date}")
        print(f"File: {excel_path.name}")
        print(f"Prices: {prices_saved}")
        print(f"Factors: {factors_saved}")
        print(f"Category stats: {stats_saved}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Load Bloomberg data from Excel")
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help="Date to load (YYYY-MM-DD)")
    parser.add_argument("--file", type=str, default=None,
                       help="Specific Excel file to load")
    args = parser.parse_args()
    
    try:
        filepath = Path(args.file) if args.file else None
        result = load_excel_data(args.date, filepath, verbose=True)
        
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            return 1
        
        print("\n✓ Excel data loaded successfully")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
