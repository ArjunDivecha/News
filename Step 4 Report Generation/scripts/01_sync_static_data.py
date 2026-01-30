#!/usr/bin/env python3
"""
=============================================================================
SYNC STATIC DATA TO SQLITE
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final Master.xlsx (for beta data)

OUTPUT:
- Updates assets table in database/market_data.db

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Sync the Final 1000 Asset Master List (with classifications and betas) 
to the SQLite database. This script should be run:
- Once initially to populate the database
- Monthly/quarterly when the asset list is updated

USAGE:
    python scripts/01_sync_static_data.py
    python scripts/01_sync_static_data.py --dry-run  # Preview without writing
=============================================================================
"""

import sqlite3
import pandas as pd
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STEP2_DIR = PROJECT_ROOT / "Step 2 Data Processing - Final1000"
FINAL_1000_PATH = STEP2_DIR / "Final 1000 Asset Master List.xlsx"
FINAL_MASTER_PATH = STEP2_DIR / "Final Master.xlsx"
DB_PATH = Path(__file__).parent.parent / "database" / "market_data.db"

# Beta column mapping: Excel column name -> database column name
BETA_COLUMN_MAP = {
    'Daily 1 Year Beta to SPX': 'beta_spx',
    'Russell 2000 Index': 'beta_russell2000',
    'Nasdaq-100 Index': 'beta_nasdaq100',
    'Russell 1000 Value Index': 'beta_russell_value',
    'Russell 1000 Growth Index': 'beta_russell_growth',
    'MSCI EAFE Index': 'beta_eafe',
    'MSCI Emerging Markets Index': 'beta_em',
    'Bloomberg US Corporate High Yield Total Return Index Value Unhedged USD': 'beta_hy_credit',
    'Bloomberg Global Agg Treasuries Total Return Index Value Unhedged USD': 'beta_treasuries',
    'Bloomberg US Treasury Inflation Notes TR Index Value Unhedged USD': 'beta_tips',
    'Commodity Index': 'beta_commodity',
    'Bloomberg Agriculture Subindex': 'beta_agriculture',
    'Bitwise 10 Crypto Index Fund': 'beta_crypto',
    'MSCI US REIT Index': 'beta_reit_us',
    'FTSE EPRA NAREIT DEVELOPED Total Return Index USD': 'beta_reit_global',
}


def load_final_1000() -> pd.DataFrame:
    """Load the Final 1000 Asset Master List."""
    print(f"\n[1/5] Loading Final 1000 Asset Master List...")
    print(f"      Path: {FINAL_1000_PATH}")
    
    if not FINAL_1000_PATH.exists():
        raise FileNotFoundError(f"File not found: {FINAL_1000_PATH}")
    
    df = pd.read_excel(FINAL_1000_PATH)
    print(f"      Loaded {len(df):,} assets")
    print(f"      Columns: {list(df.columns)}")
    
    return df


def load_beta_data() -> pd.DataFrame:
    """Load beta data from Final Master (has more columns)."""
    print(f"\n[2/5] Loading beta data from Final Master...")
    print(f"      Path: {FINAL_MASTER_PATH}")
    
    if not FINAL_MASTER_PATH.exists():
        print(f"      WARNING: Final Master not found, skipping beta data")
        return None
    
    df = pd.read_excel(FINAL_MASTER_PATH)
    print(f"      Loaded {len(df):,} rows")
    
    # Check which beta columns exist
    available_betas = [col for col in BETA_COLUMN_MAP.keys() if col in df.columns]
    print(f"      Available beta columns: {len(available_betas)}/{len(BETA_COLUMN_MAP)}")
    
    return df


def prepare_asset_data(df_final: pd.DataFrame, df_master: pd.DataFrame = None) -> pd.DataFrame:
    """Prepare asset data for insertion into database."""
    print(f"\n[3/5] Preparing asset data...")
    
    # Start with Final 1000 data
    assets = df_final.copy()
    
    # Standardize column names
    column_map = {
        'Bloomberg_Ticker': 'ticker',
        'Name': 'name',
        'Long_Description': 'description',
        'category_tier1': 'tier1',
        'category_tier2': 'tier2',
        'category_tags': 'tier3_tags',
        'source': 'source',
        'Sharpe_1Y': 'sharpe_1y',
        'Return_12M': 'return_12m',
        'Return_36M': 'return_36m',
        'Vol_12M': 'vol_12m',
        'Beta_SPX': 'beta_spx',
        'Correlation_SPX': 'correlation_spx',
        'Selection_Score': 'selection_score',
        'Quality_Score': 'quality_score',
        'Thematic_Rarity': 'thematic_rarity',
    }
    
    # Rename columns that exist
    for old_name, new_name in column_map.items():
        if old_name in assets.columns:
            assets = assets.rename(columns={old_name: new_name})
    
    # If we have master data, merge beta columns
    if df_master is not None:
        print("      Merging beta data from Final Master...")
        
        # Create ticker column in master if needed
        if 'Bloomberg_Ticker' in df_master.columns:
            df_master = df_master.rename(columns={'Bloomberg_Ticker': 'ticker'})
        elif 'Ticker' in df_master.columns:
            df_master = df_master.rename(columns={'Ticker': 'ticker'})
        
        # Select only ticker and beta columns
        beta_cols = ['ticker'] + [col for col in BETA_COLUMN_MAP.keys() if col in df_master.columns]
        df_betas = df_master[beta_cols].copy()
        
        # Rename beta columns
        for old_name, new_name in BETA_COLUMN_MAP.items():
            if old_name in df_betas.columns:
                df_betas = df_betas.rename(columns={old_name: new_name})
        
        # Merge on ticker
        assets = assets.merge(df_betas, on='ticker', how='left', suffixes=('', '_master'))
        
        # Use master beta_spx if available and Final 1000 doesn't have it
        if 'beta_spx_master' in assets.columns:
            assets['beta_spx'] = assets['beta_spx'].fillna(assets['beta_spx_master'])
            assets = assets.drop(columns=['beta_spx_master'])
    
    # Convert tier3_tags to JSON string if it's not already
    if 'tier3_tags' in assets.columns:
        def to_json(val):
            if pd.isna(val):
                return '[]'
            if isinstance(val, str):
                # Already a string, check if it's JSON
                if val.startswith('['):
                    return val
                # Convert comma-separated to JSON array
                tags = [t.strip() for t in val.split(',') if t.strip()]
                return json.dumps(tags)
            return '[]'
        
        assets['tier3_tags'] = assets['tier3_tags'].apply(to_json)
    
    # Add timestamp
    assets['updated_at'] = datetime.now().isoformat()
    
    print(f"      Prepared {len(assets):,} assets with {len(assets.columns)} columns")
    
    return assets


def sync_to_database(assets: pd.DataFrame, dry_run: bool = False) -> dict:
    """Sync asset data to SQLite database."""
    print(f"\n[4/5] Syncing to database...")
    print(f"      Database: {DB_PATH}")
    print(f"      Dry run: {dry_run}")
    
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run init_db.py first.")
    
    stats = {
        'total': len(assets),
        'inserted': 0,
        'updated': 0,
        'errors': 0,
    }
    
    if dry_run:
        print("      [DRY RUN] Would sync {stats['total']} assets")
        return stats
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get existing tickers
    cursor.execute("SELECT ticker FROM assets")
    existing_tickers = set(row[0] for row in cursor.fetchall())
    print(f"      Existing assets in DB: {len(existing_tickers)}")
    
    # Prepare insert/update
    db_columns = [
        'ticker', 'name', 'description', 'tier1', 'tier2', 'tier3_tags', 'source',
        'beta_spx', 'beta_russell2000', 'beta_nasdaq100', 'beta_russell_value',
        'beta_russell_growth', 'beta_eafe', 'beta_em', 'beta_hy_credit',
        'beta_treasuries', 'beta_tips', 'beta_commodity', 'beta_agriculture',
        'beta_crypto', 'beta_reit_us', 'beta_reit_global',
        'sharpe_1y', 'sharpe_3y', 'return_12m', 'return_36m', 'vol_12m',
        'correlation_spx', 'selection_score', 'quality_score', 'thematic_rarity',
        'updated_at'
    ]
    
    # Filter to columns that exist in assets dataframe
    available_columns = [col for col in db_columns if col in assets.columns]
    
    placeholders = ','.join(['?' for _ in available_columns])
    columns_str = ','.join(available_columns)
    
    insert_sql = f"INSERT OR REPLACE INTO assets ({columns_str}) VALUES ({placeholders})"
    
    # Insert/update each asset
    for idx, row in assets.iterrows():
        try:
            values = [row.get(col) for col in available_columns]
            # Convert NaN to None
            values = [None if pd.isna(v) else v for v in values]
            cursor.execute(insert_sql, values)
            
            if row['ticker'] in existing_tickers:
                stats['updated'] += 1
            else:
                stats['inserted'] += 1
                
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"      ERROR on {row.get('ticker', 'unknown')}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"      Inserted: {stats['inserted']}")
    print(f"      Updated: {stats['updated']}")
    print(f"      Errors: {stats['errors']}")
    
    return stats


def verify_sync() -> dict:
    """Verify the sync by querying the database."""
    print(f"\n[5/5] Verifying sync...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    results = {}
    
    # Total count
    cursor.execute("SELECT COUNT(*) FROM assets")
    results['total_assets'] = cursor.fetchone()[0]
    print(f"      Total assets: {results['total_assets']}")
    
    # Tier-1 distribution
    cursor.execute("""
        SELECT tier1, COUNT(*) as count 
        FROM assets 
        GROUP BY tier1 
        ORDER BY count DESC
    """)
    results['tier1_distribution'] = cursor.fetchall()
    print(f"\n      Tier-1 Distribution:")
    for tier1, count in results['tier1_distribution']:
        print(f"        {tier1}: {count}")
    
    # Tier-2 distribution (top 10)
    cursor.execute("""
        SELECT tier2, COUNT(*) as count 
        FROM assets 
        GROUP BY tier2 
        ORDER BY count DESC
        LIMIT 10
    """)
    results['tier2_distribution'] = cursor.fetchall()
    print(f"\n      Top 10 Tier-2 Categories:")
    for tier2, count in results['tier2_distribution']:
        print(f"        {tier2}: {count}")
    
    # Source distribution
    cursor.execute("""
        SELECT source, COUNT(*) as count 
        FROM assets 
        GROUP BY source 
        ORDER BY count DESC
    """)
    results['source_distribution'] = cursor.fetchall()
    print(f"\n      Source Distribution (internal only):")
    for source, count in results['source_distribution']:
        print(f"        {source}: {count}")
    
    # Beta data coverage
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN beta_spx IS NOT NULL THEN 1 ELSE 0 END) as has_spx_beta,
            SUM(CASE WHEN beta_russell2000 IS NOT NULL THEN 1 ELSE 0 END) as has_r2k_beta,
            SUM(CASE WHEN beta_eafe IS NOT NULL THEN 1 ELSE 0 END) as has_eafe_beta,
            COUNT(*) as total
        FROM assets
    """)
    beta_coverage = cursor.fetchone()
    results['beta_coverage'] = {
        'spx': beta_coverage[0],
        'russell2000': beta_coverage[1],
        'eafe': beta_coverage[2],
        'total': beta_coverage[3]
    }
    print(f"\n      Beta Data Coverage:")
    print(f"        SPX beta: {beta_coverage[0]}/{beta_coverage[3]}")
    print(f"        Russell 2000 beta: {beta_coverage[1]}/{beta_coverage[3]}")
    print(f"        EAFE beta: {beta_coverage[2]}/{beta_coverage[3]}")
    
    conn.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Sync Final 1000 data to SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to database")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SYNC STATIC DATA TO SQLITE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    try:
        # Load data
        df_final = load_final_1000()
        df_master = load_beta_data()
        
        # Prepare data
        assets = prepare_asset_data(df_final, df_master)
        
        # Sync to database
        stats = sync_to_database(assets, dry_run=args.dry_run)
        
        # Verify
        if not args.dry_run:
            results = verify_sync()
        
        print("\n" + "=" * 70)
        print("SYNC COMPLETE")
        print("=" * 70)
        print(f"\nAssets synced: {stats['total']}")
        print(f"Inserted: {stats['inserted']}")
        print(f"Updated: {stats['updated']}")
        print(f"Errors: {stats['errors']}")
        
        if stats['errors'] == 0:
            print("\n✓ Sync completed successfully")
            return 0
        else:
            print(f"\n⚠ Sync completed with {stats['errors']} errors")
            return 1
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
