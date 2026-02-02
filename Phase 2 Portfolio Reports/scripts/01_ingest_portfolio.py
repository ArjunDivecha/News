#!/usr/bin/env python3
"""
=============================================================================
PORTFOLIO INGESTION SCRIPT - Phase 2 Portfolio Reports
=============================================================================

Loads portfolio from Excel/CSV, resolves tickers, classifies holdings,
and saves to database.

USAGE:
    python 01_ingest_portfolio.py --portfolio TEST --file test_portfolio.xlsx
    python 01_ingest_portfolio.py --portfolio CLIENT1 --file client_holdings.csv

INPUT:
    Excel/CSV file with columns:
    - Symbol (required)
    - Long Quantity (required) - positive for LONG, negative for SHORT
    - Market Value (optional but recommended)
    - Average Price (optional)
    - Long Open Profit/Loss (optional)

OUTPUT:
    - Portfolio record in portfolios table
    - Holdings in portfolio_holdings table with classification
=============================================================================
"""

import argparse
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import (
    get_db, create_portfolio, save_holdings, get_holdings_summary,
    lookup_final1000
)
from utils.yfinance_utils import resolve_ticker, batch_get_info
from utils.taxonomy import map_stock_to_taxonomy
from utils.llm import classify_with_haiku


def load_portfolio_file(file_path: str) -> pd.DataFrame:
    """Load portfolio from Excel or CSV file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Normalize column names - handle variations
    col_mapping = {}
    for col in df.columns:
        lower_col = col.lower().strip()
        if 'symbol' in lower_col or 'ticker' in lower_col:
            col_mapping[col] = 'symbol'
        elif 'quantity' in lower_col:
            col_mapping[col] = 'quantity'
        elif 'market' in lower_col and 'value' in lower_col:
            col_mapping[col] = 'market_value'
        elif 'average' in lower_col and 'price' in lower_col:
            col_mapping[col] = 'avg_price'
        elif 'profit' in lower_col or 'loss' in lower_col or 'pnl' in lower_col:
            col_mapping[col] = 'open_pnl'
    
    df.rename(columns=col_mapping, inplace=True)
    
    # Validate required columns
    if 'symbol' not in df.columns:
        raise ValueError("File must have a 'Symbol' column")
    if 'quantity' not in df.columns:
        raise ValueError("File must have a 'Long Quantity' or 'Quantity' column")
    
    return df


def determine_position_type(row: pd.Series) -> str:
    """Determine if position is LONG or SHORT based on quantity and market value."""
    quantity = row.get('quantity', 0)
    market_value = row.get('market_value', None)
    
    # Negative quantity = SHORT
    if quantity < 0:
        return 'SHORT'
    
    # Negative market value = SHORT
    if market_value is not None and market_value < 0:
        return 'SHORT'
    
    return 'LONG'


def classify_holding(symbol: str, yf_info: dict, verbose: bool = True) -> dict:
    """
    Classify a holding using the decision tree from PRD.
    
    Decision tree:
    1. Check Final 1000 database
    2. If stock (quoteType=EQUITY) → use yfinance sector mapping
    3. If ETF/Fund → use Claude Haiku
    4. Mark as failed if all fails
    """
    classification = {
        'tier1': None,
        'tier2': None,
        'tier3_tags': [],
        'classification_source': None,
    }
    
    # Step 1: Check Final 1000 database
    final1000 = lookup_final1000(symbol)
    if final1000:
        classification['tier1'] = final1000['tier1']
        classification['tier2'] = final1000['tier2']
        classification['tier3_tags'] = final1000['tier3_tags']
        classification['classification_source'] = 'final1000'
        classification['final1000_ticker'] = final1000['ticker']
        if verbose:
            print(f"    → Final 1000 match: {final1000['tier1']}")
        return classification
    
    # Step 2: Check quote type
    quote_type = yf_info.get('security_type', 'UNKNOWN')
    
    if quote_type == 'EQUITY':
        # Step 2a: Individual stock - use yfinance sector mapping
        mapping = map_stock_to_taxonomy({
            'sector': yf_info.get('sector'),
            'country': yf_info.get('country'),
        })
        classification['tier1'] = mapping['tier1']
        classification['tier2'] = mapping['tier2']
        classification['tier3_tags'] = mapping['tier3_tags']
        classification['classification_source'] = 'yfinance_mapped'
        if verbose:
            print(f"    → Stock mapped: {mapping['tier1']} / {mapping['tier2']}")
        return classification
    
    elif quote_type in ['ETF', 'MUTUALFUND']:
        # Step 2b: ETF/Fund - use Claude Haiku
        if verbose:
            print(f"    → Classifying ETF with Haiku...", end=" ", flush=True)
        
        result = classify_with_haiku(
            ticker=symbol,
            name=yf_info.get('name', ''),
            category=yf_info.get('category'),
            metadata={
                'country': yf_info.get('country'),
            }
        )
        
        if result:
            classification['tier1'] = result.get('tier1')
            classification['tier2'] = result.get('tier2')
            classification['tier3_tags'] = result.get('tier3_tags', [])
            classification['classification_source'] = 'haiku'
            if verbose:
                print(f"✓ {result.get('tier1')}")
            return classification
        else:
            if verbose:
                print("✗")
    
    # Step 3: Fallback - mark as failed but try best guess
    classification['tier1'] = 'Equities'  # Default to Equities
    classification['tier2'] = 'Country/Regional'
    classification['tier3_tags'] = ['Equity', 'Global']
    classification['classification_source'] = 'failed'
    if verbose:
        print(f"    → Classification failed, using default")
    
    return classification


def ingest_portfolio(portfolio_id: str, file_path: str, 
                     portfolio_name: str = None, client_name: str = None,
                     verbose: bool = True) -> dict:
    """
    Main ingestion function.
    
    Args:
        portfolio_id: Unique portfolio identifier
        file_path: Path to portfolio file
        portfolio_name: Display name for portfolio
        client_name: Client name (optional)
        verbose: Print progress
        
    Returns:
        Dict with ingestion results
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("PORTFOLIO INGESTION")
        print("=" * 70)
        print(f"\nPortfolio ID: {portfolio_id}")
        print(f"File: {file_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load file
    if verbose:
        print("\n[1] Loading portfolio file...")
    df = load_portfolio_file(file_path)
    
    if verbose:
        print(f"    Found {len(df)} rows")
        print(f"    Columns: {list(df.columns)}")
    
    # Create portfolio record
    if portfolio_name is None:
        portfolio_name = portfolio_id
    
    create_portfolio(portfolio_id, portfolio_name, client_name)
    if verbose:
        print(f"\n[2] Created portfolio: {portfolio_name}")
    
    # Process each holding
    if verbose:
        print(f"\n[3] Processing {len(df)} holdings...")
    
    holdings = []
    resolved = 0
    failed = 0
    classified_haiku = 0
    classified_mapped = 0
    classified_final1000 = 0
    
    for idx, row in df.iterrows():
        symbol = str(row['symbol']).strip().upper()
        quantity = float(row['quantity'])
        market_value = float(row['market_value']) if pd.notna(row.get('market_value')) else None
        avg_price = float(row['avg_price']) if pd.notna(row.get('avg_price')) else None
        open_pnl = float(row['open_pnl']) if pd.notna(row.get('open_pnl')) else None
        
        # Determine position type
        position_type = determine_position_type(row)
        
        if verbose:
            print(f"\n  [{idx+1}/{len(df)}] {symbol} ({position_type})")
        
        # Resolve ticker
        yf_ticker, yf_raw_info = resolve_ticker(symbol)
        
        if yf_ticker is None:
            if verbose:
                print(f"    ✗ Failed to resolve ticker")
            holdings.append({
                'symbol': symbol,
                'position_type': position_type,
                'quantity': quantity,
                'market_value': market_value,
                'avg_price': avg_price,
                'open_pnl': open_pnl,
                'resolution_status': 'failed',
                'resolution_error': 'Ticker not found in Yahoo Finance',
            })
            failed += 1
            continue
        
        # Extract yfinance info
        yf_info = {
            'yf_ticker': yf_ticker,
            'name': yf_raw_info.get('longName') or yf_raw_info.get('shortName', ''),
            'security_type': yf_raw_info.get('quoteType', 'UNKNOWN'),
            'sector': yf_raw_info.get('sector'),
            'industry': yf_raw_info.get('industry'),
            'category': yf_raw_info.get('category'),
            'country': yf_raw_info.get('country'),
            'currency': yf_raw_info.get('currency', 'USD'),
        }
        
        if verbose:
            print(f"    ✓ Resolved: {yf_info['security_type']} - {yf_info['name'][:40]}")
        
        # Classify the holding
        classification = classify_holding(symbol, yf_info, verbose)
        
        # Track classification source
        if classification['classification_source'] == 'haiku':
            classified_haiku += 1
        elif classification['classification_source'] == 'yfinance_mapped':
            classified_mapped += 1
        elif classification['classification_source'] == 'final1000':
            classified_final1000 += 1
        
        # Build holding record
        holding = {
            'symbol': symbol,
            'position_type': position_type,
            'quantity': quantity,
            'market_value': market_value,
            'avg_price': avg_price,
            'open_pnl': open_pnl,
            'yf_ticker': yf_ticker,
            'security_name': yf_info['name'],
            'security_type': yf_info['security_type'],
            'yf_sector': yf_info['sector'],
            'yf_industry': yf_info['industry'],
            'yf_category': yf_info['category'],
            'country': yf_info['country'],
            'currency': yf_info['currency'],
            'tier1': classification['tier1'],
            'tier2': classification['tier2'],
            'tier3_tags': classification['tier3_tags'],
            'final1000_ticker': classification.get('final1000_ticker'),
            'classification_source': classification['classification_source'],
            'resolution_status': 'resolved' if classification['classification_source'] != 'failed' else 'failed',
        }
        
        holdings.append(holding)
        resolved += 1
        
        # Rate limiting for yfinance
        time.sleep(0.2)
    
    # Aggregate duplicate symbol+position_type combinations
    if verbose:
        print(f"\n[4] Aggregating {len(holdings)} holdings...")
    
    aggregated = {}
    for h in holdings:
        key = (h['symbol'], h['position_type'])
        if key in aggregated:
            # Combine quantities, values, and P&L
            existing = aggregated[key]
            existing['quantity'] += h['quantity'] or 0
            existing['market_value'] = (existing['market_value'] or 0) + (h['market_value'] or 0)
            existing['open_pnl'] = (existing['open_pnl'] or 0) + (h['open_pnl'] or 0)
            # Weighted average price
            if existing['quantity'] and h.get('avg_price') and existing.get('avg_price'):
                old_qty = existing['quantity'] - (h['quantity'] or 0)
                new_qty = h['quantity'] or 0
                if old_qty + new_qty > 0:
                    existing['avg_price'] = (existing['avg_price'] * old_qty + h['avg_price'] * new_qty) / (old_qty + new_qty)
        else:
            aggregated[key] = h.copy()
    
    holdings = list(aggregated.values())
    
    if verbose:
        print(f"    Aggregated to {len(holdings)} unique positions")
    
    # Save all holdings
    if verbose:
        print(f"\n[5] Saving {len(holdings)} holdings to database...")
    
    save_holdings(portfolio_id, holdings)
    
    # Get summary
    summary = get_holdings_summary(portfolio_id)
    
    elapsed = time.time() - start_time
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("INGESTION COMPLETE")
        print("=" * 70)
        print(f"\nTotal holdings: {len(holdings)}")
        print(f"  Resolved: {resolved}")
        print(f"  Failed: {failed}")
        print(f"\nClassification breakdown:")
        print(f"  Final 1000 matches: {classified_final1000}")
        print(f"  YFinance mapped: {classified_mapped}")
        print(f"  Haiku classified: {classified_haiku}")
        print(f"\nPosition breakdown:")
        print(f"  Long positions: {summary.get('long_count', 0)}")
        print(f"  Short positions: {summary.get('short_count', 0)}")
        print(f"  Long value: ${summary.get('long_value', 0):,.2f}")
        print(f"  Short value: ${summary.get('short_value', 0):,.2f}")
        print(f"\nTime elapsed: {elapsed:.1f} seconds")
    
    return {
        'portfolio_id': portfolio_id,
        'total_holdings': len(holdings),
        'resolved': resolved,
        'failed': failed,
        'classified_final1000': classified_final1000,
        'classified_mapped': classified_mapped,
        'classified_haiku': classified_haiku,
        'summary': summary,
        'elapsed_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Ingest portfolio from Excel/CSV file'
    )
    parser.add_argument('--portfolio', required=True, 
                        help='Portfolio ID (unique identifier)')
    parser.add_argument('--file', required=True,
                        help='Path to portfolio file (Excel or CSV)')
    parser.add_argument('--name', default=None,
                        help='Portfolio display name (defaults to portfolio ID)')
    parser.add_argument('--client', default=None,
                        help='Client name')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        result = ingest_portfolio(
            portfolio_id=args.portfolio,
            file_path=args.file,
            portfolio_name=args.name,
            client_name=args.client,
            verbose=not args.quiet
        )
        
        # Exit with error if too many failures
        if result['failed'] > result['resolved'] * 0.2:  # More than 20% failed
            print(f"\nWARNING: {result['failed']} holdings failed resolution")
            sys.exit(1)
        
        print("\n✓ Ingestion successful")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
