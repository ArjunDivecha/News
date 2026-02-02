#!/usr/bin/env python3
"""
=============================================================================
PORTFOLIO PIPELINE ORCHESTRATOR - Phase 2 Portfolio Reports
=============================================================================

Single entry point for the portfolio report generation pipeline.
Orchestrates: ingestion → price fetch → analytics → report generation

USAGE:
    # Full pipeline for existing portfolio
    python run_pipeline.py --portfolio TEST --date 2026-01-31
    
    # Ingest new portfolio and run pipeline
    python run_pipeline.py --portfolio NEW --file holdings.xlsx --date 2026-01-31
    
    # Just update holdings
    python run_pipeline.py --portfolio TEST --update-holdings new_holdings.xlsx
    
    # List available portfolios
    python run_pipeline.py --list

=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import get_db, get_portfolio, list_portfolios, get_holdings


def get_last_trading_day(target_date: str = None) -> str:
    """Get the last trading day (weekday)."""
    if target_date:
        dt = datetime.strptime(target_date, '%Y-%m-%d')
    else:
        dt = datetime.now()
    
    # Walk back if weekend
    while dt.weekday() > 4:
        dt = dt - timedelta(days=1)
    
    return dt.strftime('%Y-%m-%d')


def run_full_pipeline(portfolio_id: str, date: str = None, 
                      file_path: str = None, portfolio_name: str = None,
                      skip_ingest: bool = False,
                      verbose: bool = True) -> dict:
    """
    Run the full portfolio report pipeline.
    
    Steps:
    1. Ingest portfolio (if file provided or portfolio doesn't exist)
    2. Fetch daily prices
    3. Compute analytics
    4. Generate report
    
    Args:
        portfolio_id: Portfolio identifier
        date: Target date (defaults to last trading day)
        file_path: Path to portfolio file (optional, for new/update)
        portfolio_name: Display name for portfolio (optional)
        skip_ingest: Skip ingestion even if file provided
        verbose: Print progress
        
    Returns:
        Dict with pipeline results
    """
    start_time = time.time()
    results = {'steps': []}
    
    # Determine target date
    if date is None:
        date = get_last_trading_day()
    else:
        date = get_last_trading_day(date)
    
    if verbose:
        print("=" * 70)
        print("PORTFOLIO REPORT PIPELINE")
        print("=" * 70)
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    # Step 1: Check if ingestion needed
    portfolio = get_portfolio(portfolio_id)
    need_ingest = False
    
    if file_path and not skip_ingest:
        need_ingest = True
    elif not portfolio:
        if file_path:
            need_ingest = True
        else:
            print(f"\n✗ Portfolio '{portfolio_id}' not found and no file provided")
            print("  Use --file to provide a portfolio holdings file")
            sys.exit(1)
    
    # Step 1: Ingest Portfolio
    if need_ingest:
        if verbose:
            print(f"\n{'='*70}")
            print("STEP 1: PORTFOLIO INGESTION")
            print(f"{'='*70}")
        
        # Import the module with importlib to handle numeric prefix
        import importlib.util
        spec = importlib.util.spec_from_file_location("ingest", Path(__file__).parent / "01_ingest_portfolio.py")
        ingest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ingest_module)
        
        ingest_result = ingest_module.ingest_portfolio(
            portfolio_id=portfolio_id,
            file_path=file_path,
            portfolio_name=portfolio_name,
            verbose=verbose
        )
        
        results['steps'].append({
            'step': 'ingest',
            'status': 'success' if ingest_result['failed'] == 0 else 'partial',
            'holdings': ingest_result['total_holdings'],
            'resolved': ingest_result['resolved'],
            'failed': ingest_result['failed'],
        })
        
        if ingest_result['failed'] > ingest_result['resolved'] * 0.1:
            print(f"\n⚠️  Warning: {ingest_result['failed']} holdings failed resolution")
    else:
        if verbose:
            print(f"\n[1] Using existing portfolio: {portfolio.get('portfolio_name')}")
        results['steps'].append({
            'step': 'ingest',
            'status': 'skipped',
            'reason': 'portfolio exists',
        })
    
    # Check holdings exist
    holdings = get_holdings(portfolio_id, resolved_only=True)
    if holdings.empty:
        print(f"\n✗ No resolved holdings found for portfolio: {portfolio_id}")
        sys.exit(1)
    
    # Step 2: Fetch Daily Prices
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 2: DAILY PRICE FETCH")
        print(f"{'='*70}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("prices", Path(__file__).parent / "02_fetch_daily_prices.py")
    prices_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prices_module)
    
    price_result = prices_module.fetch_daily_prices(
        portfolio_id=portfolio_id,
        date=date,
        verbose=verbose
    )
    
    results['steps'].append({
        'step': 'prices',
        'status': 'success' if price_result['fetch_failed'] < price_result['holdings_processed'] * 0.1 else 'partial',
        'fetched': price_result['prices_fetched'],
        'failed': price_result['fetch_failed'],
        'portfolio_return': price_result['portfolio_return_1d'],
    })
    
    # Step 3: Compute Analytics
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 3: ANALYTICS COMPUTATION")
        print(f"{'='*70}")
    
    spec = importlib.util.spec_from_file_location("analytics", Path(__file__).parent / "03_compute_analytics.py")
    analytics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analytics_module)
    
    analytics_result = analytics_module.compute_aggregates(
        portfolio_id=portfolio_id,
        date=date,
        verbose=verbose
    )
    
    results['steps'].append({
        'step': 'analytics',
        'status': 'success',
        'aggregates': analytics_result['total_aggregates'],
    })
    
    # Step 4: Generate Report
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 4: REPORT GENERATION")
        print(f"{'='*70}")
    
    spec = importlib.util.spec_from_file_location("report", Path(__file__).parent / "04_generate_report.py")
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    
    report_result = report_module.generate_portfolio_report(
        portfolio_id=portfolio_id,
        date=date,
        verbose=verbose
    )
    
    results['steps'].append({
        'step': 'report',
        'status': 'success',
        'report_id': report_result['report_id'],
        'md_path': report_result['md_path'],
        'pdf_path': report_result['pdf_path'],
    })
    
    elapsed = time.time() - start_time
    results['elapsed_seconds'] = elapsed
    results['date'] = date
    results['portfolio_id'] = portfolio_id
    
    # Final summary
    if verbose:
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"\n✓ All steps completed successfully!")
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
        print(f"Return: {price_result['portfolio_return_1d']:+.2f}%")
        print(f"\nOutput files:")
        print(f"  Markdown: {report_result['md_path']}")
        print(f"  PDF: {report_result['pdf_path']}")
        print(f"\nTotal time: {elapsed:.1f} seconds")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio Report Pipeline Orchestrator'
    )
    parser.add_argument('--portfolio', 
                        help='Portfolio ID')
    parser.add_argument('--date',
                        help='Target date (YYYY-MM-DD), defaults to last trading day')
    parser.add_argument('--file',
                        help='Path to portfolio file (Excel or CSV) for ingestion')
    parser.add_argument('--name',
                        help='Portfolio display name')
    parser.add_argument('--update-holdings',
                        help='Update existing portfolio with new holdings file')
    parser.add_argument('--list', action='store_true',
                        help='List available portfolios')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        portfolios = list_portfolios()
        if portfolios:
            print("\nAvailable Portfolios:")
            print("-" * 50)
            for p in portfolios:
                print(f"  {p['portfolio_id']:<15} {p['portfolio_name']}")
        else:
            print("\nNo portfolios found. Create one with:")
            print("  python run_pipeline.py --portfolio ID --file holdings.xlsx")
        sys.exit(0)
    
    # Require portfolio ID for other operations
    if not args.portfolio:
        parser.print_help()
        print("\nError: --portfolio is required")
        sys.exit(1)
    
    # Handle --update-holdings
    if args.update_holdings:
        print(f"Updating holdings for {args.portfolio}...")
        import importlib.util
        spec = importlib.util.spec_from_file_location("ingest", Path(__file__).parent / "01_ingest_portfolio.py")
        ingest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ingest_module)
        result = ingest_module.ingest_portfolio(
            portfolio_id=args.portfolio,
            file_path=args.update_holdings,
            verbose=not args.quiet
        )
        print("\n✓ Holdings updated")
        sys.exit(0)
    
    # Run full pipeline
    try:
        result = run_full_pipeline(
            portfolio_id=args.portfolio,
            date=args.date,
            file_path=args.file,
            portfolio_name=args.name,
            verbose=not args.quiet
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
