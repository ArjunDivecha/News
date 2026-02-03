#!/usr/bin/env python3
"""
=============================================================================
DAILY PRICE FETCHER - Phase 2 Portfolio Reports
=============================================================================

Fetches daily prices from Yahoo Finance for all holdings in a portfolio,
computes returns, weights, and contributions, and saves to database.

USAGE:
    python 02_fetch_daily_prices.py --portfolio TEST --date 2026-01-31
    python 02_fetch_daily_prices.py --portfolio TEST  # Uses latest trading day

OUTPUT:
    - Daily snapshot in portfolio_daily table
    - Portfolio summary in portfolio_summary table
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import (
    get_db, get_holdings, save_daily_snapshot, save_portfolio_summary,
    get_portfolio
)
from utils.yfinance_utils import get_prices_for_date


def get_last_trading_day(target_date: str = None) -> str:
    """
    Get the last trading day (weekday) for the target date.
    If today is weekend, walk back to Friday.
    """
    if target_date:
        dt = datetime.strptime(target_date, '%Y-%m-%d')
    else:
        dt = datetime.now()
    
    # Walk back if weekend
    while dt.weekday() > 4:  # Saturday = 5, Sunday = 6
        dt = dt - timedelta(days=1)
    
    return dt.strftime('%Y-%m-%d')


def fetch_daily_prices(portfolio_id: str, date: str = None, 
                       verbose: bool = True) -> dict:
    """
    Fetch prices and compute portfolio metrics for a specific date.
    
    Args:
        portfolio_id: Portfolio identifier
        date: Target date (YYYY-MM-DD), defaults to last trading day
        verbose: Print progress
        
    Returns:
        Dict with fetch results and portfolio summary
    """
    start_time = time.time()
    
    # Determine target date
    if date is None:
        date = get_last_trading_day()
    else:
        date = get_last_trading_day(date)
    
    if verbose:
        print("=" * 70)
        print("DAILY PRICE FETCH")
        print("=" * 70)
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check portfolio exists
    portfolio = get_portfolio(portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio not found: {portfolio_id}")
    
    # Get holdings
    holdings = get_holdings(portfolio_id, resolved_only=True)
    if holdings.empty:
        raise ValueError(f"No resolved holdings found for portfolio: {portfolio_id}")
    
    if verbose:
        print(f"\n[1] Found {len(holdings)} resolved holdings")
    
    # Get unique tickers (same symbol may appear twice for LONG and SHORT)
    tickers = holdings['yf_ticker'].dropna().unique().tolist()
    
    if verbose:
        print(f"[2] Fetching prices for {len(tickers)} unique tickers...")
    
    # Fetch prices
    price_data = get_prices_for_date(tickers, date)
    
    # Process each holding
    daily_data = []
    
    # For portfolio calculations
    total_long_value = 0
    total_short_value = 0
    
    for _, row in holdings.iterrows():
        yf_ticker = row['yf_ticker']
        symbol = row['symbol']
        holding_id = row['id']
        position_type = row['position_type']
        quantity = row['quantity']
        
        # Get price data for this ticker
        pdata = price_data.get(yf_ticker, {})
        
        if pdata.get('status') == 'success':
            price = pdata['price']
            return_1d = pdata.get('return_1d')
            return_ytd = pdata.get('return_ytd')
            fetch_status = 'success'
        else:
            # Use market value from holdings if price fetch failed
            if row['market_value'] and row['quantity']:
                price = abs(row['market_value'] / row['quantity'])
            else:
                price = None
            return_1d = None
            return_ytd = None
            fetch_status = 'failed'
            if verbose:
                print(f"  ⚠️  {symbol}: price fetch failed")
        
        # Calculate market value
        if price is not None and quantity:
            market_value = quantity * price  # Will be negative for SHORT positions
        else:
            market_value = row['market_value']  # Use value from holdings file
        
        # Calculate cost basis and P&L
        avg_price = row['avg_price']
        if avg_price and quantity:
            cost_basis = abs(quantity) * avg_price  # Always positive
        else:
            cost_basis = None
        
        # Open P&L  
        if row['open_pnl'] is not None:
            open_pnl = row['open_pnl']
        elif market_value and cost_basis:
            if position_type == 'LONG':
                open_pnl = market_value - cost_basis
            else:
                open_pnl = cost_basis - abs(market_value)  # For shorts, profit when price drops
        else:
            open_pnl = None
        
        # P&L percentage
        if open_pnl is not None and cost_basis and cost_basis > 0:
            open_pnl_pct = (open_pnl / cost_basis) * 100
        else:
            open_pnl_pct = None
        
        # Track totals for weight calculation
        if market_value is not None:
            if position_type == 'LONG':
                total_long_value += market_value
            else:
                total_short_value += market_value  # Will be negative
        
        daily_data.append({
            'holding_id': holding_id,
            'symbol': symbol,
            'position_type': position_type,
            'quantity': quantity,
            'price': price,
            'market_value_usd': market_value,
            'avg_price': avg_price,
            'cost_basis': cost_basis,
            'open_pnl': open_pnl,
            'open_pnl_pct': open_pnl_pct,
            'daily_pnl': None,  # Would need previous day data
            'return_1d': return_1d,
            'return_ytd': return_ytd,
            'fetch_status': fetch_status,
        })
    
    # Calculate weights and contributions
    gross_exposure = total_long_value + abs(total_short_value)
    net_exposure = total_long_value + total_short_value
    
    if verbose:
        print(f"\n[3] Calculating weights and contributions...")
        print(f"    Long value: ${total_long_value:,.2f}")
        print(f"    Short value: ${total_short_value:,.2f}")
        print(f"    Gross exposure: ${gross_exposure:,.2f}")
    
    # Calculate weights (using gross exposure as denominator)
    portfolio_return = 0
    top_contributors = []
    top_detractors = []
    
    for d in daily_data:
        mv = d['market_value_usd']
        if mv is not None and gross_exposure > 0:
            # Weight is signed (negative for shorts)
            d['weight'] = mv / gross_exposure
            
            # Contribution = weight × return (in basis points)
            if d['return_1d'] is not None:
                # For shorts, positive return = loss (so we flip the sign)
                if d['position_type'] == 'SHORT':
                    contribution = -d['weight'] * d['return_1d'] * 100  # in bps
                else:
                    contribution = d['weight'] * d['return_1d'] * 100  # in bps
                d['contribution_1d'] = contribution
                portfolio_return += contribution
                
                # Track top contributors/detractors
                if contribution >= 0:
                    top_contributors.append({
                        'symbol': d['symbol'],
                        'position_type': d['position_type'],
                        'contribution': contribution,
                        'return_1d': d['return_1d'],
                        'weight': abs(d['weight']),
                    })
                else:
                    top_detractors.append({
                        'symbol': d['symbol'],
                        'position_type': d['position_type'],
                        'contribution': contribution,
                        'return_1d': d['return_1d'],
                        'weight': abs(d['weight']),
                    })
            else:
                d['contribution_1d'] = None
        else:
            d['weight'] = None
            d['contribution_1d'] = None
    
    # Sort and limit top contributors/detractors
    top_contributors = sorted(top_contributors, key=lambda x: x['contribution'], reverse=True)[:10]
    top_detractors = sorted(top_detractors, key=lambda x: x['contribution'])[:10]
    
    # Save daily snapshot
    if verbose:
        print(f"\n[4] Saving daily snapshot ({len(daily_data)} records)...")
    
    save_daily_snapshot(portfolio_id, date, daily_data)
    
    # Calculate and save portfolio summary
    success_count = sum(1 for d in daily_data if d['fetch_status'] == 'success')
    failed_count = sum(1 for d in daily_data if d['fetch_status'] == 'failed')
    long_count = sum(1 for d in daily_data if d['position_type'] == 'LONG')
    short_count = sum(1 for d in daily_data if d['position_type'] == 'SHORT')
    total_open_pnl = sum(d['open_pnl'] or 0 for d in daily_data)
    
    # Compute portfolio-level YTD return (weighted average)
    portfolio_return_ytd = 0
    total_weight_ytd = 0
    for d in daily_data:
        if d.get('weight') is not None and d.get('return_ytd') is not None:
            weight = abs(d['weight'])  # Use absolute weight for YTD calculation
            portfolio_return_ytd += weight * d['return_ytd']
            total_weight_ytd += weight
    
    if total_weight_ytd > 0:
        portfolio_return_ytd = portfolio_return_ytd / total_weight_ytd
    else:
        portfolio_return_ytd = None
    
    summary = {
        'total_market_value': total_long_value + total_short_value,
        'total_long_value': total_long_value,
        'total_short_value': total_short_value,
        'net_exposure': net_exposure,
        'gross_exposure': gross_exposure,
        'holding_count': len(daily_data),
        'long_count': long_count,
        'short_count': short_count,
        'portfolio_return_1d': portfolio_return / 100,  # Convert bps to %
        'portfolio_return_ytd': portfolio_return_ytd,
        'total_open_pnl': total_open_pnl,
        'daily_pnl': None,
        'top_contributors': top_contributors,
        'top_detractors': top_detractors,
    }
    
    save_portfolio_summary(portfolio_id, date, summary)
    
    elapsed = time.time() - start_time
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("FETCH COMPLETE")
        print("=" * 70)
        print(f"\nDate: {date}")
        print(f"Holdings processed: {len(daily_data)}")
        print(f"  Prices fetched: {success_count}")
        print(f"  Fetch failed: {failed_count}")
        print(f"\nPortfolio Summary:")
        print(f"  Portfolio return (1d): {summary['portfolio_return_1d']:.2f}%")
        print(f"  Total P&L: ${total_open_pnl:,.2f}")
        print(f"\nTop 5 Contributors:")
        for c in top_contributors[:5]:
            print(f"  {c['symbol']:6s} ({c['position_type']:5s}): {c['contribution']:+.1f}bps  (ret: {c['return_1d']:+.2f}%)")
        print(f"\nTop 5 Detractors:")
        for d in top_detractors[:5]:
            print(f"  {d['symbol']:6s} ({d['position_type']:5s}): {d['contribution']:+.1f}bps  (ret: {d['return_1d']:+.2f}%)")
        print(f"\nTime elapsed: {elapsed:.1f} seconds")
    
    return {
        'date': date,
        'holdings_processed': len(daily_data),
        'prices_fetched': success_count,
        'fetch_failed': failed_count,
        'portfolio_return_1d': summary['portfolio_return_1d'],
        'total_open_pnl': total_open_pnl,
        'top_contributors': top_contributors[:5],
        'top_detractors': top_detractors[:5],
        'elapsed_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fetch daily prices for portfolio holdings'
    )
    parser.add_argument('--portfolio', required=True,
                        help='Portfolio ID')
    parser.add_argument('--date', default=None,
                        help='Target date (YYYY-MM-DD), defaults to last trading day')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        result = fetch_daily_prices(
            portfolio_id=args.portfolio,
            date=args.date,
            verbose=not args.quiet
        )
        
        if result['fetch_failed'] > result['holdings_processed'] * 0.2:
            print(f"\nWARNING: {result['fetch_failed']} price fetches failed")
        
        print("\n✓ Price fetch successful")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
