#!/usr/bin/env python3
"""
=============================================================================
ANALYTICS COMPUTATION - Phase 2 Portfolio Reports
=============================================================================

Computes portfolio aggregates by sector, region, tier1, tier2, and tier3 tags.

USAGE:
    python 03_compute_analytics.py --portfolio TEST --date 2026-01-31

OUTPUT:
    - Aggregates in portfolio_aggregates table
=============================================================================
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import (
    get_db, get_daily_snapshot, save_aggregates, get_portfolio,
    get_portfolio_summary
)


def compute_aggregates(portfolio_id: str, date: str = None,
                       verbose: bool = True) -> dict:
    """
    Compute portfolio aggregates by various dimensions.
    
    Args:
        portfolio_id: Portfolio identifier
        date: Target date (YYYY-MM-DD)
        verbose: Print progress
        
    Returns:
        Dict with aggregation results
    """
    if verbose:
        print("=" * 70)
        print("ANALYTICS COMPUTATION")
        print("=" * 70)
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
    
    # Check portfolio exists
    portfolio = get_portfolio(portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio not found: {portfolio_id}")
    
    # Get daily snapshot
    snapshot = get_daily_snapshot(portfolio_id, date)
    if snapshot.empty:
        raise ValueError(f"No daily data found for {portfolio_id} on {date}")
    
    if verbose:
        print(f"\n[1] Loaded {len(snapshot)} holdings from daily snapshot")
    
    # Parse tier3_tags from JSON
    def parse_tags(tags_str):
        if not tags_str:
            return []
        try:
            if isinstance(tags_str, list):
                return tags_str
            return json.loads(tags_str)
        except:
            return []
    
    snapshot['tier3_list'] = snapshot['tier3_tags'].apply(parse_tags)
    
    # Dimensions to aggregate by
    dimensions = [
        ('tier1', lambda row: row['tier1']),
        ('tier2', lambda row: row['tier2']),
        ('country', lambda row: row['country']),
        ('sector', lambda row: next((t for t in row['tier3_list'] 
            if t in ['Tech', 'Energy', 'Financials', 'Healthcare', 'Industrials', 
                     'Consumer', 'Defensive', 'Materials', 'Utilities', 'Real Estate']), None)),
        ('region', lambda row: next((t for t in row['tier3_list']
            if t in ['US', 'Europe', 'Asia', 'EM', 'Global', 'China', 'Japan', 
                     'India', 'Canada', 'APAC', 'Australia']), None)),
    ]
    
    all_aggregates = []
    
    for dim_type, value_func in dimensions:
        if verbose:
            print(f"\n[2] Aggregating by {dim_type}...")
        
        aggregates = defaultdict(lambda: {
            'holding_count': 0,
            'long_count': 0,
            'short_count': 0,
            'total_weight': 0,
            'long_weight': 0,
            'short_weight': 0,
            'total_value': 0,
            'weighted_return': 0,
            'contribution': 0,
        })
        
        for _, row in snapshot.iterrows():
            value = value_func(row)
            if not value:
                value = 'Other'
            
            weight = row['weight'] if pd.notna(row['weight']) else 0
            market_value = row['market_value_usd'] if pd.notna(row['market_value_usd']) else 0
            return_1d = row['return_1d'] if pd.notna(row['return_1d']) else 0
            contribution = row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0
            
            agg = aggregates[value]
            agg['holding_count'] += 1
            agg['total_weight'] += abs(weight)
            agg['total_value'] += market_value
            agg['weighted_return'] += abs(weight) * return_1d
            agg['contribution'] += contribution
            
            if row['position_type'] == 'LONG':
                agg['long_count'] += 1
                agg['long_weight'] += abs(weight)
            else:
                agg['short_count'] += 1
                agg['short_weight'] += abs(weight)
        
        # Normalize weighted return by total weight
        for value, agg in aggregates.items():
            if agg['total_weight'] > 0:
                agg['weighted_return'] = agg['weighted_return'] / agg['total_weight']
            
            all_aggregates.append({
                'dimension_type': dim_type,
                'dimension_value': value,
                'holding_count': agg['holding_count'],
                'long_count': agg['long_count'],
                'short_count': agg['short_count'],
                'total_weight': agg['total_weight'],
                'long_weight': agg['long_weight'],
                'short_weight': agg['short_weight'],
                'total_value_usd': agg['total_value'],
                'weighted_return_1d': agg['weighted_return'],
                'contribution_1d': agg['contribution'],
            })
    
    # Also aggregate by each tier3 tag
    if verbose:
        print(f"\n[3] Aggregating by tier3 tags...")
    
    tag_aggregates = defaultdict(lambda: {
        'holding_count': 0,
        'long_count': 0,
        'short_count': 0,
        'total_weight': 0,
        'long_weight': 0,
        'short_weight': 0,
        'total_value': 0,
        'weighted_return': 0,
        'contribution': 0,
    })
    
    for _, row in snapshot.iterrows():
        tags = row['tier3_list']
        weight = row['weight'] if pd.notna(row['weight']) else 0
        market_value = row['market_value_usd'] if pd.notna(row['market_value_usd']) else 0
        return_1d = row['return_1d'] if pd.notna(row['return_1d']) else 0
        contribution = row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0
        
        for tag in tags:
            agg = tag_aggregates[tag]
            agg['holding_count'] += 1
            agg['total_weight'] += abs(weight)
            agg['total_value'] += market_value
            agg['weighted_return'] += abs(weight) * return_1d
            agg['contribution'] += contribution
            
            if row['position_type'] == 'LONG':
                agg['long_count'] += 1
                agg['long_weight'] += abs(weight)
            else:
                agg['short_count'] += 1
                agg['short_weight'] += abs(weight)
    
    for tag, agg in tag_aggregates.items():
        if agg['total_weight'] > 0:
            agg['weighted_return'] = agg['weighted_return'] / agg['total_weight']
        
        all_aggregates.append({
            'dimension_type': 'tier3_tag',
            'dimension_value': tag,
            'holding_count': agg['holding_count'],
            'long_count': agg['long_count'],
            'short_count': agg['short_count'],
            'total_weight': agg['total_weight'],
            'long_weight': agg['long_weight'],
            'short_weight': agg['short_weight'],
            'total_value_usd': agg['total_value'],
            'weighted_return_1d': agg['weighted_return'],
            'contribution_1d': agg['contribution'],
        })
    
    # Save aggregates
    if verbose:
        print(f"\n[4] Saving {len(all_aggregates)} aggregates...")
    
    save_aggregates(portfolio_id, date, all_aggregates)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("AGGREGATION COMPLETE")
        print("=" * 70)
        
        # Print tier1 breakdown
        print("\nTier-1 Breakdown:")
        tier1_aggs = [a for a in all_aggregates if a['dimension_type'] == 'tier1']
        tier1_aggs.sort(key=lambda x: x['total_weight'], reverse=True)
        print(f"  {'Category':<30} {'Weight':>8} {'Return':>8} {'Contrib':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for agg in tier1_aggs:
            print(f"  {agg['dimension_value']:<30} {agg['total_weight']*100:>7.1f}% {agg['weighted_return_1d']:>+7.2f}% {agg['contribution_1d']:>+7.1f}bp")
        
        # Print region breakdown
        print("\nRegion Breakdown:")
        region_aggs = [a for a in all_aggregates if a['dimension_type'] == 'region']
        region_aggs.sort(key=lambda x: x['total_weight'], reverse=True)
        print(f"  {'Region':<15} {'Weight':>8} {'Holdings':>8} {'Return':>8} {'Contrib':>8}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for agg in region_aggs[:10]:
            print(f"  {agg['dimension_value']:<15} {agg['total_weight']*100:>7.1f}% {agg['holding_count']:>8} {agg['weighted_return_1d']:>+7.2f}% {agg['contribution_1d']:>+7.1f}bp")
    
    return {
        'date': date,
        'total_aggregates': len(all_aggregates),
        'dimensions': list(set(a['dimension_type'] for a in all_aggregates)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute portfolio analytics and aggregates'
    )
    parser.add_argument('--portfolio', required=True,
                        help='Portfolio ID')
    parser.add_argument('--date', required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        result = compute_aggregates(
            portfolio_id=args.portfolio,
            date=args.date,
            verbose=not args.quiet
        )
        
        print("\n✓ Analytics computation successful")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
