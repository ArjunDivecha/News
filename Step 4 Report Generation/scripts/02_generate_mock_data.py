#!/usr/bin/env python3
"""
=============================================================================
MOCK DATA GENERATOR
=============================================================================

INPUT FILES:
- database/market_data.db (assets table)

OUTPUT FILES:
- Populates daily_prices, category_stats, factor_returns tables in database

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Generate realistic mock Bloomberg data for testing the report generation
pipeline without needing actual Bloomberg access. Creates:
- Daily price data with realistic returns by asset class
- Category-level statistics
- Factor returns for beta attribution

USAGE:
    python scripts/02_generate_mock_data.py                    # Generate 1 day
    python scripts/02_generate_mock_data.py --days 30          # Generate 30 days
    python scripts/02_generate_mock_data.py --scenario risk_off # Risk-off scenario
=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.db import get_db, get_assets, save_daily_prices, save_category_stats, save_factor_returns

# Database path
DB_PATH = Path(__file__).parent.parent / "database" / "market_data.db"

# Realistic return distributions by Tier-1 (mean, std in daily % terms)
TIER1_RETURN_PARAMS = {
    'Equities': {'mean': 0.04, 'std': 1.2, 'skew': -0.3},
    'Fixed Income': {'mean': 0.01, 'std': 0.4, 'skew': 0.0},
    'Commodities': {'mean': 0.02, 'std': 1.5, 'skew': 0.2},
    'Currencies (FX)': {'mean': 0.00, 'std': 0.6, 'skew': 0.0},
    'Multi-Asset / Thematic': {'mean': 0.03, 'std': 0.9, 'skew': -0.1},
    'Volatility / Risk Premia': {'mean': 0.01, 'std': 2.5, 'skew': 1.0},
    'Alternative / Synthetic': {'mean': 0.02, 'std': 1.0, 'skew': 0.0},
}

# Factor return distributions
FACTOR_PARAMS = {
    'SPX': {'mean': 0.04, 'std': 1.0},
    'Russell2000': {'mean': 0.05, 'std': 1.4},
    'Nasdaq100': {'mean': 0.05, 'std': 1.3},
    'Value': {'mean': 0.03, 'std': 0.8},
    'Growth': {'mean': 0.04, 'std': 1.2},
    'EAFE': {'mean': 0.03, 'std': 1.1},
    'EM': {'mean': 0.04, 'std': 1.5},
    'HYCredit': {'mean': 0.02, 'std': 0.5},
    'Treasuries': {'mean': 0.01, 'std': 0.3},
    'TIPS': {'mean': 0.01, 'std': 0.4},
    'Commodities': {'mean': 0.02, 'std': 1.5},
    'Agriculture': {'mean': 0.01, 'std': 1.2},
    'Crypto': {'mean': 0.10, 'std': 4.0},
    'REIT_US': {'mean': 0.03, 'std': 1.2},
    'REIT_Global': {'mean': 0.03, 'std': 1.3},
}

# Scenarios (multipliers to apply to means)
SCENARIOS = {
    'normal': {'equity_mult': 1.0, 'bond_mult': 1.0, 'vol_mult': 1.0},
    'risk_on': {'equity_mult': 2.0, 'bond_mult': -0.5, 'vol_mult': 0.8},
    'risk_off': {'equity_mult': -2.0, 'bond_mult': 2.0, 'vol_mult': 1.5},
    'vol_spike': {'equity_mult': -1.5, 'bond_mult': 1.5, 'vol_mult': 3.0},
    'rotation': {'equity_mult': 0.5, 'bond_mult': 0.5, 'vol_mult': 1.2},
}


def generate_factor_returns(scenario: str = 'normal') -> Dict[str, float]:
    """Generate factor returns for a given scenario."""
    params = SCENARIOS.get(scenario, SCENARIOS['normal'])
    
    returns = {}
    for factor, fp in FACTOR_PARAMS.items():
        # Apply scenario multipliers
        mean = fp['mean']
        if factor in ['SPX', 'Russell2000', 'Nasdaq100', 'Value', 'Growth', 'EAFE', 'EM']:
            mean *= params['equity_mult']
        elif factor in ['Treasuries', 'TIPS', 'HYCredit']:
            mean *= params['bond_mult']
        
        std = fp['std'] * params['vol_mult']
        returns[factor] = np.random.normal(mean, std)
    
    return returns


def generate_asset_returns(assets: pd.DataFrame, factor_returns: Dict[str, float],
                          scenario: str = 'normal') -> pd.DataFrame:
    """
    Generate asset returns based on beta exposures and idiosyncratic noise.
    
    Returns = sum(beta_i * factor_i) + idiosyncratic_noise
    """
    params = SCENARIOS.get(scenario, SCENARIOS['normal'])
    
    results = []
    for _, asset in assets.iterrows():
        tier1 = asset['tier1']
        tier1_params = TIER1_RETURN_PARAMS.get(tier1, TIER1_RETURN_PARAMS['Equities'])
        
        # Beta-driven return
        beta_return = 0.0
        
        # Map beta columns to factor names
        beta_map = {
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
        
        for beta_col, factor_name in beta_map.items():
            beta = asset.get(beta_col)
            if pd.notna(beta) and factor_name in factor_returns:
                try:
                    beta_float = float(beta)
                    beta_return += beta_float * factor_returns[factor_name]
                except (TypeError, ValueError):
                    pass  # Skip non-numeric betas
        
        # Idiosyncratic component
        idio_std = tier1_params['std'] * 0.5 * params['vol_mult']
        idio_return = np.random.normal(0, idio_std)
        
        # Total return
        total_return = beta_return + idio_return
        
        # Add some skewness for certain asset classes
        skew = tier1_params.get('skew', 0)
        if skew != 0 and np.random.random() < 0.1:
            # Occasional tail event
            total_return += skew * abs(np.random.normal(0, tier1_params['std']))
        
        results.append({
            'ticker': asset['ticker'],
            'return_1d': round(total_return, 4),
            'beta_predicted_return': round(beta_return, 4),
            'alpha_1d': round(idio_return, 4),
            'tier1': tier1,
            'tier2': asset['tier2'],
        })
    
    return pd.DataFrame(results)


def compute_category_stats(returns_df: pd.DataFrame, date: str) -> List[Dict]:
    """Compute aggregated statistics by category."""
    stats = []
    
    # Tier-1 stats
    for tier1, group in returns_df.groupby('tier1'):
        returns = group['return_1d']
        stats.append({
            'date': date,
            'category_type': 'tier1',
            'category_value': tier1,
            'count': len(group),
            'avg_return': round(returns.mean(), 4),
            'median_return': round(returns.median(), 4),
            'std_return': round(returns.std(), 4),
            'min_return': round(returns.min(), 4),
            'max_return': round(returns.max(), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(group) > 0 else None,
            'best_return': round(returns.max(), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(group) > 0 else None,
            'worst_return': round(returns.min(), 4),
            'percentile_60d': np.random.randint(20, 80),  # Mock historical percentile
            'streak_days': np.random.randint(-3, 4),
            'streak_direction': 'positive' if returns.mean() > 0 else 'negative',
        })
    
    # Tier-2 stats
    for tier2, group in returns_df.groupby('tier2'):
        returns = group['return_1d']
        stats.append({
            'date': date,
            'category_type': 'tier2',
            'category_value': tier2,
            'count': len(group),
            'avg_return': round(returns.mean(), 4),
            'median_return': round(returns.median(), 4),
            'std_return': round(returns.std(), 4),
            'min_return': round(returns.min(), 4),
            'max_return': round(returns.max(), 4),
            'best_ticker': group.loc[returns.idxmax(), 'ticker'] if len(group) > 0 else None,
            'best_return': round(returns.max(), 4),
            'worst_ticker': group.loc[returns.idxmin(), 'ticker'] if len(group) > 0 else None,
            'worst_return': round(returns.min(), 4),
            'percentile_60d': np.random.randint(20, 80),
            'streak_days': np.random.randint(-3, 4),
            'streak_direction': 'positive' if returns.mean() > 0 else 'negative',
        })
    
    return stats


def generate_mock_data(date: str, scenario: str = 'normal', verbose: bool = True) -> Dict:
    """
    Generate a complete day of mock data.
    
    Args:
        date: Date string (YYYY-MM-DD)
        scenario: Market scenario ('normal', 'risk_on', 'risk_off', 'vol_spike', 'rotation')
        verbose: Print progress
        
    Returns:
        Dict with generation stats
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING MOCK DATA FOR {date}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")
    
    # Load assets
    if verbose:
        print("\n[1/5] Loading assets...")
    assets = get_assets()
    if verbose:
        print(f"      Loaded {len(assets)} assets")
    
    # Generate factor returns
    if verbose:
        print("\n[2/5] Generating factor returns...")
    factor_returns = generate_factor_returns(scenario)
    if verbose:
        print(f"      Generated {len(factor_returns)} factor returns")
        print(f"      SPX: {factor_returns['SPX']:+.2f}%, Treasuries: {factor_returns['Treasuries']:+.2f}%")
    
    # Generate asset returns
    if verbose:
        print("\n[3/5] Generating asset returns...")
    returns_df = generate_asset_returns(assets, factor_returns, scenario)
    if verbose:
        print(f"      Generated returns for {len(returns_df)} assets")
        print(f"      Mean return: {returns_df['return_1d'].mean():+.2f}%")
        print(f"      Std return: {returns_df['return_1d'].std():.2f}%")
    
    # Compute category stats
    if verbose:
        print("\n[4/5] Computing category statistics...")
    stats = compute_category_stats(returns_df, date)
    if verbose:
        print(f"      Computed {len(stats)} category stats")
    
    # Save to database
    if verbose:
        print("\n[5/5] Saving to database...")
    
    # Prepare price data
    price_df = returns_df[['ticker', 'return_1d', 'beta_predicted_return', 'alpha_1d']].copy()
    
    # Add mock additional fields
    price_df['return_1w'] = price_df['return_1d'] * np.random.uniform(2, 5, len(price_df))
    price_df['return_1m'] = price_df['return_1d'] * np.random.uniform(8, 20, len(price_df))
    price_df['return_ytd'] = price_df['return_1d'] * np.random.uniform(15, 40, len(price_df))
    price_df['z_score_1d'] = price_df['return_1d'] / price_df['return_1d'].std()
    price_df['percentile_60d'] = np.random.randint(1, 100, len(price_df))
    
    # Save
    prices_saved = save_daily_prices(price_df, date)
    stats_saved = save_category_stats(stats, date)
    factors_saved = save_factor_returns(factor_returns, date)
    
    if verbose:
        print(f"      Saved {prices_saved} price records")
        print(f"      Saved {stats_saved} category stats")
        print(f"      Saved {factors_saved} factor returns")
    
    result = {
        'date': date,
        'scenario': scenario,
        'assets': len(assets),
        'prices_saved': prices_saved,
        'stats_saved': stats_saved,
        'factors_saved': factors_saved,
        'factor_returns': factor_returns,
        'tier1_returns': {
            tier1: round(group['return_1d'].mean(), 4) 
            for tier1, group in returns_df.groupby('tier1')
        },
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("MOCK DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nTier-1 Returns:")
        for tier1, ret in sorted(result['tier1_returns'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {tier1}: {ret:+.2f}%")
    
    return result


def test_mock_data() -> bool:
    """Test that mock data was generated correctly."""
    print("\n" + "=" * 60)
    print("TESTING MOCK DATA")
    print("=" * 60)
    
    conn = get_db()
    cursor = conn.cursor()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Daily prices populated
    print("\n[TEST 1] Daily prices populated...")
    cursor.execute("SELECT COUNT(*) FROM daily_prices")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"         PASSED ({count} records)")
        tests_passed += 1
    else:
        print("         FAILED (no records)")
        tests_failed += 1
    
    # Test 2: Category stats populated
    print("\n[TEST 2] Category stats populated...")
    cursor.execute("SELECT COUNT(*) FROM category_stats")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"         PASSED ({count} records)")
        tests_passed += 1
    else:
        print("         FAILED (no records)")
        tests_failed += 1
    
    # Test 3: Factor returns populated
    print("\n[TEST 3] Factor returns populated...")
    cursor.execute("SELECT COUNT(*) FROM factor_returns")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"         PASSED ({count} records)")
        tests_passed += 1
    else:
        print("         FAILED (no records)")
        tests_failed += 1
    
    # Test 4: Tier-1 summary view works
    print("\n[TEST 4] Tier-1 summary view...")
    cursor.execute("SELECT * FROM v_tier1_summary")
    rows = cursor.fetchall()
    if len(rows) == 7:  # 7 Tier-1 categories
        print(f"         PASSED ({len(rows)} Tier-1 categories)")
        tests_passed += 1
    else:
        print(f"         FAILED (expected 7, got {len(rows)})")
        tests_failed += 1
    
    # Test 5: Returns are reasonable
    print("\n[TEST 5] Returns are reasonable...")
    cursor.execute("SELECT AVG(return_1d), MIN(return_1d), MAX(return_1d) FROM daily_prices")
    avg, min_ret, max_ret = cursor.fetchone()
    if -20 < avg < 20 and min_ret > -50 and max_ret < 50:
        print(f"         PASSED (avg={avg:.2f}%, range=[{min_ret:.2f}%, {max_ret:.2f}%])")
        tests_passed += 1
    else:
        print(f"         FAILED (unreasonable returns)")
        tests_failed += 1
    
    conn.close()
    
    print("\n" + "-" * 60)
    print(f"TESTS COMPLETED: {tests_passed} passed, {tests_failed} failed")
    print("-" * 60)
    
    return tests_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Generate mock Bloomberg data")
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help="Date to generate (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1,
                       help="Number of days to generate (counting back from date)")
    parser.add_argument("--scenario", type=str, default='normal',
                       choices=['normal', 'risk_on', 'risk_off', 'vol_spike', 'rotation'],
                       help="Market scenario")
    parser.add_argument("--test", action="store_true", help="Run tests after generation")
    args = parser.parse_args()
    
    # Generate data for each day
    base_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    all_results = []
    for i in range(args.days):
        date = (base_date - timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Vary scenarios slightly for multi-day generation
        if args.days > 1:
            scenario_weights = {'normal': 0.6, 'risk_on': 0.15, 'risk_off': 0.15, 
                              'vol_spike': 0.05, 'rotation': 0.05}
            scenario = np.random.choice(
                list(scenario_weights.keys()),
                p=list(scenario_weights.values())
            )
        else:
            scenario = args.scenario
        
        result = generate_mock_data(date, scenario, verbose=(args.days == 1))
        all_results.append(result)
    
    if args.days > 1:
        print(f"\n{'='*60}")
        print(f"GENERATED {args.days} DAYS OF MOCK DATA")
        print(f"{'='*60}")
        print(f"\nDate range: {all_results[-1]['date']} to {all_results[0]['date']}")
    
    # Run tests
    if args.test or args.days == 1:
        test_success = test_mock_data()
        if not test_success:
            print("\nWARNING: Some tests failed")
            return 1
    
    print("\n✓ Mock data generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
