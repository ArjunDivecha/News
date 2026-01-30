#!/usr/bin/env python3
"""
=============================================================================
DAILY REPORT GENERATOR
=============================================================================

INPUT FILES:
- database/market_data.db (daily_prices, category_stats, factor_returns)
- prompts/daily_wrap.md (prompt template)

OUTPUT FILES:
- outputs/daily/daily_wrap_YYYY-MM-DD.md (Markdown report)
- outputs/daily/daily_wrap_YYYY-MM-DD.pdf (PDF report) [if PDF enabled]
- Updates reports table in database

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Generate daily market wrap report using multiple LLMs in parallel.
Pulls data from SQLite, injects into prompt template, generates reports.

USAGE:
    python scripts/03_generate_daily_report.py                    # Today's date
    python scripts/03_generate_daily_report.py --date 2026-01-30
    python scripts/03_generate_daily_report.py --provider anthropic # Single provider
    python scripts/03_generate_daily_report.py --test              # Test mode (no LLM)
=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.db import (get_db, get_latest_prices, get_assets, 
                      get_tier1_distribution, get_tier2_distribution, save_report)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "daily"
DB_PATH = PROJECT_DIR / "database" / "market_data.db"


def load_prompt_template() -> tuple:
    """Load the daily wrap prompt template and split into system/user parts."""
    prompt_path = PROMPTS_DIR / "daily_wrap.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    # Split on "USER" marker
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
    else:
        system_prompt = content
        user_prompt = ""
    
    return system_prompt, user_prompt


def get_tier1_stats(date: str) -> str:
    """Get Tier-1 category statistics for the date."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT category_value, avg_return, median_return, std_return,
               min_return, max_return, count, percentile_60d, streak_days, streak_direction
        FROM category_stats
        WHERE date = ? AND category_type = 'tier1'
        ORDER BY avg_return DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No Tier-1 data available"
    
    lines = ["| Category | Avg Return | Median | Std | Min | Max | Count | 60d Pctl | Streak |"]
    lines.append("|----------|------------|--------|-----|-----|-----|-------|----------|--------|")
    
    for row in rows:
        streak_str = f"{row[8]:+d}" if row[8] else "0"
        lines.append(
            f"| {row[0]} | {row[1]:+.2f}% | {row[2]:+.2f}% | {row[3]:.2f}% | "
            f"{row[4]:+.2f}% | {row[5]:+.2f}% | {row[6]} | {row[7]}% | {streak_str} |"
        )
    
    return "\n".join(lines)


def get_tier2_stats(date: str, limit: int = 20) -> str:
    """Get Tier-2 category statistics (top and bottom performers)."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT category_value, avg_return, count, percentile_60d, streak_days
        FROM category_stats
        WHERE date = ? AND category_type = 'tier2'
        ORDER BY avg_return DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No Tier-2 data available"
    
    # Top 10 and Bottom 10
    top_10 = rows[:10]
    bottom_10 = rows[-10:][::-1]
    
    lines = ["**TOP 10 BY RETURN:**"]
    lines.append("| Strategy | Avg Return | Count | 60d Pctl | Streak |")
    lines.append("|----------|------------|-------|----------|--------|")
    
    for row in top_10:
        streak_str = f"{row[4]:+d}" if row[4] else "0"
        lines.append(f"| {row[0]} | {row[1]:+.2f}% | {row[2]} | {row[3]}% | {streak_str} |")
    
    lines.append("\n**BOTTOM 10 BY RETURN:**")
    lines.append("| Strategy | Avg Return | Count | 60d Pctl | Streak |")
    lines.append("|----------|------------|-------|----------|--------|")
    
    for row in bottom_10:
        streak_str = f"{row[4]:+d}" if row[4] else "0"
        lines.append(f"| {row[0]} | {row[1]:+.2f}% | {row[2]} | {row[3]}% | {streak_str} |")
    
    return "\n".join(lines)


def get_factor_returns(date: str) -> str:
    """Get factor returns for beta attribution."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT factor_name, return_1d
        FROM factor_returns
        WHERE date = ?
        ORDER BY return_1d DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No factor return data available"
    
    lines = ["| Factor | 1-Day Return |"]
    lines.append("|--------|--------------|")
    
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]:+.2f}% |")
    
    return "\n".join(lines)


def get_unusual_flags(date: str) -> str:
    """Identify unusual patterns in the data."""
    conn = get_db()
    
    # Find categories with extreme returns (|z-score| > 1.5)
    df = pd.read_sql_query("""
        SELECT category_type, category_value, avg_return, std_return, percentile_60d, streak_days
        FROM category_stats
        WHERE date = ?
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty:
        return "No unusual patterns detected (insufficient data)"
    
    flags = []
    
    # Check for extreme percentiles
    extremes = df[(df['percentile_60d'] <= 10) | (df['percentile_60d'] >= 90)]
    for _, row in extremes.iterrows():
        direction = "high" if row['percentile_60d'] >= 90 else "low"
        flags.append(f"- {row['category_value']} at {row['percentile_60d']}th percentile (historically {direction})")
    
    # Check for long streaks
    long_streaks = df[abs(df['streak_days']) >= 3]
    for _, row in long_streaks.iterrows():
        direction = "positive" if row['streak_days'] > 0 else "negative"
        flags.append(f"- {row['category_value']}: {abs(row['streak_days'])}-day {direction} streak")
    
    # Check for extreme returns
    if df['avg_return'].std() > 0:
        df['z_score'] = (df['avg_return'] - df['avg_return'].mean()) / df['avg_return'].std()
        outliers = df[abs(df['z_score']) > 1.5]
        for _, row in outliers.iterrows():
            direction = "outperforming" if row['z_score'] > 0 else "underperforming"
            flags.append(f"- {row['category_value']}: {direction} with z-score of {row['z_score']:.1f}")
    
    if not flags:
        return "No significant unusual patterns detected today"
    
    return "\n".join(flags[:10])  # Limit to top 10


def prepare_data_summary(date: str) -> Dict[str, str]:
    """Prepare all data summaries for prompt injection."""
    return {
        'date': date,
        'tier1_stats': get_tier1_stats(date),
        'tier2_stats': get_tier2_stats(date),
        'tier3_regional_stats': "(Regional breakdown computed from Tier-3 tags)",
        'tier3_sector_stats': "(Sector breakdown computed from Tier-3 tags)",
        'tier3_style_stats': "(Style factor breakdown computed from Tier-3 tags)",
        'tier3_strategy_stats': "(Strategy breakdown computed from Tier-3 tags)",
        'factor_returns': get_factor_returns(date),
        'streaks': "(Streak analysis from historical data)",
        'extremes': "(Historical extremes analysis)",
        'similar_days': "(Similar day pattern matching)",
        'regime_indicators': "(Current regime indicators)",
    }


def inject_data_into_prompt(user_prompt: str, data: Dict[str, str]) -> str:
    """Inject data summaries into the user prompt template."""
    result = user_prompt
    
    # Replace placeholders
    for key, value in data.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    
    # Add the unusual flags at the end
    result += f"\n\nUNUSUAL PATTERNS DETECTED:\n{data.get('unusual_flags', '')}"
    
    return result


def generate_daily_report(date: str, providers: List[str] = None,
                         test_mode: bool = False, verbose: bool = True) -> Dict:
    """
    Generate daily market wrap report.
    
    Args:
        date: Date string (YYYY-MM-DD)
        providers: List of LLM providers to use (default: all)
        test_mode: If True, skip LLM calls and return test output
        verbose: Print progress
        
    Returns:
        Dict with generation results
    """
    if providers is None:
        providers = ['anthropic']  # Default to Anthropic only for cost
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING DAILY REPORT FOR {date}")
        print(f"{'='*70}")
        print(f"Providers: {', '.join(providers)}")
    
    # Load prompt template
    if verbose:
        print("\n[1/5] Loading prompt template...")
    system_prompt, user_prompt_template = load_prompt_template()
    if verbose:
        print(f"      System prompt: {len(system_prompt)} chars")
        print(f"      User prompt template: {len(user_prompt_template)} chars")
    
    # Prepare data
    if verbose:
        print("\n[2/5] Preparing data summary...")
    data = prepare_data_summary(date)
    data['unusual_flags'] = get_unusual_flags(date)
    if verbose:
        print(f"      Tier-1 stats: {len(data['tier1_stats'])} chars")
        print(f"      Tier-2 stats: {len(data['tier2_stats'])} chars")
        print(f"      Factor returns: {len(data['factor_returns'])} chars")
    
    # Inject data into prompt
    if verbose:
        print("\n[3/5] Injecting data into prompt...")
    user_prompt = inject_data_into_prompt(user_prompt_template, data)
    if verbose:
        print(f"      Final user prompt: {len(user_prompt)} chars")
    
    # Generate reports
    results = {'date': date, 'providers': {}, 'data_summary': data}
    
    if test_mode:
        if verbose:
            print("\n[4/5] TEST MODE - Skipping LLM generation...")
        results['providers']['test'] = {
            'content': f"# TEST REPORT FOR {date}\n\nThis is a test report.\n\n" + 
                      f"Data Summary:\n{data['tier1_stats']}\n",
            'model': 'test',
            'provider': 'test',
            'time_ms': 0,
        }
    else:
        if verbose:
            print(f"\n[4/5] Generating reports from {len(providers)} provider(s)...")
        
        # Import LLM utilities
        from utils.llm import generate_parallel, generate_report as gen_single
        
        if len(providers) == 1:
            # Single provider
            provider = providers[0]
            if verbose:
                print(f"      Calling {provider}...")
            result = gen_single(system_prompt, user_prompt, provider, 'daily', 4000)
            results['providers'][provider] = result
            if verbose:
                if 'error' in result:
                    print(f"      ERROR: {result['error']}")
                else:
                    print(f"      Success: {len(result.get('content', ''))} chars, {result.get('time_ms', 0)}ms")
        else:
            # Multiple providers in parallel
            if verbose:
                print(f"      Calling {len(providers)} providers in parallel...")
            parallel_results = generate_parallel(system_prompt, user_prompt, providers, 'daily', 4000)
            results['providers'] = parallel_results
            for provider, result in parallel_results.items():
                if verbose:
                    if 'error' in result:
                        print(f"      [{provider}] ERROR: {result['error']}")
                    else:
                        print(f"      [{provider}] Success: {len(result.get('content', ''))} chars")
    
    # Save outputs
    if verbose:
        print("\n[5/5] Saving outputs...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for provider, result in results['providers'].items():
        if 'content' in result and result['content']:
            # Save markdown
            md_filename = f"daily_wrap_{date}_{provider}.md"
            md_path = OUTPUT_DIR / md_filename
            md_path.write_text(result['content'])
            saved_files.append(md_path)
            
            # Save to database
            report_id = f"daily_{date}_{provider}"
            save_report(
                report_id=report_id,
                report_type='daily',
                report_date=date,
                content_md=result['content'],
                model_name=result.get('model', 'unknown'),
                tokens_input=result.get('tokens_input'),
                tokens_output=result.get('tokens_output'),
                generation_time_ms=result.get('time_ms'),
            )
            
            if verbose:
                print(f"      Saved: {md_filename}")
    
    results['saved_files'] = [str(f) for f in saved_files]
    
    if verbose:
        print(f"\n{'='*70}")
        print("DAILY REPORT GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nReports saved to: {OUTPUT_DIR}")
        for f in saved_files:
            print(f"  - {f.name}")
    
    return results


def test_report_generation(date: str) -> bool:
    """Test the report generation pipeline."""
    print("\n" + "=" * 70)
    print("TESTING REPORT GENERATION")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Prompt template loads
    print("\n[TEST 1] Load prompt template...")
    try:
        system, user = load_prompt_template()
        if len(system) > 100 and len(user) > 100:
            print(f"         PASSED (system={len(system)} chars, user={len(user)} chars)")
            tests_passed += 1
        else:
            print("         FAILED (prompt too short)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Data preparation
    print("\n[TEST 2] Prepare data summary...")
    try:
        data = prepare_data_summary(date)
        if data['tier1_stats'] != "No Tier-1 data available":
            print(f"         PASSED (tier1={len(data['tier1_stats'])} chars)")
            tests_passed += 1
        else:
            print("         FAILED (no tier-1 data)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Prompt injection
    print("\n[TEST 3] Inject data into prompt...")
    try:
        _, user_template = load_prompt_template()
        data = prepare_data_summary(date)
        injected = inject_data_into_prompt(user_template, data)
        if "{date}" not in injected and date in injected:
            print(f"         PASSED (injected={len(injected)} chars)")
            tests_passed += 1
        else:
            print("         FAILED (placeholders not replaced)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Test mode generation
    print("\n[TEST 4] Test mode generation...")
    try:
        result = generate_daily_report(date, test_mode=True, verbose=False)
        if result['providers'].get('test', {}).get('content'):
            print(f"         PASSED (test report generated)")
            tests_passed += 1
        else:
            print("         FAILED (no test report)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Output file created
    print("\n[TEST 5] Output file created...")
    try:
        result = generate_daily_report(date, test_mode=True, verbose=False)
        if result['saved_files']:
            test_file = Path(result['saved_files'][0])
            if test_file.exists():
                print(f"         PASSED ({test_file.name})")
                tests_passed += 1
            else:
                print("         FAILED (file not found)")
                tests_failed += 1
        else:
            print("         FAILED (no files saved)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    print("\n" + "-" * 70)
    print(f"TESTS COMPLETED: {tests_passed} passed, {tests_failed} failed")
    print("-" * 70)
    
    return tests_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Generate daily market wrap report")
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help="Date to generate report for (YYYY-MM-DD)")
    parser.add_argument("--provider", type=str, nargs='+',
                       choices=['openai', 'anthropic', 'google'],
                       default=['anthropic'],
                       help="LLM provider(s) to use")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (skip LLM calls)")
    parser.add_argument("--run-tests", action="store_true",
                       help="Run pipeline tests")
    args = parser.parse_args()
    
    if args.run_tests:
        success = test_report_generation(args.date)
        return 0 if success else 1
    
    try:
        result = generate_daily_report(
            date=args.date,
            providers=args.provider,
            test_mode=args.test,
            verbose=True,
        )
        
        if any('error' in r for r in result['providers'].values()):
            print("\n⚠ Some providers failed")
            return 1
        
        print("\n✓ Daily report generation complete")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
