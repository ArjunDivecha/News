#!/usr/bin/env python3
"""
=============================================================================
DAILY REPORT GENERATOR
=============================================================================

PURPOSE:
Generate professional daily market reports with automatic data loading.
This is the PRIMARY script for Phase 1 report generation.

USAGE:
    python scripts/05_compare_models.py --date 2026-02-03 --models anthropic --structured
    python scripts/05_compare_models.py --date 2026-02-03 --models anthropic --structured --skip-load-data

OUTPUT:
    outputs/comparison/{date}/
      - {model}_report.json (structured data)
      - {model}_report.md (markdown fallback)
      - {model}_report.pdf (PDF via PrinceXML)
      - comparison_summary.txt (quick comparison)
=============================================================================
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
sys.path.insert(0, str(SCRIPT_DIR))

from utils.db import get_db, save_daily_prices, save_category_stats, save_factor_returns, get_assets
from utils.llm import generate_parallel, generate_report as gen_single

# Import structured parser
try:
    from utils.pdf_prince.parser import parse_structured_report, extract_json_from_response
    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    print("WARNING: Structured parser not available")


OUTPUT_DIR = PROJECT_DIR / "outputs" / "comparison"


# =============================================================================
# HELPER FUNCTIONS (previously in 03_generate_daily_report.py)
# =============================================================================

def get_last_trading_day(target_date: Optional[str] = None) -> str:
    """Get the last trading day with data in the database."""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Check if weekend
    if dt.weekday() == 5:  # Saturday
        dt = dt - pd.Timedelta(days=1)
    elif dt.weekday() == 6:  # Sunday
        dt = dt - pd.Timedelta(days=2)
    
    # Check if we have data for this date
    conn = get_db()
    cursor = conn.cursor()
    
    for i in range(10):
        check_date = (dt - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE date = ?", (check_date,))
        count = cursor.fetchone()[0]
        
        if count > 0:
            conn.close()
            return check_date
    
    conn.close()
    return dt.strftime('%Y-%m-%d')


def load_prompt_template() -> tuple:
    """Load markdown prompt template."""
    prompt_path = PROJECT_DIR / "prompts" / "daily_wrap.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
    else:
        system_prompt = content
        user_prompt = ""
    
    return system_prompt, user_prompt


def prepare_data_summary(date: str) -> Dict:
    """Prepare data summary from database for the given date."""
    conn = get_db()
    
    # Get tier1 stats
    tier1_df = pd.read_sql_query("""
        SELECT category_value, avg_return, count, best_ticker, best_return, worst_ticker, worst_return
        FROM category_stats
        WHERE date = ? AND category_type = 'tier1'
        ORDER BY avg_return DESC
    """, conn, params=[date])
    
    tier1_stats = tier1_df.to_dict('records') if not tier1_df.empty else []
    
    # Get tier2 stats
    tier2_df = pd.read_sql_query("""
        SELECT category_value, avg_return, count, best_ticker, best_return, worst_ticker, worst_return
        FROM category_stats
        WHERE date = ? AND category_type = 'tier2'
        ORDER BY avg_return DESC
    """, conn, params=[date])
    
    tier2_stats = tier2_df.to_dict('records') if not tier2_df.empty else []
    
    # Get factor returns
    factor_df = pd.read_sql_query("""
        SELECT factor_name, return_1d
        FROM factor_returns
        WHERE date = ?
        ORDER BY return_1d DESC
    """, conn, params=[date])
    
    factor_returns = factor_df.to_dict('records') if not factor_df.empty else []
    
    conn.close()
    
    return {
        'date': date,
        'tier1_stats': tier1_stats,
        'tier2_stats': tier2_stats,
        'factor_returns': factor_returns,
    }


def inject_data_into_prompt(template: str, data: Dict) -> str:
    """Inject data summary into prompt template."""
    # Format tier1 stats
    tier1_lines = []
    for stat in data.get('tier1_stats', []):
        tier1_lines.append(
            f"  {stat['category_value']}: {stat['avg_return']:+.2f}% "
            f"({stat['count']} assets, best: {stat.get('best_ticker', 'N/A')} {stat.get('best_return', 0):+.2f}%, "
            f"worst: {stat.get('worst_ticker', 'N/A')} {stat.get('worst_return', 0):+.2f}%)"
        )
    tier1_str = "\n".join(tier1_lines) if tier1_lines else "No data available"
    
    # Format tier2 stats
    tier2_lines = []
    for stat in data.get('tier2_stats', []):
        tier2_lines.append(
            f"  {stat['category_value']}: {stat['avg_return']:+.2f}% ({stat['count']} assets)"
        )
    tier2_str = "\n".join(tier2_lines) if tier2_lines else "No data available"
    
    # Format factor returns
    factor_lines = []
    for factor in data.get('factor_returns', []):
        factor_lines.append(f"  {factor['factor_name']}: {factor['return_1d']:+.2f}%")
    factor_str = "\n".join(factor_lines) if factor_lines else "No data available"
    
    # Inject into template
    prompt = template
    prompt = prompt.replace("{{DATE}}", data.get('date', 'Unknown'))
    prompt = prompt.replace("{{TIER1_STATS}}", tier1_str)
    prompt = prompt.replace("{{TIER2_STATS}}", tier2_str)
    prompt = prompt.replace("{{FACTOR_RETURNS}}", factor_str)
    
    # Add data summary section if template doesn't have placeholders
    if "{{" not in template:
        data_summary = f"""
DATE: {data.get('date', 'Unknown')}

TIER-1 ASSET CLASS PERFORMANCE:
{tier1_str}

TIER-2 STRATEGY PERFORMANCE:
{tier2_str}

FACTOR RETURNS:
{factor_str}
"""
        prompt = prompt + "\n\n" + data_summary
    
    return prompt


# Column mapping for Excel data
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
        candidates = [
            DATA_DIR / f"bloomberg_data_{date}.xlsx",
            DATA_DIR / f"Bloomberg_Data_{date}.xlsx",
        ]
        for path in candidates:
            if path.exists():
                return path
    
    template = DATA_DIR / "Bloomberg_Data_Template.xlsx"
    if template.exists():
        return template
    
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))
    if xlsx_files:
        return max(xlsx_files, key=lambda p: p.stat().st_mtime)
    
    return None


def load_excel_data(date: str, verbose: bool = True) -> Dict:
    """Load Bloomberg data from Excel into database."""
    if verbose:
        print(f"\n[DATA LOADING] Loading Bloomberg Excel data for {date}...")
    
    excel_path = find_excel_file(date)
    if excel_path is None or not excel_path.exists():
        error = f"No Excel file found for date {date}"
        if verbose:
            print(f"  ERROR: {error}")
        return {'error': error}
    
    if verbose:
        print(f"  Found: {excel_path.name}")
    
    # Load asset data
    try:
        df = pd.read_excel(excel_path, sheet_name="Asset_Data")
    except:
        df = pd.read_excel(excel_path, sheet_name=0)
    
    df = df.rename(columns=COLUMN_MAP)
    
    # Clean error values
    error_values = ['#N/A', '#N/A N/A', '#NAME?', '#REF!', '#VALUE!', '#DIV/0!']
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(error_values, np.nan)
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.startswith('#') else x)
    
    # Convert numeric columns
    numeric_cols = ['price', 'return_1d', 'return_5d', 'return_1m', 'return_ytd', 'return_1y',
                    'rsi_14', 'volatility_30d', 'volatility_240d']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Load factor returns
    factor_returns = {}
    try:
        factor_df = pd.read_excel(excel_path, sheet_name="Factor_Returns")
        for _, row in factor_df.iterrows():
            factor_name = row.get('Factor_Name')
            return_1d = row.get('Chg_1D')
            if pd.notna(factor_name) and pd.notna(return_1d):
                try:
                    factor_returns[factor_name] = float(return_1d)
                except (TypeError, ValueError):
                    pass
    except:
        pass
    
    # Save to database
    prices_saved = save_daily_prices(df, date)
    factors_saved = save_factor_returns(factor_returns, date) if factor_returns else 0
    
    # Compute category stats
    assets = get_assets()
    stats = []
    merged = df.copy()
    if 'tier1' not in merged.columns or 'tier2' not in merged.columns:
        merged = df.merge(assets[['ticker', 'tier1', 'tier2']], on='ticker', how='left')
    
    if 'return_1d' in merged.columns:
        for tier1, group in merged.groupby('tier1'):
            if pd.isna(tier1):
                continue
            returns = group['return_1d'].dropna()
            if len(returns) > 0:
                stats.append({
                    'category_type': 'tier1',
                    'category_value': tier1,
                    'count': len(group),
                    'avg_return': round(float(returns.mean()), 4),
                    'median_return': round(float(returns.median()), 4),
                    'std_return': round(float(returns.std()), 4) if len(returns) > 1 else 0,
                    'min_return': round(float(returns.min()), 4),
                    'max_return': round(float(returns.max()), 4),
                    'best_ticker': group.loc[returns.idxmax(), 'ticker'],
                    'best_return': round(float(returns.max()), 4),
                    'worst_ticker': group.loc[returns.idxmin(), 'ticker'],
                    'worst_return': round(float(returns.min()), 4),
                    'percentile_60d': None,
                    'streak_days': None,
                    'streak_direction': None,
                })
        
        for tier2, group in merged.groupby('tier2'):
            if pd.isna(tier2):
                continue
            returns = group['return_1d'].dropna()
            if len(returns) > 0:
                stats.append({
                    'category_type': 'tier2',
                    'category_value': tier2,
                    'count': len(group),
                    'avg_return': round(float(returns.mean()), 4),
                    'median_return': round(float(returns.median()), 4),
                    'std_return': round(float(returns.std()), 4) if len(returns) > 1 else 0,
                    'min_return': round(float(returns.min()), 4),
                    'max_return': round(float(returns.max()), 4),
                    'best_ticker': group.loc[returns.idxmax(), 'ticker'],
                    'best_return': round(float(returns.max()), 4),
                    'worst_ticker': group.loc[returns.idxmin(), 'ticker'],
                    'worst_return': round(float(returns.min()), 4),
                    'percentile_60d': None,
                    'streak_days': None,
                    'streak_direction': None,
                })
    
    stats_saved = save_category_stats(stats, date) if stats else 0
    
    if verbose:
        print(f"  Loaded {prices_saved} prices, {factors_saved} factors, {stats_saved} category stats")
    
    return {
        'date': date,
        'prices_saved': prices_saved,
        'factors_saved': factors_saved,
        'stats_saved': stats_saved,
    }


def load_structured_prompt_template() -> tuple:
    """Load structured prompt template."""
    prompt_path = PROJECT_DIR / "prompts" / "daily_wrap_structured.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Structured prompt not found: {prompt_path}")
    
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


def save_structured_output(provider: str, date: str, llm_response: str, 
                          output_dir: Path) -> Dict[str, Path]:
    """
    Save structured output from LLM.
    
    Returns:
        Dict with paths to saved files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # Try to extract JSON
    if STRUCTURED_AVAILABLE:
        try:
            data = parse_structured_report(llm_response)
            data['report_date'] = datetime.strptime(date, '%Y-%m-%d').strftime('%B %d, %Y')
            
            # Save JSON
            json_path = output_dir / f"{provider}_report.json"
            json_path.write_text(json.dumps(data, indent=2))
            saved_files['json'] = json_path
            
            # Try to generate PDF with PrinceXML
            try:
                from utils.pdf_prince import convert_from_data
                pdf_path = output_dir / f"{provider}_report.pdf"
                result = convert_from_data(data, str(pdf_path))
                if result:
                    saved_files['pdf'] = Path(result)
            except Exception as e:
                print(f"  WARNING: Could not generate PDF: {e}")
                
        except Exception as e:
            print(f"  WARNING: Could not parse structured output: {e}")
            print(f"  Saving raw response as markdown...")
    
    # Save raw response as markdown fallback
    md_path = output_dir / f"{provider}_report.md"
    md_path.write_text(llm_response)
    saved_files['md'] = md_path
    
    return saved_files


def generate_comparison_summary(date: str, results: Dict[str, Dict], output_dir: Path):
    """Generate a summary file comparing all models."""
    summary_path = output_dir / "comparison_summary.txt"
    
    lines = [
        "=" * 70,
        f"MODEL COMPARISON SUMMARY - {date}",
        "=" * 70,
        "",
    ]
    
    for provider, result in results.items():
        if 'error' in result:
            lines.append(f"{provider.upper()}: ERROR - {result['error']}")
        else:
            content_len = len(result.get('content', ''))
            model = result.get('model', 'unknown')
            tokens_in = result.get('tokens_input') or 0
            tokens_out = result.get('tokens_output') or 0
            time_ms = result.get('time_ms') or 0
            
            lines.append(f"{provider.upper()} ({model}):")
            lines.append(f"  Content: {content_len:,} chars")
            lines.append(f"  Tokens: {tokens_in:,} in / {tokens_out:,} out")
            lines.append(f"  Time: {time_ms:,}ms ({time_ms/1000:.1f}s)")
            
            # Check if structured JSON was extracted
            if STRUCTURED_AVAILABLE:
                try:
                    data = extract_json_from_response(result['content'])
                    if data:
                        sections = len(data.get('sections', []))
                        tables = sum(len(s.get('tables', [])) for s in data.get('sections', []))
                        lines.append(f"  Structured: ✓ ({sections} sections, {tables} tables)")
                    else:
                        lines.append(f"  Structured: ✗ (could not extract JSON)")
                except:
                    lines.append(f"  Structured: ✗ (parse error)")
            
            lines.append("")
    
    lines.append("=" * 70)
    lines.append("FILES SAVED:")
    for provider, result in results.items():
        if 'saved_files' in result:
            for file_type, path in result['saved_files'].items():
                lines.append(f"  {provider}/{file_type}: {path.name}")
    
    summary_path.write_text("\n".join(lines))
    print(f"\nComparison summary saved: {summary_path}")


def compare_models(date: str, providers: List[str], structured: bool = False, load_data: bool = True) -> Dict[str, Any]:
    """
    Generate reports with multiple models for comparison.
    
    Args:
        date: Date string (YYYY-MM-DD)
        providers: List of provider names
        structured: If True, use structured JSON output mode
        
    Returns:
        Dict with results from each provider
    """
    print(f"\n{'='*70}")
    print(f"DAILY REPORT GENERATION FOR {date}")
    print(f"{'='*70}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Mode: {'Structured JSON' if structured else 'Markdown'}")
    
    # Load data if requested
    if load_data:
        print(f"\n[STEP 1/5] Loading Bloomberg data from Excel...")
        load_result = load_excel_data(date, verbose=True)
        if 'error' in load_result:
            print(f"\nERROR: Failed to load data - {load_result['error']}")
            print("Tip: Use --skip-load-data if data is already loaded")
            return {'error': load_result['error']}
        print(f"  ✓ Data loaded successfully")
    else:
        print(f"\n[STEP 1/5] Skipping data load (--skip-load-data flag set)")
    
    # Load prompt template
    if structured:
        print("\n[STEP 2/5] Loading structured prompt template...")
        system_prompt, user_prompt_template = load_structured_prompt_template()
    else:
        print("\n[STEP 2/5] Loading markdown prompt template...")
        system_prompt, user_prompt_template = load_prompt_template()
    
    # Prepare data
    print("\n[STEP 3/5] Preparing data summary...")
    data = prepare_data_summary(date)
    data['unusual_flags'] = ""  # Simplified for comparison
    
    # Inject data into prompt
    print("\n[STEP 4/5] Injecting data into prompt...")
    user_prompt = inject_data_into_prompt(user_prompt_template, data)
    
    # Generate reports
    print(f"\n[STEP 5/5] Generating reports from {len(providers)} provider(s)...")
    
    results = {}
    if len(providers) == 1:
        provider = providers[0]
        print(f"  Calling {provider}...")
        result = gen_single(system_prompt, user_prompt, provider, 'daily', 12000)
        results[provider] = result
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Success: {len(result.get('content', ''))} chars")
    else:
        print(f"  Calling {len(providers)} providers in parallel...")
        parallel_results = generate_parallel(system_prompt, user_prompt, providers, 'daily', 12000)
        results = parallel_results
        for provider, result in parallel_results.items():
            if 'error' in result:
                print(f"  [{provider}] ERROR: {result['error']}")
            else:
                print(f"  [{provider}] Success: {len(result.get('content', ''))} chars")
    
    # Save outputs
    print("\nSaving outputs...")
    output_dir = OUTPUT_DIR / date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for provider, result in results.items():
        if 'content' in result and result['content']:
            print(f"  Saving {provider} output...")
            if structured:
                saved_files = save_structured_output(provider, date, result['content'], output_dir)
            else:
                # Save as markdown
                md_path = output_dir / f"{provider}_report.md"
                md_path.write_text(result['content'])
                saved_files = {'md': md_path}
            
            result['saved_files'] = saved_files
    
    # Generate comparison summary
    generate_comparison_summary(date, results, output_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare multiple LLM models")
    parser.add_argument("--date", type=str, default=None,
                       help="Date (YYYY-MM-DD). Defaults to last trading day.")
    parser.add_argument("--models", type=str, nargs='+',
                       default=['anthropic', 'openai', 'google'],
                       choices=['anthropic', 'openai', 'google'],
                       help="Models to compare")
    parser.add_argument("--structured", action="store_true",
                       help="Use structured JSON output mode (RECOMMENDED)")
    parser.add_argument("--skip-load-data", action="store_true",
                       help="Skip loading Excel data (use if data already loaded)")
    args = parser.parse_args()
    
    load_data = not args.skip_load_data
    
    # Determine date
    if args.date:
        # If date explicitly provided, use it directly (don't override)
        report_date = args.date
    else:
        # Only auto-detect last trading day if no date specified
        report_date = get_last_trading_day(datetime.now().strftime('%Y-%m-%d'))
    
    try:
        results = compare_models(report_date, args.models, args.structured, load_data)
        
        # Check for errors
        errors = [p for p, r in results.items() if 'error' in r]
        if errors:
            print(f"\n⚠ Some providers failed: {', '.join(errors)}")
            return 1
        
        print("\n✓ Report generation complete")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
