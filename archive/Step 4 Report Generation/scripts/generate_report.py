#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: generate_report.py
=============================================================================

DESCRIPTION:
    Generates professional daily market reports by orchestrating Bloomberg
    Excel data loading, meme/social signal ingestion, prompt template
    injection, and LLM report generation. The script resolves a target date
    (via --date or auto-detected last trading day), loads market data from an
    Excel spreadsheet into a SQLite database (daily_prices, assets,
    category_stats, factor_returns), optionally ingests a meme-stock social
    media snapshot JSON, reads a markdown or structured prompt template from
    disk, injects the data into the prompt, sends it to one or more LLM
    providers (anthropic, openai, google), and saves the resulting reports as
    markdown and/or structured JSON with optional PDF via PrinceXML. When
    multiple providers are used, a comparison_summary.txt is generated.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/prompts/daily_wrap.md
        Markdown prompt template for non-structured (markdown) report mode.
        Contains SYSTEM and USER sections split on the "USER" marker.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/prompts/daily_wrap_structured.md
        Structured JSON prompt template for --structured report mode.
        Contains SYSTEM and USER sections split on the "USER" marker.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/data/bloomberg_data_{date}.xlsx
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/data/Bloomberg_Data_{date}.xlsx
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/data/Bloomberg_Data_Template.xlsx
        Bloomberg market data Excel. The script searches candidates by date
        specificity, then falls back to Bloomberg_Data_Template.xlsx, then to
        the most recently modified .xlsx in the data directory. Read via
        pd.read_excel() using "Asset_Data" and "Factor_Returns" sheets.
    Meme/social snapshot JSON file (path configured via --meme-social-file CLI
        argument; when omitted, utils.db uses its own default lookup).
        Contains social media mention data for meme stocks, loaded via
        ingest_meme_social_snapshot().
    SQLite database (path managed by utils.db, typically in the project data
        directory). Stores daily_prices, assets, category_stats, factor_returns,
        and meme_social_snapshots tables. Accessed via get_db() which creates
        the connection.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/outputs/comparison/{date}/{provider}_report.json
        Structured report output as parsed JSON (generated only when
        --structured mode is active and utils.pdf_prince.parser is available).
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/outputs/comparison/{date}/{provider}_report.md
        Raw LLM response saved as markdown. Primary output in markdown mode;
        fallback when structured parsing fails in --structured mode.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/outputs/comparison/{date}/{provider}_report.pdf
        PDF report generated from structured JSON via PrinceXML (optional;
        requires utils.pdf_prince module and only attempted in --structured
        mode).
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation/outputs/comparison/{date}/comparison_summary.txt
        Quick-reference text file comparing model outputs by content length,
        token usage, generation time, and structured parsing success.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas (data manipulation, reading Excel via pd.read_excel)
    - numpy (data type coercion, NaN handling)
    - utils.db (local module: SQLite database CRUD for daily_prices, assets,
        category_stats, factor_returns, meme_social_snapshots)
    - utils.llm (local module: parallel/single LLM report generation via
        Anthropic, OpenAI, Google APIs with token tracking)
    - utils.pdf_prince.parser (local optional module: structured JSON
        extraction and validation from LLM responses)
    - utils.pdf_prince (local optional module: PDF rendering from structured
        data via PrinceXML)

USAGE:
    python generate_report.py
    python generate_report.py --date 2026-02-03 --models anthropic --structured
    python generate_report.py --date 2026-02-03 --models anthropic openai google --structured --skip-load-data

NOTES:
    - Requires a SQLite database initialized by utils.db with the expected
      schema (daily_prices, assets, category_stats, factor_returns tables).
    - LLM API keys must be available via environment variables or the
      utils.llm module's provider configuration.
    - The --structured flag (RECOMMENDED) produces validated JSON output;
      requires utils.pdf_prince.parser to be installed.
    - PDF generation requires PrinceXML installed on the system and the
      utils.pdf_prince module.
    - The meme/social snapshot path is user-configurable; if omitted,
      ingest_meme_social_snapshot() searches its default location.
=============================================================================
"""

import sys
import json
import argparse
import re
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

from utils.db import (
    get_db,
    save_daily_prices,
    save_category_stats,
    save_factor_returns,
    get_assets,
    ingest_meme_social_snapshot,
    get_meme_social_context,
)
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


def _is_missing(value: Any) -> bool:
    """Return True when a value is effectively missing for prompt formatting."""
    return value is None or pd.isna(value)


def _format_pct(value: Any) -> str:
    """Format a numeric percentage with sign, or N/A when unavailable."""
    if _is_missing(value):
        return "N/A"
    return f"{float(value):+.2f}%"


def _format_number(value: Any, decimals: int = 2) -> str:
    """Format a generic numeric field, or N/A when unavailable."""
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _format_joined(values: List[Any], limit: int = 4, fallback: str = "N/A") -> str:
    """Join a short list for prompt injection."""
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned[:limit]) if cleaned else fallback


def _format_signal_rankings(
    signals: List[Dict[str, Any]],
    key: str,
    limit: int = 3,
    reverse: bool = True,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> str:
    """Format a short ranked list of signals for prompt injection."""
    ranked = []
    for signal in signals:
        if _is_missing(signal.get(key)):
            continue
        value = float(signal.get(key) or 0)
        if minimum is not None and value < minimum:
            continue
        if maximum is not None and value > maximum:
            continue
        ranked.append(signal)
    ranked.sort(key=lambda signal: float(signal.get(key) or 0), reverse=reverse)
    if not ranked:
        return "N/A"
    return ", ".join(
        f"{signal.get('symbol', 'N/A')} {_format_pct(signal.get(key))}"
        for signal in ranked[:limit]
    )


def _format_most_mentioned(signals: List[Dict[str, Any]], limit: int = 3) -> str:
    """Format the most-mentioned names for prompt injection."""
    ranked = sorted(
        signals,
        key=lambda signal: (
            int(signal.get('mentions') or 0),
            float(signal.get('meme_score') or 0),
        ),
        reverse=True,
    )
    if not ranked:
        return "N/A"
    return ", ".join(
        f"{signal.get('symbol', 'N/A')} ({signal.get('mentions', 0)} mentions)"
        for signal in ranked[:limit]
    )


def format_tier1_stats(stats: List[Dict[str, Any]]) -> str:
    """Format Tier-1 stats for prompt injection."""
    lines = []
    for stat in stats:
        lines.append(
            f"  {stat['category_value']}: {_format_pct(stat.get('avg_return'))}, "
            f"YTD {_format_pct(stat.get('avg_return_ytd'))} "
            f"({stat.get('count', 0)} assets, best {stat.get('best_ticker', 'N/A')} "
            f"{_format_pct(stat.get('best_return'))}, worst {stat.get('worst_ticker', 'N/A')} "
            f"{_format_pct(stat.get('worst_return'))})"
        )
    return "\n".join(lines) if lines else "No data available"


def format_tier2_stats(stats: List[Dict[str, Any]]) -> str:
    """Format Tier-2 stats for prompt injection."""
    lines = []
    for stat in stats:
        lines.append(
            f"  {stat['category_value']}: {_format_pct(stat.get('avg_return'))}, "
            f"YTD {_format_pct(stat.get('avg_return_ytd'))} "
            f"({stat.get('count', 0)} assets)"
        )
    return "\n".join(lines) if lines else "No data available"


def format_factor_returns(factors: List[Dict[str, Any]]) -> str:
    """Format factor returns for prompt injection."""
    lines = [
        f"  {factor['factor_name']}: {_format_pct(factor.get('return_1d'))}"
        for factor in factors
    ]
    return "\n".join(lines) if lines else "No data available"


def format_meme_social_flow(context: Dict[str, Any], report_date: str) -> str:
    """Format meme/social context for the prompt with concrete signal detail."""
    status = context.get('status')
    if status == 'missing':
        return "No meme/social snapshot has been ingested. Treat this section as unavailable."
    if status == 'empty':
        return (
            f"Snapshot date: {context.get('snapshot_date', 'Unknown')}.\n"
            "A meme/social snapshot exists, but it contains no qualifying meme-stock rows."
        )

    freshness = str(context.get('freshness', 'unknown')).upper()
    top_signals = context.get('top_signals', [])
    lines = [
        "Section intent: write this as a standalone meme/social memo. Do not compare it to "
        "factor returns, sector rotation, theme performance, or any other part of the daily report.",
        f"Report date: {report_date}",
        f"Social data as of: {context.get('as_of_date', 'Unknown')}",
        f"Snapshot date: {context.get('snapshot_date', 'Unknown')}",
        f"Snapshot timestamp: {context.get('snapshot_ts', 'Unknown')}",
        (
            f"Freshness status: {freshness} "
            f"({context.get('business_days_old', 'N/A')} business days old, "
            f"{context.get('calendar_days_old', 'N/A')} calendar days old)"
        ),
        (
            f"Coverage: {context.get('total_rows', 0)} meme names, "
            f"{context.get('total_mentions', 0)} total mentions"
        ),
        (
            f"Aggregate tape: avg today {_format_pct(context.get('avg_price_change_1d'))}, "
            f"avg 5-day {_format_pct(context.get('avg_price_change_5d'))}, "
            f"avg volume spike {_format_number(context.get('avg_volume_spike'))}x"
        ),
        (
            "Signal flags: "
            f"{context.get('reason_flags', {}).get('volume_spike', 0)} volume-spike names, "
            f"{context.get('reason_flags', {}).get('one_day_move', 0)} 1-day extremes, "
            f"{context.get('reason_flags', {}).get('five_day_move', 0)} 5-day extremes"
        ),
    ]

    if top_signals:
        lines.append(f"Most mentioned names: {_format_most_mentioned(top_signals)}")
        lines.append(
            f"Strongest today: "
            f"{_format_signal_rankings(top_signals, 'price_change_1d', minimum=0)}"
        )
        lines.append(
            f"Strongest 5-day: "
            f"{_format_signal_rankings(top_signals, 'price_change_5d', minimum=0)}"
        )
        lines.append(
            f"Weakest 5-day: "
            f"{_format_signal_rankings(top_signals, 'price_change_5d', reverse=False, maximum=0)}"
        )

    source_summary = context.get('source_summary', [])
    if source_summary:
        source_str = ", ".join(
            f"{row['domain']} ({row['count']})"
            for row in source_summary[:6]
        )
        lines.append(f"Main platforms in the signal set: {source_str}")

    if top_signals:
        lines.append("Top social names to mention explicitly:")
        for signal in top_signals:
            reasons = _format_joined(signal.get('reasons', []), limit=3, fallback="no clear trigger captured")
            sources = _format_joined(signal.get('sources', []), limit=4)
            evidence = _format_joined(signal.get('sample_context', []), limit=2, fallback="no snippet available")
            lines.append(
                "  "
                f"{signal.get('symbol', 'N/A')} | {signal.get('name', 'Unknown')} | "
                f"mentions {signal.get('mentions', 0)} across {signal.get('source_count', 0)} sources | "
                f"today {_format_pct(signal.get('price_change_1d'))} | "
                f"5-day {_format_pct(signal.get('price_change_5d'))} | "
                f"volume {_format_number(signal.get('volume_spike'))}x | "
                f"score {_format_number(signal.get('meme_score'))} | "
                f"reasons: {reasons} | sources: {sources} | evidence: {evidence}"
            )

    if freshness in {'AGING', 'STALE'}:
        lines.append(
            "Freshness note: the social snapshot itself is not current as of today. The section "
            "should say that clearly and avoid treating it as a live signal."
        )

    return "\n".join(lines)


def prepare_data_summary(date: str) -> Dict:
    """Prepare data summary from database for the given date."""
    conn = get_db()
    
    # Get tier1 stats with YTD
    tier1_df = pd.read_sql_query("""
        SELECT 
            a.tier1 as category_value,
            AVG(dp.return_1d) as avg_return,
            AVG(dp.return_ytd) as avg_return_ytd,
            COUNT(*) as count,
            MAX(CASE WHEN dp.return_1d = (SELECT MAX(return_1d) FROM daily_prices dp2 JOIN assets a2 ON dp2.ticker = a2.ticker WHERE dp2.date = ? AND a2.tier1 = a.tier1) THEN dp.ticker END) as best_ticker,
            MAX(dp.return_1d) as best_return,
            MIN(CASE WHEN dp.return_1d = (SELECT MIN(return_1d) FROM daily_prices dp2 JOIN assets a2 ON dp2.ticker = a2.ticker WHERE dp2.date = ? AND a2.tier1 = a.tier1) THEN dp.ticker END) as worst_ticker,
            MIN(dp.return_1d) as worst_return
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ?
        GROUP BY a.tier1
        ORDER BY avg_return DESC
    """, conn, params=[date, date, date])
    
    tier1_stats = tier1_df.to_dict('records') if not tier1_df.empty else []
    
    # Get tier2 stats with YTD
    tier2_df = pd.read_sql_query("""
        SELECT 
            a.tier2 as category_value,
            AVG(dp.return_1d) as avg_return,
            AVG(dp.return_ytd) as avg_return_ytd,
            COUNT(*) as count,
            MAX(CASE WHEN dp.return_1d = (SELECT MAX(return_1d) FROM daily_prices dp2 JOIN assets a2 ON dp2.ticker = a2.ticker WHERE dp2.date = ? AND a2.tier2 = a.tier2) THEN dp.ticker END) as best_ticker,
            MAX(dp.return_1d) as best_return,
            MIN(CASE WHEN dp.return_1d = (SELECT MIN(return_1d) FROM daily_prices dp2 JOIN assets a2 ON dp2.ticker = a2.ticker WHERE dp2.date = ? AND a2.tier2 = a.tier2) THEN dp.ticker END) as worst_ticker,
            MIN(dp.return_1d) as worst_return
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ?
        GROUP BY a.tier2
        ORDER BY avg_return DESC
    """, conn, params=[date, date, date])
    
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

    meme_social_context = get_meme_social_context(date)
    
    return {
        'date': date,
        'tier1_stats': tier1_stats,
        'tier2_stats': tier2_stats,
        'factor_returns': factor_returns,
        'meme_social_context': meme_social_context,
    }


def inject_data_into_prompt(template: str, data: Dict) -> str:
    """Inject data summary into prompt template."""
    tier1_str = format_tier1_stats(data.get('tier1_stats', []))
    tier2_str = format_tier2_stats(data.get('tier2_stats', []))
    factor_str = format_factor_returns(data.get('factor_returns', []))
    meme_social_str = format_meme_social_flow(data.get('meme_social_context', {}), data.get('date', 'Unknown'))

    replacements = {
        'date': data.get('date', 'Unknown'),
        'tier1_stats': tier1_str,
        'tier2_stats': tier2_str,
        'factor_returns': factor_str,
        'meme_social_flow': meme_social_str,
    }

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{{{key.upper()}}}}}", str(value))
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        prompt = prompt.replace(f"{{{key.upper()}}}", str(value))
        prompt = prompt.replace(f"{{{key}}}", str(value))

    placeholder_pattern = re.compile(r'\{([a-zA-Z0-9_]+)\}')
    unresolved = {
        match.group(1)
        for match in placeholder_pattern.finditer(prompt)
    }
    for key in unresolved:
        prompt = prompt.replace(f"{{{key}}}", "No data available")

    if "{{" not in template and "{date}" not in template and "{tier1_stats}" not in template:
        data_summary = f"""
DATE: {data.get('date', 'Unknown')}

TIER-1 ASSET CLASS PERFORMANCE:
{tier1_str}

TIER-2 STRATEGY PERFORMANCE:
{tier2_str}

FACTOR RETURNS:
{factor_str}

MEME / SOCIAL FLOW:
{meme_social_str}
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
    
    # Handle new Excel format where first column header is the timestamp
    # and contains ticker values
    first_col = df.columns[0]
    if isinstance(first_col, (datetime, pd.Timestamp)) or (
        isinstance(first_col, str) and 
        ('-' in first_col or '/' in first_col) and  # Looks like a date string
        first_col not in ['Ticker', 'Name', 'Tier1', 'Tier2']  # Not a proper header
    ):
        # Rename the first column to 'ticker'
        df = df.rename(columns={first_col: 'ticker'})
        # Add date column with the extracted date from header
        if isinstance(first_col, (datetime, pd.Timestamp)):
            extracted_date = first_col.strftime('%Y-%m-%d')
        else:
            # Try to parse date from string
            try:
                extracted_date = pd.to_datetime(first_col).strftime('%Y-%m-%d')
            except:
                extracted_date = date
        df['date'] = extracted_date
    
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


def compare_models(
    date: str,
    providers: List[str],
    structured: bool = False,
    load_data: bool = True,
    ingest_meme_social: bool = True,
    meme_social_file: Optional[str] = None,
) -> Dict[str, Any]:
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
        print(f"\n[STEP 1/6] Loading Bloomberg data from Excel...")
        load_result = load_excel_data(date, verbose=True)
        if 'error' in load_result:
            print(f"\nERROR: Failed to load data - {load_result['error']}")
            print("Tip: Use --skip-load-data if data is already loaded")
            return {'error': load_result['error']}
        print(f"  ✓ Data loaded successfully")
    else:
        print(f"\n[STEP 1/6] Skipping data load (--skip-load-data flag set)")

    if ingest_meme_social:
        print("\n[STEP 2/6] Ingesting meme/social snapshot...")
        meme_result = ingest_meme_social_snapshot(meme_social_file)
        if meme_result.get('status') == 'missing_file':
            print(f"  WARNING: {meme_result.get('message')}")
        else:
            print(
                f"  Snapshot {meme_result.get('snapshot_date')} loaded from "
                f"{Path(meme_result.get('source_file')).name}"
            )
            print(
                f"  {meme_result.get('rows_loaded', 0)} rows, "
                f"{meme_result.get('matched_count', 0)} mapped to curated assets"
            )
    else:
        print("\n[STEP 2/6] Skipping meme/social ingestion (--skip-ingest-meme-social flag set)")

    meme_context = get_meme_social_context(date)
    if meme_context.get('status') == 'ok':
        print(
            f"  Meme/social freshness: {meme_context.get('freshness')} "
            f"({meme_context.get('business_days_old')} business days old)"
        )
        if meme_context.get('freshness') in {'aging', 'stale'}:
            print("  WARNING: Meme/social snapshot is stale for this report date")
    elif meme_context.get('status') in {'missing', 'empty'}:
        print(f"  WARNING: {meme_context.get('summary')}")

    # Load prompt template
    if structured:
        print("\n[STEP 3/6] Loading structured prompt template...")
        system_prompt, user_prompt_template = load_structured_prompt_template()
    else:
        print("\n[STEP 3/6] Loading markdown prompt template...")
        system_prompt, user_prompt_template = load_prompt_template()
    
    # Prepare data
    print("\n[STEP 4/6] Preparing data summary...")
    data = prepare_data_summary(date)
    data['unusual_flags'] = ""  # Simplified for comparison
    
    # Inject data into prompt
    print("\n[STEP 5/6] Injecting data into prompt...")
    user_prompt = inject_data_into_prompt(user_prompt_template, data)
    
    # Generate reports
    print(f"\n[STEP 6/6] Generating reports from {len(providers)} provider(s)...")
    
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
    parser.add_argument("--skip-ingest-meme-social", action="store_true",
                       help="Skip ingesting search_based_meme_stocks.json before report generation")
    parser.add_argument("--meme-social-file", type=str, default=None,
                       help="Optional path to a specific meme/social snapshot JSON file")
    args = parser.parse_args()
    
    load_data = not args.skip_load_data
    ingest_meme_social = not args.skip_ingest_meme_social
    
    # Determine date
    if args.date:
        # If date explicitly provided, use it directly (don't override)
        report_date = args.date
    else:
        # Only auto-detect last trading day if no date specified
        report_date = get_last_trading_day(datetime.now().strftime('%Y-%m-%d'))
    
    try:
        results = compare_models(
            report_date,
            args.models,
            args.structured,
            load_data,
            ingest_meme_social,
            args.meme_social_file,
        )
        
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
